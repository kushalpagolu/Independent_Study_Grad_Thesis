import gym
import numpy as np
from gym import spaces
import os
import logging
import time
import threading
from datetime import datetime
from stream_data import EmotivStreamer
from model_utils import load_or_create_model  # Updated import for model utilities
from drone_control import TelloController  # Updated import for TelloController
from lstm_handler import LSTMHandler  # Import the LSTMHandler class
import signal
import select
import sys
import queue
import random
import torch
from shared_events import stop_saving_thread, stop_input_listener, stop_main_loop
# === put near the top of learning_rlagent.py (after imports) =============
import queue, threading


# =========================================================================

# -----------------------------------------------------------------------------
# CONSTANTS & LOGGING
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = os.path.join(BASE_DIR, "models", "drone_rl_eeg_bandit")
ACTION_DELAY = 5                       # Seconds between actions
HUMAN_FEEDBACK_TIMEOUT = 10            # Timeout for user feedback (seconds)
MAX_SPEED = 20                         # Maximum speed percentage
EXPECTED_OBSERVATION_SIZE = 13          # Observation vector length
REPEAT_ACTION_PENALTY = 1.0            # Penalty for repeating same discrete action
MAX_REPEAT_COUNT = 3                   # How many repeats are allowed before penalty stacks further

# Timestamped run‑specific log file
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)
log_filename = os.path.join(logs_dir, f"rl_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



# -----------------------------------------------------------------------------
# ENVIRONMENT CLASS
# -----------------------------------------------------------------------------
class DroneControlEnv(gym.Env):
    """Custom Gym environment wrapping real / simulated Tello control for SAC."""

    def __init__(self, connect_drone: bool = False, max_speed: int = MAX_SPEED):
        super().__init__()
        self.logger = logger
        self.ready_for_fresh_data = False  # Flag to indicate if fresh data is ready
        self.reward_history = []

        # Observation is last 5 numeric outputs from the pipeline (can be tuned)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(EXPECTED_OBSERVATION_SIZE,), dtype=np.float32)

        # Action: [discrete_action, cont1, cont2, cont3, cont4]
        #   discrete_action ∈ {0 … 8}
        #   continuous       ∈ [‑1, 1]
        self.action_space = spaces.Box(
            low=np.array([0,  -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([8,  1.0,  1.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

        # State & misc
        self.current_state         = np.zeros(EXPECTED_OBSERVATION_SIZE, dtype=np.float32)
        self.connect_drone         = connect_drone
        self.max_speed             = max_speed
        self.has_taken_off         = False
        self.last_action_time      = 0.0
        self.prev_discrete_action  = None
        self.repeat_count          = 0

        # Interfaces
        self.eeg_processor   = EmotivStreamer()
        self.lstm_handler    = LSTMHandler()
        self.model           = None  # Populated externally via model_utils

        # Drone interface (optional)
        self.drone_controller = TelloController() if self.connect_drone else None
        self.drone_connected  = False
        if self.connect_drone:
            self._connect_drone_controller()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _connect_drone_controller(self):
        """Attempt to open a Tello connection."""
        if self.drone_controller and not self.drone_connected:
            self.drone_connected = self.drone_controller.connect()
            if self.drone_connected:
                self.logger.info("Drone connected successfully.")
            else:
                self.logger.error("Failed to connect to the drone – falling back to simulation mode.")

    def _takeoff_if_needed(self):
        if self.drone_connected and not self.has_taken_off:
            try:
                self.logger.info("Taking off…")
                self.drone_controller.takeoff()
                self.has_taken_off = True
            except Exception as e:
                self.logger.error(f"Takeoff failed: {e}")

    def _land_if_needed(self):
        if self.drone_connected and self.has_taken_off:
            try:
                self.logger.info("Landing…")
                self.drone_controller.land()
            except Exception as e:
                self.logger.error(f"Landing failed: {e}")
            finally:
                self.has_taken_off = False

    # ------------------------------------------------------------------
    # Core Gym API
    # ------------------------------------------------------------------
    def step(self, action):
        """
        Executes a step in the environment.
        Args:
            action (np.ndarray): Flattened action array.
                - action[0]: Discrete action (0-8)
                - action[1:]: Continuous actions (4 values for velocities)
        Returns:
            tuple: (state, reward, done, info)
        """
        reward = 0
        done = False # Default to False, an episode is not done until explicitly stated.
        info = {}

        # Ensure the drone is connected and has taken off
        if self.connect_drone and not self.drone_connected:
            self.logger.error("Drone is not connected. Simulating step")

        # --- First Action: Takeoff ---
        if not self.has_taken_off and self.drone_connected:
            self.logger.info("First action detected. Overriding to 'Takeoff'.")
            if self.drone_controller:
                action = np.array([5, 0, 0, 0, 0])
                reward += 10
                self.takeoff_drone()
                if not self.has_taken_off:
                    self.logger.error("Takeoff failed.")
                    # Even if takeoff fails, the episode is not over. The agent can try again.
                    return self.current_state, reward, False, {"info": "Takeoff failed"}
            else:
                self.logger.error("Drone controller not initialized. Cannot take off.")
                return self.current_state, reward, False, {"info": "Drone controller not initialized"}

        # Validate the action
        if not isinstance(action, np.ndarray) or action.shape[0] != 5:
            self.logger.error(f"Invalid action format: {action}")
            return self.current_state, reward, False, {"info": "Invalid action"} # FIX: done is False

        # Get the discrete action for repetition checking
        discrete_action = int(np.clip(round(action[0]), 0, 8))

        # Check for and penalize repetitive actions
        if discrete_action == self.prev_discrete_action:
            self.repeat_count += 1
            if self.repeat_count > MAX_REPEAT_COUNT:
                penalty = REPEAT_ACTION_PENALTY * (self.repeat_count - MAX_REPEAT_COUNT)
                reward -= penalty
                self.logger.warning(f"Penalizing repetitive action '{discrete_action}' with -{penalty:.2f} reward.")
        else:
            self.repeat_count = 0 # Reset counter if action is different

        self.prev_discrete_action = discrete_action # Update the last action
        # Map the action to a command
        command = self._map_action_to_command(action)
        self.logger.info(f"Mapped Command: {command['command']}, Velocities: {command['velocities']}")

        # Get human feedback
        get_human_feedback = self._get_human_feedback()
        if get_human_feedback == "timeout":
            reward -= 0.02  # Small penalty for waiting
            self.logger.info("No user input received. Continuing.")
            self.ready_for_fresh_data = True
            # FIX: A timeout does not end the episode.
            return self.current_state, reward, False, {"info": "User thinking processing next prediction"}

        elif get_human_feedback == "no":
            self.logger.info("User input was 'n'. Action rejected.")
            self.ready_for_fresh_data = True
            # This was already correct. An episode continues after rejection.
            return self.current_state, -2, False, {"info": "Action rejected by user"}

        elif get_human_feedback == "yes":
            self.logger.info("Action approved. Proceeding with movement command.")
            reward += 7  # Positive reward for successful action
            current_time = time.time()
            if current_time - self.last_action_time >= ACTION_DELAY:
                self.last_action_time = current_time
                if self.connect_drone and self.drone_connected:
                    try:
                        self.drone_controller.send_rc_control(
                            command["velocities"]["left_right_velocity"],
                            command["velocities"]["forward_backward_velocity"],
                            command["velocities"]["up_down_velocity"],
                            command["velocities"]["yaw_velocity"]
                        )
                        self.logger.info(f"Drone action executed: {command['command']}")
                        self.ready_for_fresh_data = True
                        # FIX: A successful action does not end the episode.
                        return self.current_state, reward, False, {"info": "Action executed successfully"}
                    except Exception as e:
                        self.logger.error(f"Error sending RC control: {e}")
                        self.ready_for_fresh_data = True
                        # FIX: A failed command does not end the episode.
                        return self.current_state, -1, False, {"info": "Drone command failed"}
                else:
                    self.logger.info(f"Simulating action: {command['command']}")
                    self.current_state = self._get_simulated_state(command)
                    self.ready_for_fresh_data = True
                    # FIX: A simulated command does not end the episode.
                    return self.current_state, reward, False, {"info": "Drone command simulated"}
            else:
                self.logger.info(f"Action delayed. Remaining time: {ACTION_DELAY - (current_time - self.last_action_time):.2f} seconds")
                reward -= 0.02
                self.ready_for_fresh_data = True
                # FIX: An action delay does not end the episode.
                return self.current_state, reward, False, {"info": "Action delayed"}

        elif get_human_feedback == "stop":
            self.logger.info("User requested to stop. Exiting the environment.")
            reward = 0
            done = True  # THIS IS CORRECT. 'stop' is a terminal state.
            info = {"info": "User requested stop"}
            self.ready_for_fresh_data = True
            return self.current_state, reward, done, info

        # Fallback return
        self.ready_for_fresh_data = True
        return self.current_state, reward, done, info
    

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = np.zeros(EXPECTED_OBSERVATION_SIZE, dtype=np.float32)
        self.prev_discrete_action = None
        self.repeat_count = 0
        return self.current_state.copy(), {}

    def close(self):
        self._land_if_needed()
        if self.drone_controller and self.drone_connected:
            self.drone_controller.tello.end()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _map_action_to_command(self, action: np.ndarray):
        """Translate the full action vector into velocity commands."""
        # --- FIX: Unpack the action vector correctly ---
        discrete_action = int(np.clip(round(action[0]), 0, 8))
        cont = action[1:]
        # --- END OF FIX ---
        
        scaled = (cont * 100).astype(int)  # map [-1,1] → [-100,100]
        lr, fb, ud, yaw = scaled

        cmd = { "command": "Hover", "velocities": {
            "left_right_velocity": 0,
            "forward_backward_velocity": 0,
            "up_down_velocity": 0,
            "yaw_velocity": 0
        }}

        if discrete_action == 0:
            pass  # hover
        elif discrete_action == 1:
            cmd["command"] = "Move Forward";   cmd['velocities']['forward_backward_velocity'] = max(20, fb)
        elif discrete_action == 2:
            cmd["command"] = "Move Backward";  cmd['velocities']['forward_backward_velocity'] = min(-20, fb)
        elif discrete_action == 3:
            cmd["command"] = "Move Left";      cmd['velocities']['left_right_velocity']  = min(-20, lr)
        elif discrete_action == 4:
            cmd["command"] = "Move Right";     cmd['velocities']['left_right_velocity']  = max(20, lr)
        elif discrete_action == 5:
            cmd["command"] = "Ascend";         cmd['velocities']['up_down_velocity']      = max(20, ud)
        elif discrete_action == 6:
            cmd["command"] = "Descend";        cmd['velocities']['up_down_velocity']      = min(-20, ud)
        elif discrete_action == 7:
            cmd["command"] = "Rotate Left";   cmd['velocities']['yaw_velocity']          = min(-20, yaw)
        elif discrete_action == 8:
            cmd["command"] = "Rotate Right";  cmd['velocities']['yaw_velocity']          = max(20, yaw )
        return cmd

    # ------------------------------------------------------------------
    # Simulation utilities
    # ------------------------------------------------------------------
    def _simulate_state_transition(self, command: dict):
        """Very light physics‑free simulation – can be expanded."""
        noise = np.random.normal(0, 0.01, size=EXPECTED_OBSERVATION_SIZE)
        vel   = command['velocities']
        return np.array([
            vel['left_right_velocity']      / 100.0,
            vel['forward_backward_velocity'] / 100.0,
            vel['up_down_velocity']          / 100.0,
            vel['yaw_velocity']              / 100.0,
            np.random.rand()                # dummy sensor value / battery etc.
        ], dtype=np.float32) + noise

    # ------------------------------------------------------------------
    # Human feedback (blocking call with timeout)
    # ------------------------------------------------------------------
    def _get_human_feedback(self):
        """
        Blocks up to HUMAN_FEEDBACK_TIMEOUT waiting for 'y' / 'n' / 's'
        pushed by the keyboard listener running in the main thread.
        """
        print("Enter 'y' to approve, 'n' to reject, or 's' to stop: ", end='', flush=True)

        # Wait 7 seconds for user input
        rlist, _, _ = select.select([sys.stdin], [], [], HUMAN_FEEDBACK_TIMEOUT)

        if rlist:
            user_input = sys.stdin.readline().strip().lower()
            self.logger.info(f"User input received: {user_input}")
            if user_input == 'y':
                return "yes"
            elif user_input == 'n':
                return "no"
            elif user_input == 's':
                return "stop"
        else:
            self.logger.info("No input received. Timeout occurred.")
            return "timeout"

    # HUMAN-READABLE SNAPSHOT OF PARAMETERS  (called from main.py)
    def log_model_parameters(self, lstm_model, bandit_agent):
        """
        Save LSTM weights + bandit θ-vectors to logs/model_parameters_*.log
        so you can diff runs or make quick plots in pandas.
        """
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join("logs", f"model_parameters_{ts}.log")
            os.makedirs("logs", exist_ok=True)

            with open(path, "w") as f:
                # ---- LSTM --------------------------------------------------
                f.write("### LSTM parameters\n")
                for name, param in lstm_model.named_parameters():
                    f.write(f"{name}: {param.detach().cpu().numpy().tolist()}\n")

                # ---- Bandit ------------------------------------------------
                f.write("\n### Bandit theta vectors (A^-1 * b)\n")
                theta_mat = bandit_agent.get_theta_matrix()  # (9, 5)
                for a, theta in enumerate(theta_mat):
                    f.write(f"action {a}: {theta.tolist()}\n")

            self.logger.info(f"Model parameters logged → {path}")
        except Exception as e:
            self.logger.error(f"Could not log parameters: {e}")

    # ────────────────────────────────────────────────────────────────────────
    # PROGRESS DUMP (theta + reward curve)   …called from signal_handler.py
    def save_progress(self, bandit_agent, directory: str = 'logs'):
        """
        Saves a compact .npz containing:
          • theta:  (9,5) current LinUCB parameter matrix
          • rewards: (N,) per-step reward history for the session
        """
        os.makedirs(directory, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        np.savez_compressed(
            os.path.join(directory, f"bandit_progress_{ts}.npz"),
            theta=bandit_agent.get_theta_matrix(),
            rewards=np.array(self.reward_history, dtype=np.float32)
        )
        self.logger.info(f"Progress snapshot written → bandit_progress_{ts}.npz")
# -----------------------------------------------------------------------------
# Convenience factory (used by main.py via model_utils)
# -----------------------------------------------------------------------------

def make_env(connect_drone: bool = False):
    env = DroneControlEnv(connect_drone=connect_drone)
    return env
