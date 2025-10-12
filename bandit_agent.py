# ============================
# bandit_agent.py  (Improved)
# ============================
"""LinUCB with ε‑greedy exploration and tie‑break randomisation.
   Suitable for <30 samples per session.
"""
import numpy as np
import os, pickle

class LinUCBAgent:
    def __init__(self, n_actions: int = 9, dim: int = 5, alpha: float = 1.5,
                 epsilon: float = 0.3, min_epsilon: float = 0.05, decay: float = 0.995):
        self.n_actions    = n_actions
        self.dim          = dim
        self.alpha        = alpha

        # ε‑greedy params
        self.epsilon      = epsilon
        self.min_epsilon  = min_epsilon
        self.decay        = decay
        self.step_count   = 0

        # LinUCB stats
        self.A = [np.eye(dim) for _ in range(n_actions)]  # (d,d)
        self.b = [np.zeros((dim, 1)) for _ in range(n_actions)]

    # ------------------------------------------------------------------
    def predict(self, context: np.ndarray, deterministic: bool = False):
        """Return full action‑vector (discrete + 4 continuous values)."""
        disc = self._choose_action(context)
        action_vec = np.concatenate([[disc], context[1:]])  # keep LSTM cont.
        return action_vec, None

    def _choose_action(self, ctx: np.ndarray) -> int:
        # ε‑greedy exploration
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            # Compute UCB scores & random tie‑break
            x = ctx.reshape(-1, 1)
            scores = []
            for A, b in zip(self.A, self.b):
                theta = np.linalg.inv(A) @ b
                mean  = float(theta.T @ x)
                var   = float(x.T @ np.linalg.inv(A) @ x)
                scores.append(mean + self.alpha * np.sqrt(var))
            max_score = max(scores)
            best = [i for i, s in enumerate(scores) if np.isclose(s, max_score)]
            action = int(np.random.choice(best))
        return action

    # ------------------------------------------------------------------
    def update(self, context: np.ndarray, discrete_action: int, reward: float):
        x = context.reshape(-1, 1)
        a = int(discrete_action)
        self.A[a] += x @ x.T
        self.b[a] += reward * x

        # decay ε each step
        self.step_count += 1
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)

    # ------------------------------------------------------------------
    def get_theta_matrix(self) -> np.ndarray:
        return np.stack([(np.linalg.inv(A) @ b).flatten() for A, b in zip(self.A, self.b)])

    # ------------------------------------------------------------------
    # Persistence helpers
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(dict(A=self.A, b=self.b, eps=self.epsilon), f)

    @classmethod
    def load(cls, path: str, dim: int, n_actions: int):
        with open(f"{path}.pkl", "rb") as f:
            data = pickle.load(f)
        agent = cls(n_actions=n_actions, dim=dim)
        agent.A, agent.b, agent.epsilon = data['A'], data['b'], data['eps']
        return agent

"""
# ============================
# model_utils.py  (replace file)
# ============================
import os
import logging
from bandit_agent import LinUCBAgent

MODEL_FILENAME = os.path.join(os.getcwd(), "models", "drone_rl_eeg_bandit")


def load_or_create_model(env=None):
    #Return a LinUCB agent – loads from disk if possible.
    logger = logging.getLogger(__name__)
    dim = 5        # context length (matches LSTM output)
    n_actions = 9  # discrete 0‑8
    try:
        if os.path.exists(f"{MODEL_FILENAME}.pkl"):
            model = LinUCBAgent.load(MODEL_FILENAME, dim, n_actions)
            logger.info("Loaded existing LinUCB bandit model")
        else:
            model = LinUCBAgent(n_actions=n_actions, dim=dim, alpha=1.5)
            logger.info("Created new LinUCB bandit model")
    except Exception as e:
        logger.error(f"Failed to init LinUCB: {e}")
        raise
    return model


# No replay buffer, no learn() needed. save() is called from signal_handler.

# ============================
# preprocessing_thread.py  (patch – replace marked block)
# ============================
# --- inside the main while‑loop, after obtaining `lstm_output` ---
# OLD (SAC‑specific):
#     action, _ = model_agent.predict(lstm_output, deterministic=False)
#     ...
#     if model_agent and hasattr(model_agent, 'replay_buffer'):
#         model_agent.replay_buffer.add(...)
#
# REPLACE WITH:
# --------------------------------------------------------------
        # Predict discrete+continuous action vector using bandit
        action_vec, _ = model_agent.predict(lstm_output)
        logging.info(f"[Preproc] Bandit chose action {action_vec[0]}")

        prev_state = env.current_state.copy()
        state, reward, done, info = env.step(action_vec)

        # Online bandit update – no buffers required
        model_agent.update(lstm_output, action_vec[0], reward)

        step_count += 1
# --------------------------------------------------------------
# (Remove the entire replay_buffer add / learn sections.)

# ============================
# signal_handler.py  (patch – comment out SAC learn)
# ============================
# Remove every block that references model.replay_buffer or model.learn().
# Add instead right before exit:
# --------------------------------------------------------------
    if hasattr(env, 'model') and env.model and hasattr(env.model, 'save'):
        try:
            env.model.save(MODEL_FILENAME)
            logger.info(f"Bandit model saved to {MODEL_FILENAME}.pkl")
        except Exception as e:
            logger.error(f"Failed to save bandit model: {e}")
# --------------------------------------------------------------

"""