import threading
import logging
import signal
from shared_events import stop_main_loop, stop_saving_thread, stop_input_listener, visualization_ready
from signal_handler import signal_handler
import numpy as np
from sklearn.exceptions import ConvergenceWarning  # Import ConvergenceWarning

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Initialize feature data store
feature_data_store = {
    "BandPower": [],
    "HjorthParameters": [],
    "SpectralEntropy": [],
    "FractalDimension": [],
    "FirstOrderDerivatives": [],
    "SecondOrderDerivatives": [],
    "EEGFiltered": []
}

def preprocessing_thread(data_queue, feature_queue, env, model_agent, visualizer, emotiv, stop_main_loop, feature_data_store, lstm_handler, signal_handler, MODEL_FILENAME, session_training_data, data_store, data_lock, raw_path, processed_path):
    """
    Thread for preprocessing data and making predictions in chunks of 256 frames.

    """
    
    lock = threading.Lock()
    step_count = 0  # Initialize step count
    logging.info(f"Buffers getting filled")
    while not stop_main_loop.is_set():  # Check stop_main_loop event
        try:
            

            # Collect a single packet from the queue
            packet = data_queue.get(timeout=1)
            if packet and len(packet) > 0:
                # Update EEG buffers
                buffers_full = emotiv.update_eeg_buffers(
                    packet,
                    emotiv.channel_names,
                    emotiv.primary_buffer,
                    emotiv.secondary_buffer,
                    emotiv.processing_in_progress,
                    feature_queue  # Pass feature_queue here
                )

                if isinstance(buffers_full, np.ndarray) and buffers_full.size > 0:

                    # Check if the feature window is ready for LSTM
                    feature_sequence = emotiv.get_feature_sequence()
                    if feature_sequence is not None:
                        # Extract features and update feature_data_store
                        preprocessed_data = emotiv.preprocess_eeg_data(buffers_full)
                        if preprocessed_data is not None:
                            feature_vector = emotiv.extract_features(preprocessed_data)
                            if feature_vector is not None:
                                # Update feature_data_store with extracted features
                                feature_data_store["EEGFiltered"].append(feature_vector[7118:])  # Example: Remaining values
                                feature_data_store["BandPower"].append(feature_vector[:70])  # Example: First 70 values for BandPower
                                feature_data_store["HjorthParameters"].append(feature_vector[70:98])  # Example: Next 28 values
                                feature_data_store["SpectralEntropy"].append(feature_vector[98:112])  # Example: Next 14 values
                                feature_data_store["FractalDimension"].append(feature_vector[112:126])  # Example: Next 14 values
                                feature_data_store["FirstOrderDerivatives"].append(feature_vector[126:3622])  # Example: Next 3496 values
                                feature_data_store["SecondOrderDerivatives"].append(feature_vector[3622:7118])  # Example: Next 3496 values

                        # Predict using the LSTM model
                        with lock:
                            logging.info(f"Feature sequence ready for prediction")

                            lstm_output = lstm_handler.predict(feature_sequence)

                            if lstm_output is not None:
                                try:
                                    # The output from the LSTM is the observation for the RL agent.
                                    observation = lstm_output

                                    # Predict an action using the RL agent based on the brain-state observation.
                                    action, _ = model_agent.predict(observation, deterministic=False)
                                    logging.info(f"[Preprocessing Thread] Predicted action from RL Agent: {action}")

                                    # Pass the action to the environment's step function.
                                    # We only need the reward and done status from the environment.
                                    # The 'state' returned here is the simulated drone state, which we will IGNORE for learning.
                                    _, reward, done, info = env.step(action)
                                    step_count += 1

                                    if env.ready_for_fresh_data:
                                        emotiv.reset_streaming_state()
                                        logging.debug("[Preprocessing Thread] Buffers and feature window cleared after step.")

                                    # Add the experience to the agent's memory (replay buffer).
                                    # This now correctly links the brain-state observation to the outcome.
                                    if model_agent and hasattr(model_agent, 'replay_buffer'):
                                        try:
                                            model_agent.replay_buffer.add(
                                                obs=observation,            # The brain-state observation before the action.
                                                next_obs=observation,       # The brain-state observation after the action.
                                                                            # Note: In a more advanced setup, you'd get a new observation here,
                                                                            # but for now, this correctly fixes the data type issue.
                                                action=action,
                                                reward=reward,
                                                done=done,
                                                infos=[{}]                  # infos should be a list of dictionaries.
                                            )
                                            #logging.info(f"[RL] Transition added. Buffer size: {model_agent.replay_buffer.size()}")
                                        except Exception as e:
                                            logging.error(f"[RL] Failed to add to replay buffer: {e}")

                                    # Handle user-requested stop
                                    if reward == 0.0 and info.get("info") == "User requested stop":
                                        signal_handler(signal.SIGINT, None, env, emotiv, data_store, stop_saving_thread, stop_main_loop, stop_input_listener, MODEL_FILENAME, feature_data_store, session_training_data, data_lock, raw_path, processed_path)
                                    
                                        

                                    
                                    logging.debug("[Preprocessing Thread] Buffers cleared after prediction.")
                                except Exception as e:
                                    logging.error(f"[Preprocessing Thread] Error during RL agent prediction or step execution: {e}")
                            else:
                                logging.warning("[Preprocessing Thread] LSTM prediction returned None. Skipping...")
                    else:
                        continue
            else:
                logging.warning("[Preprocessing Thread] Invalid or empty packet received. Skipping...")
        except KeyboardInterrupt:
            logging.info("[Preprocessing Thread] KeyboardInterrupt received. Stopping preprocessing thread.")
            stop_main_loop.set()  # Set the stop flag to exit the loop
            break
        except Exception as e:
            logging.error(f"[Preprocessing Thread] Error during preprocessing: {e}")
