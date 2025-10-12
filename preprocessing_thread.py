import threading
import logging
import signal
import queue
import numpy as np
from sklearn.exceptions import ConvergenceWarning

# FIX: Import the shared stop events and the signal_handler function
from shared_events import stop_main_loop, stop_saving_thread, stop_input_listener
from signal_handler import signal_handler

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger(__name__)

def preprocessing_thread(data_queue, feature_queue, env, model_agent, emotiv, stop_main_loop, feature_data_store, lstm_handler, MODEL_FILENAME, session_training_data, data_store, data_lock, raw_path, processed_path):
    """
    Thread for preprocessing data, making predictions, and handling shutdown signals.
    """
    
    step_count = 0
    logger.info("Preprocessing thread started, waiting for data...")
    
    while not stop_main_loop.is_set():
        try:
            packet = data_queue.get(timeout=1)
            
            if packet and len(packet) > 0:
                processed_eeg_chunk = emotiv.update_eeg_buffers(
    packet,
    emotiv.channel_names,
    emotiv.primary_buffer,
    emotiv.secondary_buffer,
    emotiv.processing_in_progress,
    feature_queue
)

                if isinstance(processed_eeg_chunk, np.ndarray) and processed_eeg_chunk.size > 0:
                    preprocessed_data = emotiv.preprocess_eeg_data(processed_eeg_chunk)
                    if preprocessed_data is None:
                        continue

                    feature_vector = emotiv.extract_features(preprocessed_data)
                    if feature_vector is not None:
                        feature_queue.put(feature_vector)
                        emotiv.update_feature_window(feature_vector)
                        
                        # Store features for saving
                        feature_data_store["BandPower"].append(feature_vector[:70])
                        feature_data_store["HjorthParameters"].append(feature_vector[70:98])
                        feature_data_store["SpectralEntropy"].append(feature_vector[98:112])
                        # (add other feature appends here if needed)

                    feature_sequence = emotiv.get_feature_sequence()
                    if feature_sequence is not None:
                        logger.info("10-second feature sequence ready for prediction.")
                        
                        lstm_output = lstm_handler.predict(feature_sequence)
                        if lstm_output is not None:
                            try:
                                action_vec, _ = model_agent.predict(lstm_output)
                                logger.info(f"[Preproc] Bandit chose discrete action: {action_vec[0]}")

                                prev_state = env.current_state.copy()
                                state, reward, done, info = env.step(action_vec)
                                
                                # --- FIX: ADDED SHUTDOWN LOGIC ---
                                # Check if the environment step returned a user-initiated stop signal
                                if done and info.get("info") == "User requested stop":
                                    logger.info("User requested stop. Initiating graceful shutdown.")
                                    # Call the main signal handler to save everything and exit
                                    signal_handler(
                                        signal.SIGINT, None, env, emotiv, data_store,
                                        stop_saving_thread, stop_main_loop, stop_input_listener,
                                        MODEL_FILENAME, feature_data_store, session_training_data,
                                        data_lock, raw_path, processed_path
                                    )
                                    # Break the loop to ensure this thread terminates immediately
                                    break
                                # --- END OF FIX ---

                                # (rest of the prediction logic remains the same) ...
                                emotiv.feature_window.clear()
                                logger.info("[Preprocessing Thread] Feature window cleared for next 10s cycle.")

                            except Exception as e:
                                logger.error(f"[Preprocessing] Error during RL step: {e}", exc_info=True)
                        else:
                            logger.warning("[Preprocessing] LSTM prediction was None.")
                            
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            logger.info("[Preprocessing] KeyboardInterrupt. Stopping.")
            stop_main_loop.set()
            break
        except Exception as e:
            logger.error(f"[Preprocessing] An error occurred: {e}", exc_info=True)
            
    logger.info("Preprocessing thread has terminated.")

