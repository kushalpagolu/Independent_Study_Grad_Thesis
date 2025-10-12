import logging
import os
import time
import signal
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import threading
from kalman_filter import KalmanFilter  # Assuming you have a kalman_filter module
import torch
import numpy as np
import sys
import signal
# Global variables for human feedback
human_feedback = None
feedback_condition = threading.Condition()
from lstm_handler import LSTMHandler
from lstm_model import LSTMModel, LSTMTrainer

def save_features_to_excel(feature_data_store):
    """
    Save feature-specific data into separate Excel sheets dynamically.
    Args:
        feature_data_store (dict): Dictionary containing feature data for each feature type.
    """
    try:
        # Ensure the "data" directory exists
        os.makedirs("data", exist_ok=True)

        eeg_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for feature_name, feature_data in feature_data_store.items():
            # Generate column names dynamically based on feature type
            if feature_name == "BandPower":
                column_names = [f"{channel}_Band_{j+1}" for channel in eeg_channels for j in range(5)]  # 14 channels × 5 bands
            elif feature_name == "HjorthParameters":
                column_names = [f"{channel}_Hjorth_{j+1}" for channel in eeg_channels for j in range(2)]  # 14 channels × 2 parameters
            elif feature_name == "SpectralEntropy":
                column_names = [f"{channel}_Entropy" for channel in eeg_channels]  # 1 per channel
            elif feature_name == "FractalDimension":
                column_names = [f"{channel}_Fractal" for channel in eeg_channels]  # 1 per channel
            elif feature_name == "FirstOrderDerivatives":
                column_names = [f"{channel}_Sample_{j+1}" for channel in eeg_channels for j in range(256)]  # 14 channels × 256 samples
            elif feature_name == "SecondOrderDerivatives":
                column_names = [f"{channel}_Sample_{j+1}" for channel in eeg_channels for j in range(256)]
            elif feature_name == "EEGFiltered":
                # Dynamically adjust column names based on the actual data length
                num_samples = len(feature_data[0]) if feature_data else 0
                num_channels = len(eeg_channels)
                if num_samples % num_channels != 0:
                    logging.warning(f"[Save Features] Mismatch in EEGFiltered data: {num_samples} samples not divisible by {num_channels} channels.")
                column_names = [f"{channel}_Filtered_Sample_{j+1}" for j in range(num_samples // num_channels) for channel in eeg_channels]
                column_names = column_names[:num_samples]  # Ensure the column names match the exact number of samples
                logging.debug(f"[Save Features] Adjusted column names for EEGFiltered with {num_samples} samples.")
            else:
                column_names = [f"Feature_{i+1}" for i in range(len(feature_data[0]))]  # Generic column names

            # Adjust column names to match the data shape
            num_columns = len(feature_data[0]) if feature_data else 0
            column_names = column_names[:num_columns]

            # Save the feature data to an Excel file, even if it contains null values
            filename = os.path.join("data", f"{feature_name}_Features_{timestamp}.xlsx")
            df = pd.DataFrame(feature_data, columns=column_names)
            df.to_excel(filename, index=False)
            logging.info(f"[Save Features] {feature_name} features saved to {filename}")
    except Exception as e:
        logging.error(f"[Save Features] Error saving feature data: {e}")


    
def signal_handler(sig, frame, env, emotiv, data_store, stop_saving_thread, stop_main_loop, stop_input_listener, MODEL_FILENAME, feature_data_store, session_training_data):
    """
    Handles the SIGINT signal to stop the program, save data, and clean up resources.
    """
    logger = logging.getLogger(__name__)
    logger.info("Signal handler triggered. Performing cleanup and saving data...")

    # Set stop flags for all threads
    stop_main_loop.set()
    stop_saving_thread.set()
    stop_input_listener.set()

    # Save feature-specific data at the end of the session
    if feature_data_store:
        logger.info("[Signal Handler] Saving feature-specific data...")
        save_features_to_excel(feature_data_store)

    # Ensure the model is saved before exit
    if hasattr(env, 'model') and env.model:
        logger.info(f"Saving model before exiting...")
        env.model.save(MODEL_FILENAME)
        logger.info(f"Model saved to {MODEL_FILENAME}.zip")

    # Save RL Agent
    if hasattr(env, 'model_agent') and env.model:
        logger.info(f"Saving RL model before exiting...")
        env.model.save(MODEL_FILENAME)
        logger.info(f"RL Model saved to {MODEL_FILENAME}.zip")

    # Save LSTM model (PyTorch)
    if hasattr(env, 'model_lstm') and env.model_lstm:
        lstm_model_path = os.path.join("models", "lstm_model.pth")
        try:
            if session_training_data:
                logger.info(f"Training LSTM model with {len(session_training_data)} samples from this session.")
                
                # Split into input and label arrays
                X_train = np.array([x for x, y in session_training_data])
                y_train = np.array([y for x, y in session_training_data])
                
                # Train the model
                env.model_lstm.trainer.train(X_train, y_train, epochs=10, batch_size=4)

            # Save the updated model
            env.model_lstm.trainer.save_model(lstm_model_path)
            logger.info(f"LSTM model trained and saved to {lstm_model_path}")

        except Exception as e:
            logger.error(f"Error during LSTM training or saving: {e}")


    # Save remaining data at the end of the session
    if data_store:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_filename = os.path.join("data", f"EEG_Raw_{timestamp}.xlsx")
        processed_filename = os.path.join("data", f"Processed_Data_{timestamp}.xlsx")

        try:
            # Define EEG and gyro channel names
            eeg_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
            gyro_channels = ["gyro_x", "gyro_y"]
            column_names = eeg_channels + gyro_channels

            # Convert data_store to a DataFrame
            df = pd.DataFrame(data_store, columns=column_names)

            # Compute volts for EEG channels
            for channel in eeg_channels:
                df[f"{channel}_volts"] = df[channel] * 0.51

            # Apply Kalman filter to gyro data
            kalman_filter_x = KalmanFilter()
            kalman_filter_y = KalmanFilter()
            df["gyro_x_deg_s"] = df["gyro_x"].apply(kalman_filter_x.update)
            df["gyro_y_deg_s"] = df["gyro_y"].apply(kalman_filter_y.update)

            # Integrate gyro data for angles
            df["head_roll_deg"] = df["gyro_x_deg_s"].cumsum() * (1 / 128)
            df["head_pitch_deg"] = df["gyro_y_deg_s"].cumsum() * (1 / 128)

            # Median subtraction for EEG channels
            for channel in eeg_channels:
                df[f"{channel}_med_subtracted"] = df[channel].subtract(df[eeg_channels].median(axis=1), axis=0)

            # Clip and smooth EEG data
            for i in range(1, len(df)):
                delta = df[eeg_channels].iloc[i] - df[eeg_channels].iloc[i-1]
                delta = delta.clip(-15, 15)
                df.loc[i, eeg_channels] = df.loc[i-1, eeg_channels] + delta

            # Save raw data
            df[eeg_channels + gyro_channels].to_excel(raw_filename, index=False)
            logger.info(f"Raw Data saved to {raw_filename}")

            # Save processed data
            df.to_excel(processed_filename, index=False)
            logger.info(f"Processed Data saved to {processed_filename}")
        except Exception as e:
            logger.error(f"Error saving data to Excel: {str(e)}")

    plt.close('all')
    exit(0)

def feedback_signal_handler(sig, frame):
    """
    Signal handler for human feedback.
    SIGUSR1: Approve action.
    SIGUSR2: Reject action.
    """
    global human_feedback
    with feedback_condition:
        if sig == signal.SIGUSR1:
            human_feedback = True
            logging.info("SIGUSR1 received: Approve action.")
        elif sig == signal.SIGUSR2:
            human_feedback = False
            logging.info("SIGUSR2 received: Reject action.")
        else:
            logging.warning(f"Unexpected signal received: {sig}")
        feedback_condition.notify()  # Notify waiting threads
        logging.info("feedback_condition.notify() called.")
