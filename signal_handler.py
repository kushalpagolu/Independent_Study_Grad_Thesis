import logging
import os
import time
import signal
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import threading
from kalman_filter import KalmanFilter
import torch
import numpy as np
import sys
from lstm_handler import LSTMHandler
from lstm_model import LSTMModel, LSTMTrainer
from openpyxl import load_workbook

human_feedback = None
feedback_condition = threading.Condition()

def save_features_to_excel(feature_data_store):
    try:
        os.makedirs("data/features", exist_ok=True)
        eeg_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for feature_name, feature_data in feature_data_store.items():
            if not feature_data:
                continue

            if feature_name == "BandPower":
                column_names = [f"{channel}_Band_{j+1}" for channel in eeg_channels for j in range(5)]
            elif feature_name == "HjorthParameters":
                column_names = [f"{channel}_Hjorth_{j+1}" for channel in eeg_channels for j in range(2)]
            elif feature_name == "SpectralEntropy":
                column_names = [f"{channel}_Entropy" for channel in eeg_channels]
            elif feature_name == "FractalDimension":
                column_names = [f"{channel}_Fractal" for channel in eeg_channels]
            elif feature_name == "FirstOrderDerivatives":
                column_names = [f"{channel}_Sample_{j+1}" for channel in eeg_channels for j in range(256)]
            elif feature_name == "SecondOrderDerivatives":
                column_names = [f"{channel}_Sample_{j+1}" for channel in eeg_channels for j in range(256)]
            elif feature_name == "EEGFiltered":
                num_channels = len(eeg_channels)
                actual_len = len(feature_data[0])
                samples_per_channel = actual_len // num_channels
                expected_len = samples_per_channel * num_channels

                if expected_len != actual_len:
                    logging.warning(f"[Save Features] Trimming EEGFiltered feature from {actual_len} to {expected_len} to match {num_channels} channels x {samples_per_channel} samples.")
                    feature_data = [row[:expected_len] for row in feature_data]

                column_names = [f"{channel}_Filtered_Sample_{j+1}" for channel in eeg_channels for j in range(samples_per_channel)]


            filename = os.path.join("data/features", f"{feature_name}_Features_{timestamp}.xlsx")
            df = pd.DataFrame(feature_data, columns=column_names)
            df.to_excel(filename, index=False)
            logging.info(f"[Save Features] {feature_name} features saved to {filename}")
    except Exception as e:
        logging.error(f"[Save Features] Error saving feature data: {e}")


def signal_handler(sig, frame, env, emotiv, data_store, stop_saving_thread, stop_main_loop, stop_input_listener, MODEL_FILENAME, feature_data_store, session_training_data, data_lock, raw_path, processed_path):
    logger = logging.getLogger(__name__)
    logger.info(f"Signal handler triggered. Performing cleanup and saving data...Model :{str(env.model)}" )
    with data_lock:
        local_data = data_store.copy()
        data_store.clear()
    stop_main_loop.set()
    stop_saving_thread.set()
    stop_input_listener.set()
    
    if feature_data_store:
        #logger.info("[Signal Handler] Saving feature-specific data...")
        save_features_to_excel(feature_data_store)

    if hasattr(env, 'model') and env.model and hasattr(env.model, 'save'):
        try:
            env.model.save(MODEL_FILENAME)
            logger.info(f"Bandit model saved to {MODEL_FILENAME}.pkl")
        except Exception as e:
            logger.error(f"Error saving bandit model: {e}")


    if hasattr(env, 'model_agent') and env.model:
        agent_replay_buffer_size = env.model_agent.replay_buffer.size()
        if agent_replay_buffer_size > 10:
            env.model_agent.learn(total_timesteps=agent_replay_buffer_size, reset_num_timesteps=False)

        logger.info("Saving RL model before exiting...")
        env.model.save(MODEL_FILENAME)
        logger.info(f"RL Model saved to {MODEL_FILENAME}.zip")

    if hasattr(env, 'model_lstm') and env.model_lstm:
        lstm_model_path = os.path.join("models", "lstm_model.pth")
        try:
            if session_training_data:
                logger.info(f"Training LSTM model with {len(session_training_data)} samples from this session.")
                X_train = np.array([x for x, y in session_training_data])
                y_train = np.array([y for x, y in session_training_data])
                env.model_lstm.trainer.train(X_train, y_train, epochs=10, batch_size=4)
            env.model_lstm.trainer.save_model(lstm_model_path)
            logger.info(f"LSTM model trained and saved to {lstm_model_path}")
        except Exception as e:
            logger.error(f"Error during LSTM training or saving: {e}")


   


        # just before exit(0)
    if hasattr(env, 'save_progress'):
        try:
            env.save_progress(env.model)         # bandit agent = env.model
        except Exception as e:
            logger.error(f"Progress save failed: {e}")



    if local_data:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_filename = os.path.join("data", f"raw_data_{timestamp}.xlsx")
        os.makedirs("Raw_Data", exist_ok=True)
        os.makedirs("Processed_Data", exist_ok=True)
        raw_filename       = os.path.join("Raw_Data",      f"raw_data_{timestamp}.xlsx")
        processed_filename = os.path.join("Processed_Data", f"processed_data_{timestamp}.xlsx")

        eeg_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        gyro_channels = ["gyro_x", "gyro_y"]
        column_names = eeg_channels + gyro_channels

        #logger.warning(f"[Signal Handler] Data store size: {len(local_data)}")
        #logger.warning(f"[Signal Handler] First row: {local_data[0] if local_data else 'Empty'}")

        if not all(isinstance(row, list) and len(row) == len(column_names) for row in local_data):
            logger.error("[Signal Handler] Invalid local_data format. Skipping save.")
        else:
            try:
                #logger.info(f"[Signal Handler] Saving {len(local_data)} rows to raw/processed Excel files.")
                df = pd.DataFrame(local_data, columns=column_names)

                for channel in eeg_channels:
                    df[f"{channel}_volts"] = df[channel] * 0.51

                kalman_filter_x = KalmanFilter()
                kalman_filter_y = KalmanFilter()
                df["gyro_x_deg_s"] = df["gyro_x"].apply(kalman_filter_x.update)
                df["gyro_y_deg_s"] = df["gyro_y"].apply(kalman_filter_y.update)
                df["head_roll_deg"] = df["gyro_x_deg_s"].cumsum() * (1 / 128)
                df["head_pitch_deg"] = df["gyro_y_deg_s"].cumsum() * (1 / 128)

                for channel in eeg_channels:
                    df[f"{channel}_med_subtracted"] = df[channel].subtract(df[eeg_channels].median(axis=1), axis=0)

                for i in range(1, len(df)):
                    delta = df[eeg_channels].iloc[i] - df[eeg_channels].iloc[i-1]
                    delta = delta.clip(-15, 15)
                    df.loc[i, eeg_channels] = df.loc[i-1, eeg_channels] + delta

                # write a fresh file *with* headers
                df[eeg_channels + gyro_channels].to_excel(raw_filename, index=False)
                df.to_excel(processed_filename, index=False)
                


                logger.info(f"[Signal Handler] Data saved successfully to {raw_filename} and {processed_filename}.")
                
            except Exception as e:
                logger.error(f"Error saving data to Excel: {str(e)}")

    plt.close('all')
    exit(0)


