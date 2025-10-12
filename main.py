import threading
import queue  # For decoupling streaming and preprocessing
from datetime import datetime
from main_thread_fix import run_visualizations_on_main_thread  # Import refactored visualization function
from streaming_thread_fix import streaming_thread  # Import refactored streaming thread
from data_saver import save_data_continuously  # Import refactored data saver
import argparse
import signal  
from preprocessing_thread_fix import preprocessing_thread  # Import refactored preprocessing thread
from signal_handler import signal_handler  # Import refactored signal handler
from stream_data import EmotivStreamer
from learning_rlagent import DroneControlEnv  # Import refactored input listener
from model_utils import load_or_create_model
from lstm_handler import LSTMHandler
from visualizer_realtime3D_fix import RealtimeEEGVisualizer
import time
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import logging
from feature_visualizer_band_power import FeatureVisualizer  # Import FeatureVisualizer
from feature_visualizer_first_order_derivative import FeatureVisualizerFirstOrderDerivative  # Import FeatureVisualizerAllChannels
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from feature_visualizer_fractal_new import FractalVisualizer  # Import FractalVisualizer
from feature_visualizer_entropy_features import FeatureVisualizerEntropyFeatures  # Import FeatureVisualizerEntropyFeatures
from feature_visualizer_second_order_derivatives import FeatureVisualizerSecondOrderDerivatives  # Import FeatureVisualizerSecondOrderDerivatives
from feature_visualizer_eeg_filtered import FeatureVisualizerEEGFiltered  # Import FeatureVisualizerEEGFiltered
from feature_visualizer_hjorth_params import HjorthVisualizer  # Import FeatureVisualizerHjorthParameters
import psutil  # Add this import for monitoring memory usage
from sklearn.exceptions import ConvergenceWarning  # Import ConvergenceWarning
from shared_events import stop_saving_thread, stop_input_listener, stop_main_loop
session_training_data = []  # Store tuples of (feature_sequence, label)

lstm_handler = LSTMHandler()

SESSION_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

RAW_PATH = os.path.join("data", f"EEG_Raw_{SESSION_TIMESTAMP}.xlsx")
PROCESSED_PATH = os.path.join("data", f"Processed_Data_{SESSION_TIMESTAMP}.xlsx")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = os.path.join(BASE_DIR, "models", "drone_rl_eeg_bandit")
data_queue = queue.Queue()  # Shared queue for streaming and preprocessing
data_store = []  # Initialize data store
data_lock = threading.Lock()

visualization_queue = queue.Queue()  # Queue for sending data to the visualizer
# Queue for handling user input
input_queue = queue.Queue()
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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




def log_memory_usage():
    """
    Log the current memory usage of the program.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    logging.info(f"[Memory Usage] RSS: {memory_info.rss / (1024 * 1024):.2f} MB, VMS: {memory_info.vms / (1024 * 1024):.2f} MB")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Data Streamer and Drone Control")
    parser.add_argument("--connect-drone", action="store_true", help="Connect to the drone")
    args = parser.parse_args()
    emotiv = EmotivStreamer()
    visualizer = RealtimeEEGVisualizer()
    env = DroneControlEnv(connect_drone=args.connect_drone)  # Initialize the drone control environment
    channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    feature_visualizer = FeatureVisualizer(channel_names= channel_names, fs=256)  # Initialize feature visualizer
    derivatives_visualizer = FeatureVisualizerFirstOrderDerivative(channel_names= channel_names, fs=256)  # Initialize feature visualizer
    second_order_derivatives_visualizer = FeatureVisualizerSecondOrderDerivatives(channel_names= channel_names, fs=256)  # Initialize feature visualizer
    fractal_visualizer = FractalVisualizer(channel_names= channel_names, fs=256)  # Initialize feature visualizer
    entropy_visualizer = FeatureVisualizerEntropyFeatures(channel_names= channel_names, fs=256)  # Initialize feature visualizer
    eeg_filtered_visualizer = FeatureVisualizerEEGFiltered(channel_names= channel_names, fs=256)  # Initialize feature visualizer
    hjorth_visualizer = HjorthVisualizer(channel_names= channel_names, fs=256)  # Initialize feature visualizer
    # Start the input listener thread

    # Initialize LSTMHandler separately if not part of DroneControlEnv
    model_lstm = LSTMHandler()  # Initialize the LSTMHandler
    model_agent = load_or_create_model(env)
    env.model = model_agent
 
    # Initialize feature queue and feature visualizer
    feature_queue = queue.Queue()  # Queue for sending features to the feature visualizer
    feature_visualization_ready = threading.Event()  # Event to signal when feature visualization can start

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(
        sig, frame, env, emotiv, data_store, stop_saving_thread, stop_main_loop, stop_input_listener, MODEL_FILENAME, feature_data_store, session_training_data, data_lock, RAW_PATH, PROCESSED_PATH))

    # Periodically log memory usage in a separate thread
    def memory_monitor():
        while not stop_main_loop.is_set():
            log_memory_usage()
            time.sleep(240)  # Log memory usage every 120 seconds

    memory_thread = threading.Thread(target=memory_monitor, daemon=True)
    memory_thread.start()

    if emotiv.connect():
        logging.info("Emotiv EEG device connected. Starting real-time EEG streaming.")
        try:
            # Start background thread for data saving
            # This thread requires 5 arguments.
            logging.info("[Main] Starting save_data_continuously thread.")
            save_thread = threading.Thread(
                target=save_data_continuously,
                args=(data_store, stop_saving_thread, data_lock, RAW_PATH, PROCESSED_PATH)
            )
            save_thread.daemon = True
            save_thread.start()

            # Start streaming and preprocessing threads
            # The streaming thread requires 5 arguments.
            stream_thread = threading.Thread(
                target=streaming_thread,
                args=(emotiv, data_queue, visualization_queue, data_store, data_lock)
            )

            # The preprocessing thread requires all 14 arguments in this specific order.
            preprocess_thread = threading.Thread(
                target=preprocessing_thread,
                args=(
                    data_queue,
                    feature_queue,
                    env,
                    model_agent,
                    emotiv,  # Correctly positioned as the 5th argument
                    stop_main_loop, # Correctly positioned as the 6th argument
                    feature_data_store,
                    lstm_handler,
                    MODEL_FILENAME,
                    session_training_data,
                    data_store,
                    data_lock,
                    RAW_PATH,
                    PROCESSED_PATH
                )
            )
            stream_thread.start()
            preprocess_thread.start()

            # Run visualization on the main thread
            run_visualizations_on_main_thread(visualizer, visualization_queue, feature_visualizer, feature_queue, derivatives_visualizer, second_order_derivatives_visualizer, fractal_visualizer, entropy_visualizer, eeg_filtered_visualizer, hjorth_visualizer)
        except Exception as e:
            logging.error(f"An error occurred: {e}")

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received. Triggering signal handler.")
            signal_handler(signal.SIGINT, None, env, emotiv, data_store, stop_saving_thread, stop_main_loop, stop_input_listener, MODEL_FILENAME, feature_data_store, session_training_data, data_lock, RAW_PATH, PROCESSED_PATH)
        finally:
            logging.info("[Main] Stopping all threads.")
            #stop_saving_thread.set()  # Signal the save_data_continuously thread to stop
            #stop_main_loop.set()  # Signal other threads to stop
            #stop_input_listener.set()  # Signal the input listener thread to stop

            preprocess_thread.join()
            stream_thread.join()

            save_thread.join()  # Ensure save_data_continuously thread stops
            memory_thread.join()  # Ensure memory monitor thread stops
            logging.info("[Main] All threads stopped.")

            # Log model parameters to a file
            logging.info("Logging LSTM model and RL agent parameters to a file...")
            env.log_model_parameters(model_lstm, model_agent)

            emotiv.disconnect()
    else:
        logging.error("Failed to connect to Emotiv device.")
