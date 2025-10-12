import hid
import numpy as np
from Crypto.Cipher import AES
from datetime import datetime
import threading
import queue
import logging
import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from collections import deque
from scipy.signal import butter, filtfilt, welch
from scipy.stats import entropy
from numpy.fft import fft
from kalman_filter import KalmanFilter  # Ensure the correct import
from sklearn.exceptions import ConvergenceWarning  # Import ConvergenceWarning
from sklearn.decomposition import FastICA
#from feature_visualizer import FeatureVisualizer
from feature_extraction import (
    apply_bandpass_filter, apply_notch_filter, common_average_reference,
    apply_ica, apply_anc, apply_hanning_window, apply_dwt_denoising,
    compute_band_power, compute_hjorth_parameters, compute_spectral_entropy,
    higuchi_fractal_dimension, normalize_features
)
from filtering import bandpass_filter
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# Ensure the logs directory exists
logs_dir = "/Users/kushalpagolu/Desktop/EmotivEpoch/epoch_tello_RL_3DBrain/logs"
os.makedirs(logs_dir, exist_ok=True)
stop_main_loop = threading.Event()

# Generate a timestamped log file name
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"{logs_dir}/stream_data_{timestamp}.log"

# Update logging configuration to save logs to a new file for each run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),  # Save logs to a timestamped file
        logging.StreamHandler()  # Also display logs in the console
    ]
)

class EmotivStreamer:
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.vid = 0x1234
        self.pid = 0xed02
        self.device = None
        self.cipher = None
        self.cypher_key = bytes.fromhex("31003554381037423100354838003750")
        self.channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        self.fs = 256  # Sampling frequency
        self.buffer_size = self.fs  # 1-second buffer size
        self.primary_buffer = self.initialize_buffers(self.channel_names, self.buffer_size)
        self.secondary_buffer = self.initialize_buffers(self.channel_names, self.buffer_size)
        self.processing_in_progress = False  # Flag to indicate if processing is happening
        self.logger = logging.getLogger(__name__)
        self.consecutive_invalid_packets = 0  # Counter for invalid packets
        self.max_invalid_packets = 256  # Threshold for reconnection
        self.is_buffer_ready = False  # Initialize buffer readiness flag
        self.eeg_buffers = {channel: [] for channel in self.channel_names}
        self.buffer_size = 256  # Example buffer size
        self.required_buffer_size = self.fs  # Set required buffer size to match 1 second of data (256 samples)
        self.ready_for_fresh_data = False


        self.reference_channel = 0  # Example reference channel index
        self.use_ica = True
        self.use_dwt = True
        self.use_hfd = True
        self.use_bandpass = True
        self.use_hjorth = True
        self.use_entropy = True
        self.use_bandpower = True
        self.use_hanning = True
        self.use_anc = False
        self.feature_window = deque(maxlen=10)  # Store the last 10 feature vectors (10 seconds)


    def initialize_buffers(self, channel_names, buffer_size):
        # Configure logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        """
        Initialize rolling buffers for EEG data.
        """
        buffers = {ch: deque(maxlen=buffer_size) for ch in channel_names}
        #self.logger.info(f"Initialized buffers for channels: {channel_names}")
        return buffers

    def reset_streaming_state(self):
        """
        Fully reset buffers and feature window for fresh LSTM prediction cycle.
        """
        self.primary_buffer = self.initialize_buffers(self.channel_names, self.buffer_size)
        self.secondary_buffer = self.initialize_buffers(self.channel_names, self.buffer_size)
        self.feature_window.clear()
        self.ready_for_fresh_data = False
        self.logger.info("[EmotivStreamer] Streaming buffers and feature window reset for fresh LSTM prediction cycle....")
        self.logger.info("[EmotivStreamer] Updating Buffers.....")



    def connect(self):
            try:
                self.device = hid.device()

                self.device.open(self.vid, self.pid)

                if self.device is None:
                    self.logger.error("Device object is None after opening. Check VID/PID or permissions.")
                    return False

                self.logger.info(f"Connected to Emotiv device {self.vid:04x}:{self.pid:04x}")
                self.device.set_nonblocking(1)
                self.cipher = AES.new(self.cypher_key, AES.MODE_ECB)
                return True

            except Exception as e:
                self.logger.error(f"Connection failed: {str(e)}")
                return False

    def disconnect(self):
        if self.device:
            self.device.close()
            self.logger.info("Disconnected from Emotiv device")


    def read_emotiv_data(self):
        """
        Reads a packet from the Emotiv device and handles empty packets gracefully.
        """
        try:
            encrypted = bytes(self.device.read(32))
            #self.logger.info(f"Encrypted packet length: {len(encrypted)}")

            if len(encrypted) == 0:
               # self.logger.warning("Empty packet received. Skipping...")
                return None

            if len(encrypted) != 32:
               # self.logger.error(f"Invalid packet length ({len(encrypted)}). Skipping packet.")
                return None

            decrypted = self.cipher.decrypt(encrypted)
            #self.logger.info(f"Decrypted packet length: {len(decrypted)}")

            packet = list(decrypted)
            packet_dict = {
                'timestamp': datetime.now().isoformat(),
                'counter': decrypted[0],
                'gyro_x': decrypted[29],
                'gyro_y': decrypted[30],
                'battery': (decrypted[31] & 0x0F)
            }

            for i, channel_name in enumerate(self.channel_names):
                start_idx = 2 * i + 1
                end_idx = start_idx + 2
                packet_dict[channel_name] = int.from_bytes(decrypted[start_idx:end_idx], 'big', signed=True)

            #self.logger.info(f"Packet received")
            return packet_dict

        except Exception as e:
            self.logger.error(f"Error reading packet: {e}")
            return None

    def update_eeg_buffers(self, raw_data, channel_names, primary_buffer, secondary_buffer, processing_in_progress, feature_queue):
        """
        Updates the EEG buffers with new raw data and processes features.
        Args:
            raw_data (dict): Raw EEG data packet.
            channel_names (list): List of EEG channel names.
            primary_buffer (dict): Primary buffer for EEG data.
            secondary_buffer (dict): Secondary buffer for EEG data.
            processing_in_progress (bool): Flag indicating if processing is ongoing.
            feature_queue (queue.Queue): Queue to send features to the FeatureVisualizer.
        Returns:
            bool: True if buffers are full and ready for feature extraction, False otherwise.
        """
        try:
            # Update the buffers with raw data
            for channel in channel_names:
                if channel in raw_data:
                    primary_buffer[channel].append(raw_data[channel])

            # Check if the primary buffer is full
            if len(primary_buffer[channel_names[0]]) >= self.required_buffer_size:
                # Move data to secondary buffer for processing
                for channel in channel_names:
                    secondary_buffer[channel] = list(primary_buffer[channel])[:self.required_buffer_size]
                    primary_buffer[channel] = list(primary_buffer[channel])[self.required_buffer_size:]

                # Validate secondary buffer
                if all(len(secondary_buffer[channel]) == self.required_buffer_size for channel in channel_names):
                    eeg_data = np.array([list(secondary_buffer[channel]) for channel in channel_names])
                    self.process_and_extract_features(eeg_data, feature_queue)  # Extract features and update the feature window
                    return eeg_data
                else:
                    self.logger.warning("[update_eeg_buffers] Secondary buffer is not fully populated. Skipping processing.")
            return False
        except Exception as e:
            self.logger.error(f"Error updating EEG buffers: {e}")
            return False

    def preprocess_eeg_data(self, eeg_data):
        """
        Preprocess EEG data using optimized preprocessing techniques.
        """
        try:
            #self.logger.info("[Preprocess EEG Data] Starting preprocessing...")
            self.processing_in_progress = True

            # Retrieve data from the secondary buffer
            #eeg_data = np.array([list(self.secondary_buffer[channel]) for channel in self.channel_names])
            if eeg_data.shape[1] == 0:
                #self.logger.error("[Preprocess EEG Data] Secondary buffer is empty. Skipping preprocessing.")
                self.processing_in_progress = False
                return None

            #self.logger.info(f"[Preprocess EEG Data] Raw EEG data shape: {eeg_data.shape}")

            # Check if the buffer is sufficiently filled
            if self.reference_channel:
                noise_ref = np.array(self.secondary_buffer[self.reference_channel])  # Use dedicated reference channel
            else:
                noise_ref = np.mean(eeg_data, axis=0, keepdims=True)  # Estimate noise from EEG data

            # 1. Noise Removal (Powerline + Band Filtering)
            eeg_data = apply_notch_filter(eeg_data, fs=self.fs)  # Remove powerline noise (50/60 Hz)
            #self.logger.info(f"[Preprocess EEG Data] After Notch Filter shape: {eeg_data.shape}")

            eeg_data = apply_bandpass_filter(eeg_data, lowcut=1.0, highcut=50.0, sampling_rate=self.fs)  # Retain relevant EEG bands
            #self.logger.info(f"[Preprocess EEG Data] After Bandpass Filter shape: {eeg_data.shape}")

            # 2. Noise Reduction & Re-referencing
            eeg_data = common_average_reference(eeg_data)  # Reduce background noise using CAR
            #self.logger.info(f"[Preprocess EEG Data] After Common Average Reference shape: {eeg_data.shape}")

            # 3. Adaptive Noise Cancellation (ANC) - If a noise reference exists
            if self.use_anc:
                eeg_data = apply_anc(eeg_data, noise_ref)  # Remove structured noise
                #self.logger.info(f"[Preprocess EEG Data] After ANC shape: {eeg_data.shape}")

            # 4. Independent Component Analysis (ICA) - Optional for real-time processing
            if self.use_ica:
                eeg_data = apply_ica(eeg_data)  # Remove eye blink & muscle artifacts
                #self.logger.info(f"[Preprocess EEG Data] After ICA shape: {eeg_data.shape}")

            # 5. Apply Smoothing - Hanning Window
            eeg_data = apply_hanning_window(eeg_data)  # Smooth signal for better feature extraction
            #self.logger.info(f"[Preprocess EEG Data] After Hanning Window shape: {eeg_data.shape}")

            # 6. Denoising using Discrete Wavelet Transform (DWT) - Optional for real-time use
            if self.use_dwt:
                eeg_data = apply_dwt_denoising(eeg_data)  # Reduce high-frequency noise
                #self.logger.info(f"[Preprocess EEG Data] After DWT Denoising shape: {eeg_data.shape}")



            self.processing_in_progress = False
            return eeg_data

        except Exception as e:
            self.logger.error(f"[Preprocess EEG Data] Error during preprocessing: {e}")
            self.processing_in_progress = False
            return None


    def extract_features(self, eeg_filtered):
        """
        Extract features using all available feature extraction methods.
        :param eeg_filtered: Preprocessed EEG data of shape (14, 256).
        :return: Feature vector of shape (total_features,).
        """
        try:
            band_power_features = []
            hjorth_features = []
            entropy_features = []
            fractal_features = []

            for i, channel in enumerate(self.channel_names):
                # Ensure the channel data is 1D
                channel_data = eeg_filtered[i]
                if len(channel_data.shape) != 1:
                    raise ValueError(f"Channel {channel} data is not 1D. Shape: {channel_data.shape}")

                # Compute band power
                band_power = compute_band_power(channel_data, fs=self.fs)
                band_power_array = np.array(list(band_power.values()))
                #self.logger.info(f"Band power features array shape of {channel}: {band_power_array.shape}")
                band_power_features.append(band_power_array)
                #self.logger.info(f"Band power features shape of {channel}: {band_power_array.shape}")
                #self.logger.info(f"Band power features {channel}: {band_power}")

                # Compute Hjorth parameters
                hjorth = compute_hjorth_parameters(channel_data)
                #self.logger.info(f"Hjorth parameters shape of {channel}: {hjorth.shape}")
                hjorth_features.append(hjorth)
                #self.logger.info(f"Hjorth features for {channel}: {hjorth_features}")

                # Compute spectral entropy
                spectral_entropy = compute_spectral_entropy(channel_data, self.fs)
                #self.logger.info(f"Spectral entropy shape of {channel}: {spectral_entropy.shape}")
                entropy_features.append(spectral_entropy)
                #self.logger.info(f"Entropy features of {channel}: {entropy_features}")

                # Compute Higuchi fractal dimension
                fractal_dimension = higuchi_fractal_dimension(channel_data)
                #self.logger.info(f"Fractal dimension shape of {channel}: {fractal_dimension.shape}")
                fractal_features.append(fractal_dimension)
                #self.logger.info(f"Fractal features of {channel}: {fractal_features}")

            # Convert lists to numpy arrays
            band_power_features = np.array(band_power_features)  # (14, num_bands)
            #self.logger.info(f"Band power features shape after conversion: {band_power_features.shape}")
            hjorth_features = np.array(hjorth_features)          # (14, 2)
            #self.logger.info(f"Hjorth features shape after conversion: {hjorth_features.shape}")
            entropy_features = np.array(entropy_features)        # (14,)
            #self.logger.info(f"Entropy features shape after conversion: {entropy_features.shape}")
            fractal_features = np.array(fractal_features)        # (14,)
            #self.logger.info(f"Fractal features shape after conversion: {fractal_features.shape}")
            # Ensure all features are 1D

            # Compute Temporal Derivatives
            first_order_derivatives = np.diff(eeg_filtered, axis=1, prepend=eeg_filtered[:, :1])  # (14, 256)
            second_order_derivatives = np.diff(first_order_derivatives, axis=1, prepend=first_order_derivatives[:, :1])  # (14, 256)
            #self.logger.info(f"First order derivatives shape: {first_order_derivatives.shape}")
            #self.logger.info(f"Second order derivatives shape: {second_order_derivatives.shape}")

            # Normalize all features
            band_power_features = normalize_features(band_power_features)  # Normalize static features
            #self.logger.info(f"Band power features shape after normalization: {band_power_features.shape}")
            hjorth_features = normalize_features(hjorth_features)
            #self.logger.info(f"Hjorth features shape after normalization: {hjorth_features.shape}")
            entropy_features = normalize_features(entropy_features.reshape(-1, 1)).flatten()
            #self.logger.info(f"Entropy features shape after normalization: {entropy_features.shape}")
            fractal_features = normalize_features(fractal_features.reshape(-1, 1)).flatten()
            #self.logger.info(f"Fractal features shape after normalization: {fractal_features.shape}")

            first_order_derivatives = normalize_features(first_order_derivatives)
            #self.logger.info(f"First order derivatives shape after normalization: {first_order_derivatives.shape}")
            second_order_derivatives = normalize_features(second_order_derivatives)
            #self.logger.info(f"Second order derivatives shape after normalization: {second_order_derivatives.shape}")
            #self.logger.info(f"First order derivatives shape: {first_order_derivatives.shape}")
            #self.logger.info(f"Second order derivatives shape: {second_order_derivatives.shape}")

            # Concatenate all features
            static_features = np.concatenate((
                band_power_features.flatten(),  # Flatten static features into a single array
                hjorth_features.flatten(),
                entropy_features,
                fractal_features
            ))
            #self.logger.info(f"Static features shape: {static_features.shape}")
            dynamic_features = np.concatenate((
                eeg_filtered.flatten(),           # Flatten dynamic features into a single array
                first_order_derivatives.flatten(),
                second_order_derivatives.flatten()
            ))
            #self.logger.info(f"Dynamic features shape: {dynamic_features.shape}")
            feature_vector = np.concatenate((static_features, dynamic_features))
            #self.logger.info(f"Feature vector shape: {feature_vector.shape}")
            #self.logger.info(f"Extracted features for channel {self.channel_names[i]}: {feature_vector}")
            return feature_vector

        except Exception as e:
            self.logger.error(f"Error during feature extraction: {e}")

    def extract_features_sequence(self, eeg_data_sequence):
        """
        Extract features for a sequence of EEG data (10 seconds).
        :param eeg_data_sequence: EEG data of shape (10, 14, 256).
        :return: Feature sequence of shape (10, 43008).
        """
        feature_sequence = []
        for eeg_data in eeg_data_sequence:
            features = self.extract_features(eeg_data)  # Extract features for 1 second
            if features is not None:
                feature_sequence.append(features)  # Append the flattened feature vector
                #self.logger.info(f"[Extract Features Sequence] Extracted features for 1 second. Shape: {features.shape}")
        return np.array(feature_sequence)  # Shape: (10, 43008)

    def get_10_second_window(self):
        """
        Retrieve the last 10 seconds of EEG data for processing.
        :return: EEG data of shape (10, 14, 256).
        """
        if all(len(self.primary_buffer[channel]) >= 2560 for channel in self.channel_names):
            eeg_data_sequence = np.array([
                list(self.primary_buffer[channel])[-2560:].reshape(10, 256)
                for channel in self.channel_names
            ])
            return eeg_data_sequence.transpose(1, 0, 2)  # Shape: (10, 14, 256)
        else:
            self.logger.warning("Not enough data for a 10-second window.")
            return None

    def are_buffers_full(self, buffer_type="primary"):
            """
            Check if all buffers are full.
            Args:
                buffer_type (str): "primary" or "secondary".
            Returns:
                bool: True if all buffers are full, False otherwise.
            """
            buffer = self.primary_buffer if buffer_type == "primary" else self.secondary_buffer
            is_full = all(len(buffer[channel]) >= self.buffer_size for channel in self.channel_names)
            #self.logger.info(f"[Are Buffers Full] Buffer sizes: { {channel: len(buffer[channel]) for channel in self.channel_names} }")
            #self.logger.info(f"[Are Buffers Full] All buffers full: {is_full}")
            return is_full



    def start_streaming(self):
        """
        Continuously read data from the EEG headset and populate buffers.
        """
        self.logger.info("Starting EEG data streaming...")
        loop_count = 0  # Initialize loop counter

        if not self.connect():
            #self.logger.error("Failed to connect to the EEG device. Exiting streaming process.")
            return

        try:
            while not stop_main_loop.is_set():  # Check stop_main_loop event
                loop_count += 1
                #self.logger.info(f"[Streaming Loop] Iteration: {loop_count}")
                if stop_main_loop.is_set():  # Check before reading data
                    break
                data = self.read_emotiv_data()
                if stop_main_loop.is_set():  # Check after reading data
                    break
                if data:
                    # Log the Gyro data being added to the visualizer
                    if 'gyro_x' in data and 'gyro_y' in data:
                        #self.logger.info(f"Adding Gyro data to visualizer: gyro_x={data['gyro_x']}, gyro_y={data['gyro_y']}")
                        visualizer.update_gyro_data(data['gyro_x'], data['gyro_y'])

                    # Update EEG buffers
                    buffers_full = self.update_eeg_buffers(data)

                    # Check if buffers are full and process data
                    if isinstance(buffers_full, np.ndarray) and buffers_full.size > 0:

                        #self.logger.info("Buffers are full. Proceeding to data processing stage.")
                        processed_data = self.preprocess_eeg_data(eeg_data)
                        if processed_data is not None:
                            self.logger.info("Data processing complete. Proceeding to prediction.")

                else:
                    if stop_main_loop.is_set():  # Check during retry
                        break
                    self.logger.warning("No data received. Retrying...")
                    time.sleep(0.01)  # Small delay to avoid busy-waiting

        except KeyboardInterrupt:
            self.logger.info("Ctrl+C detected. Shutting down...")
        finally:
            self.disconnect()

    def get_latest_data(self):
        """
        Retrieves the latest EEG data from the buffers.
        Returns:
            dict: A dictionary containing the latest EEG data for each channel.
        """
        start_time = time.time()  # Record the start time
        timeout = 10  # Timeout in seconds to prevent infinite looping

        while not self.is_buffer_ready:
            #self.logger.info("Waiting for buffers to be fully populated...")
            self.is_buffer_ready = self.are_buffers_full("primary")  # Check buffer status
            if time.time() - start_time > timeout:
                #self.logger.error("Timeout while waiting for buffers to be fully populated.")
                return None  # Return None if timeout occurs
            time.sleep(0.1)  # Small delay to prevent busy-waiting

        # Retrieve the latest data
        latest_data = {channel: self.primary_buffer[channel][-1] if len(self.primary_buffer[channel]) > 0 else 0
                       for channel in self.channel_names}
        #self.logger.info(f"Latest EEG data retrieved: {latest_data, len(latest_data)}")
        #self.logger.info(f"Latest data shape: {np.array(list(latest_data.values())).shape}")
        return latest_data

    def get_buffer_data(self, buffer_type="primary"):
        """
        Retrieve data from the specified buffer.
        Args:
            buffer_type (str): "primary" or "secondary".
        Returns:
            dict: A dictionary containing buffer data for each channel.
        """
        buffer = self.primary_buffer if buffer_type == "primary" else self.secondary_buffer
        buffer_data = {channel: list(buffer[channel]) for channel in self.channel_names}
        #self.logger.info(f"Retrieved {buffer_type} buffer data: {[(ch, len(data)) for ch, data in buffer_data.items()]}")
        return buffer_data

    def get_buffer_sizes(self):
        """
        Retrieve the current sizes of the primary and secondary buffers for each channel.
        Returns:
            dict: A dictionary containing the sizes of the buffers for each channel.
        """
        buffer_sizes = {
            "primary": {channel: len(self.primary_buffer[channel]) for channel in self.channel_names},
            "secondary": {channel: len(self.secondary_buffer[channel]) for channel in self.channel_names}
        }
        #self.logger.info(f"Buffer sizes: {buffer_sizes}")
        return buffer_sizes

    

    def add_data(self, data):
        """
        Add raw EEG data to the buffers for visualization.
        """
        if isinstance(data, dict):  # Assuming data is a dictionary with channel names as keys
            for i, channel in enumerate(self.channel_names):
                if channel in data:
                    self.data_buffers[i].append(data[channel])
        elif isinstance(data, list):  # Handle raw data as a list
            for i in range(min(len(data), self.num_channels)):
                self.data_buffers[i].append(data[i])
        else:
            self.logger.warning("Unsupported data format for visualization.")  # Use self.logger instead of logging.warning

    def update_feature_window(self, feature_vector):
        """
        Add a 1-second feature vector to the 10-second feature window.
        """
        self.feature_window.append(feature_vector)
        #self.logger.info(f"[update_feature_window] Updated feature window. Current size: {len(self.feature_window)}")

    def get_feature_sequence(self):
        """
        Retrieve the 10-second feature sequence if the window is full.
        :return: Feature sequence of shape (10, feature_vector_length) or None.
        """
        if len(self.feature_window) == 10:
            #self.logger.info("[get_feature_sequence] Feature window is full. Returning feature sequence.")
            return np.array(self.feature_window)  # Shape: (10, feature_vector_length)
        else:
            return None

    def process_and_extract_features(self, eeg_data, feature_queue):
        """
        Process raw EEG data, extract features, and update the feature window.
        """
        try:
            # Preprocess the EEG data
            preprocessed_data = self.preprocess_eeg_data(eeg_data)
            if preprocessed_data is not None:
                # Extract features for 1 second
                feature_vector = self.extract_features(preprocessed_data)  # Shape: (14, feature_vector_length_per_channel)
                if feature_vector is not None:
                    # Update the feature window for LSTM
                    self.update_feature_window(feature_vector.flatten())  # Flatten for LSTM
                    #self.logger.info(f"[process_and_extract_features] Feature vector added. Feature window size: {len(self.feature_window)}")

                    # Send the 2D feature vector to the FeatureVisualizer
                    feature_queue.put(feature_vector)  # Keep as 2D for FeatureVisualizer
                    #self.logger.info("[process_and_extract_features] Feature vector sent to FeatureVisualizer.")
                else:
                    self.logger.warning("[process_and_extract_features] Feature vector is None. Skipping update to feature window.")
            else:
                self.logger.warning("[process_and_extract_features] Preprocessed data is None. Skipping feature extraction.")
        except Exception as e:
            self.logger.error(f"[process_and_extract_features] Error during feature processing: {e}")



if __name__ == "__main__":
    streamer = EmotivStreamer()
    #feature_visualizer = FeatureVisualizer(channel_names= ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"], fs=256)  # Initialize feature visualizer

    streamer.test_data_processing()
    if streamer.connect():
        try:
            streamer.start_streaming()
        except KeyboardInterrupt:
            streamer.logger.info("Streaming stopped by user.")
        finally:
            streamer.disconnect()
    else:
        streamer.logger.error("Failed to connect to the EEG device.")
