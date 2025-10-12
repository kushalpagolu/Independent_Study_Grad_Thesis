import hid
import numpy as np
from Crypto.Cipher import AES
from datetime import datetime
import threading
import queue
import logging
import os
import time
from collections import deque
from feature_extraction import (
    apply_bandpass_filter, apply_notch_filter, common_average_reference,
    apply_ica, apply_anc, apply_hanning_window, apply_dwt_denoising,
    compute_band_power, compute_hjorth_parameters, compute_spectral_entropy,
    higuchi_fractal_dimension, normalize_features
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

        # Single buffer to hold incoming data, acting as a small backlog.
        # It holds up to 5 seconds of data to prevent loss if the preprocessor is busy.
        self.buffer_size = self.fs * 5
        self.eeg_buffer = self.initialize_buffers(self.channel_names, self.buffer_size)
        
        # This deque will store the sequence of 1-second feature vectors for the LSTM
        self.feature_window = deque(maxlen=10)

        # Preprocessing flags
        self.reference_channel = 0
        self.use_ica = True
        self.use_dwt = True
        self.use_hfd = True
        self.use_bandpass = True
        self.use_hjorth = True
        self.use_entropy = True
        self.use_bandpower = True
        self.use_hanning = True
        self.use_anc = False

    def initialize_buffers(self, channel_names, buffer_size):
        """Initializes rolling buffers for EEG data using deques."""
        return {ch: deque(maxlen=buffer_size) for ch in channel_names}

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
        """Reads and decrypts a single data packet from the Emotiv device."""
        try:
            encrypted = bytes(self.device.read(32))
            if len(encrypted) != 32:
                return None  # Invalid or empty packet

            decrypted = self.cipher.decrypt(encrypted)
            packet = {'timestamp': datetime.now().isoformat()}
            for i, channel_name in enumerate(self.channel_names):
                start_idx = 2 * i + 1
                packet[channel_name] = int.from_bytes(decrypted[start_idx:start_idx+2], 'big', signed=True)
            
            packet['gyro_x'] = decrypted[29]
            packet['gyro_y'] = decrypted[30]
            return packet
        except Exception as e:
            self.logger.error(f"Error reading packet: {e}")
            return None

    def update_eeg_buffers(self, raw_data):
        """
        Updates the EEG buffer with a new raw packet.
        If a full 1-second (256 samples) chunk is available, it returns the chunk
        for processing and removes it from the buffer.
        """
        # Append new data to the buffer
        for channel in self.channel_names:
            if channel in raw_data:
                self.eeg_buffer[channel].append(raw_data[channel])

        # Check if we have a full 1-second chunk ready
        if len(self.eeg_buffer[self.channel_names[0]]) >= self.fs:
            eeg_data_list = []
            for ch in self.channel_names:
                channel_data = []
                # Efficiently pop the oldest 256 samples from the left
                for _ in range(self.fs):
                    channel_data.append(self.eeg_buffer[ch].popleft())
                eeg_data_list.append(channel_data)
            
            return np.array(eeg_data_list) # Shape: (14, 256)
        
        # Not enough data for a full chunk yet
        return None

    def preprocess_eeg_data(self, eeg_data):
        """Preprocesses a 1-second chunk of EEG data."""
        try:
            if eeg_data.shape != (14, 256):
                self.logger.warning(f"Invalid shape for preprocessing: {eeg_data.shape}. Skipping.")
                return None

            noise_ref = np.mean(eeg_data, axis=0, keepdims=True)
            eeg_data = apply_notch_filter(eeg_data, fs=self.fs)
            eeg_data = apply_bandpass_filter(eeg_data, lowcut=1.0, highcut=50.0, sampling_rate=self.fs)
            eeg_data = common_average_reference(eeg_data)
            if self.use_anc: eeg_data = apply_anc(eeg_data, noise_ref)
            if self.use_ica: eeg_data = apply_ica(eeg_data)
            eeg_data = apply_hanning_window(eeg_data)
            if self.use_dwt: eeg_data = apply_dwt_denoising(eeg_data)
            return eeg_data
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}", exc_info=True)
            return None

    def extract_features(self, eeg_filtered):
        """Extracts a comprehensive feature vector from a preprocessed 1-second chunk."""
        try:
            features = []
            # Band Power (14 channels * 5 bands = 70 features)
            for i in range(14):
                features.extend(list(compute_band_power(eeg_filtered[i], fs=self.fs).values()))
            
            # Hjorth Parameters (14 channels * 2 params = 28 features)
            for i in range(14):
                features.extend(compute_hjorth_parameters(eeg_filtered[i]))

            # Spectral Entropy (14 channels * 1 value = 14 features)
            for i in range(14):
                features.append(compute_spectral_entropy(eeg_filtered[i], self.fs))

            # Fractal Dimension (14 channels * 1 value = 14 features)
            for i in range(14):
                features.append(higuchi_fractal_dimension(eeg_filtered[i]))
            
            # Normalize the statistical features computed so far
            # Total statistical features = 70 + 28 + 14 + 14 = 126
            statistical_features = np.array(features).reshape(1, -1)
            normalized_statistical_features = normalize_features(statistical_features).flatten()

            # Temporal Derivatives
            first_order = np.diff(eeg_filtered, axis=1, prepend=eeg_filtered[:, :1])
            second_order = np.diff(first_order, axis=1, prepend=first_order[:, :1])
            
            # Concatenate all features into a single vector
            feature_vector = np.concatenate((
                normalized_statistical_features,
                normalize_features(first_order).flatten(),
                normalize_features(second_order).flatten(),
                normalize_features(eeg_filtered).flatten()
            ))
            return feature_vector
        except Exception as e:
            self.logger.error(f"Error during feature extraction: {e}", exc_info=True)
            return None

    def update_feature_window(self, feature_vector):
        """Adds a new 1-second feature vector to the 10-second rolling window."""
        self.feature_window.append(feature_vector)
        self.logger.debug(f"Feature window updated. Size: {len(self.feature_window)}/10")

    def get_feature_sequence(self):
        """
        Returns the 10-second feature sequence if the window is full.
        Otherwise, returns None.
        """
        if len(self.feature_window) == 10:
            return np.array(self.feature_window)
        return None
