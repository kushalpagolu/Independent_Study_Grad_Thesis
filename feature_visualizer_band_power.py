import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import logging
from sklearn.exceptions import ConvergenceWarning  # Import ConvergenceWarning
import os
import pandas as pd
logger = logging.getLogger(__name__)

class FeatureVisualizer:
    def __init__(self, channel_names, fs=256):
        """
        Initialize the FeatureVisualizer with a bar chart.
        :param channel_names: List of EEG channel names.
        :param fs: Sampling frequency of the EEG data.
        """
        self.channel_names = channel_names
        self.fs = fs
        self.num_channels = len(channel_names)
        self.num_bands = 5  # delta, theta, alpha, beta, gamma

        # Initialize data buffer for band power (channels × bands)
        self.band_power_data = np.zeros((self.num_channels, self.num_bands))

        # Create a figure for the bar chart
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.bar_width = 0.15  # Width of each bar
        self.x_indices = np.arange(self.num_channels)  # X positions for channels

        # Initialize bars for each frequency band
        self.bars = []
        self.band_labels = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Modern color palette
        for i, band in enumerate(self.band_labels):
            bars = self.ax.bar(
                self.x_indices + i * self.bar_width,
                self.band_power_data[:, i],
                self.bar_width,
                label=band,
                color=self.colors[i]
            )
            self.bars.append(bars)

        # Configure the plot
        self.ax.set_title("EEG Band Power Visualization (Bar Chart)")
        self.ax.set_xlabel("Channels")
        self.ax.set_ylabel("Band Power")
        self.ax.set_xticks(self.x_indices + self.bar_width * (self.num_bands - 1) / 2)
        self.ax.set_xticklabels(channel_names)
        self.ax.legend(title="Frequency Bands")

    def update_data(self, band_power_features):
        """
        Update the data buffer with new band power values.
        :param band_power_features: Numpy array of shape (70,).
        """
        try:
            if band_power_features is None or not isinstance(band_power_features, np.ndarray):
                logger.warning("[FeatureVisualizer] Invalid data received. Skipping update.")
                return  # Skip invalid data

            # Ensure the data has the correct size
            expected_size = self.num_channels * self.num_bands
            if band_power_features.size != expected_size:
                logger.error(f"[FeatureVisualizer] Invalid shape for band_power_features: {band_power_features.shape}")
                return  # Skip invalid data

            # Reshape the data into (channels × bands)
            reshaped_features = band_power_features.reshape(self.num_channels, self.num_bands)
            # logger.debug(f"[FeatureVisualizer] Reshaped band_power_features: {reshaped_features}")

            # Update the band power data buffer
            self.band_power_data = reshaped_features
            # logger.info("[FeatureVisualizer] Band power data buffers updated successfully.")
        except Exception as e:
            logger.error(f"Error updating band power data in FeatureVisualizer: {e}")

    def update_plot(self, frame):
        """
        Update the bar chart with the latest band power data.
        """
        try:
            # Update the height of each bar
            for i, bars in enumerate(self.bars):
                for bar, new_height in zip(bars, self.band_power_data[:, i]):
                    # logger.debug(f"[FeatureVisualizer] Updating bar for band '{self.band_labels[i]}' with height: {new_height}")
                    bar.set_height(new_height)

            # Dynamically adjust the Y-axis limits
            min_val = np.min(self.band_power_data)
            max_val = np.max(self.band_power_data)
            if min_val == max_val:  # Handle case where all values are the same
                min_val -= 1
                max_val += 1
            self.ax.set_ylim(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val))
            # logger.debug(f"[FeatureVisualizer] Y-axis limits set to: ({min_val - 0.1 * abs(min_val)}, {max_val + 0.1 * abs(max_val)})")

            # Redraw the canvas
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            # logger.info("[FeatureVisualizer] Bar chart updated successfully.")
        except Exception as e:
            logger.error(f"Error updating bar chart in FeatureVisualizer: {e}")

    def start_animation(self):
        """
        Start the animation for live plotting.
        """
        try:
            #logger.debug("[FeatureVisualizer] Starting animation.")

            # Assign the animation object to an instance variable to prevent garbage collection
            self.animation = FuncAnimation(
                self.fig,
                self.update_plot,
                frames=np.arange(0, 1000),  # Provide a frame source
                interval=100,  # Update every 100 ms
                blit=False
            )
            plt.show()  # Show the plot without blocking
        except Exception as e:
            logger.error(f"Error starting animation in FeatureVisualizer: {e}")
