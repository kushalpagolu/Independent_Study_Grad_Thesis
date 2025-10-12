import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureVisualizerEntropyFeatures:
    def __init__(self, channel_names, fs=256):
        """
        Initialize the FeatureVisualizerEntropyFeatures with a bar chart.
        :param channel_names: List of EEG channel names.
        :param fs: Sampling frequency of the EEG data.
        """
        self.channel_names = channel_names
        self.fs = fs
        self.num_channels = len(channel_names)

        # Initialize data buffer for entropy features (1 value per channel)
        self.feature_data = np.zeros(self.num_channels)

        # Create a figure for the bar chart
        self.fig, self.ax = plt.subplots(figsize=(12, 6))

        # Use the plasma colormap to assign unique colors to each channel
        colors = plt.cm.plasma(np.linspace(0, 1, self.num_channels))
        self.bars = self.ax.bar(self.channel_names, self.feature_data, color=colors)

        # Configure the plot
        self.ax.set_title("EEG Entropy Features (Bar Chart)")
        self.ax.set_xlabel("Channels")
        self.ax.set_ylabel("Entropy Value")
        self.ax.set_ylim(-2, 2)  # Initial Y-axis limits

    def update_data(self, entropy_features):
        """
        Update the data buffer with new entropy feature values.
        :param entropy_features: Numpy array containing entropy features for all channels.
        """
        try:
            if entropy_features is None or not isinstance(entropy_features, np.ndarray):
                logger.warning("[FeatureVisualizerEntropyFeatures] Invalid data received. Skipping update.")
                return  # Skip invalid data

            # Ensure the data has the correct number of channels
            if entropy_features.size != self.num_channels:
                logger.error(f"[FeatureVisualizerEntropyFeatures] Invalid shape for entropy_features: {entropy_features.shape}")
                return  # Skip invalid data

            # Update the feature data buffer
            self.feature_data = entropy_features

            # Log the updated feature data
            #logger.debug(f"[FeatureVisualizerEntropyFeatures] Updated feature_data: {self.feature_data}")
        except Exception as e:
            logger.error(f"Error updating entropy data in FeatureVisualizerEntropyFeatures: {e}")

    def update_plot(self, frame):
        """
        Update the bar chart with the latest entropy feature data.
        """
        try:
            # Update the height of each bar
            for bar, new_height in zip(self.bars, self.feature_data):
                bar.set_height(new_height)

            # Dynamically adjust the Y-axis limits
            min_val = np.min(self.feature_data)
            max_val = np.max(self.feature_data)
            if min_val == max_val:  # Handle case where all values are the same
                min_val -= 1
                max_val += 1
            self.ax.set_ylim(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val))

            # Redraw the canvas
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            logger.error(f"Error updating bar chart in FeatureVisualizerEntropyFeatures: {e}")

    def start_animation(self):
        """
        Start the animation for live plotting.
        """
        try:
            self.animation = FuncAnimation(
                self.fig,
                self.update_plot,
                frames=np.arange(0, 1000),  # Provide a frame source
                interval=100,  # Update every 100 ms
                blit=False
            )
            plt.show()  # Show the plot without blocking
        except Exception as e:
            logger.error(f"Error starting animation in FeatureVisualizerEntropyFeatures: {e}")
