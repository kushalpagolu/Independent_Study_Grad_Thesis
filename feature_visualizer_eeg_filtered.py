import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureVisualizerEEGFiltered:
    def __init__(self, channel_names, fs=256):
        """
        Initialize the FeatureVisualizerEEGFiltered.
        :param channel_names: List of EEG channel names.
        :param fs: Sampling frequency of the EEG data.
        """
        self.channel_names = channel_names
        self.fs = fs
        self.num_channels = len(channel_names)
        self.samples_per_channel = 256  # Number of samples per channel (1 second of data)
        self.time_window = 10  # 10 seconds of data
        self.time_axis = np.linspace(-self.time_window, 0, self.time_window * self.fs)

        # Initialize data buffer for eeg_filtered
        self.feature_data = np.zeros((self.num_channels, self.time_window * self.samples_per_channel))

        # Create a single figure with subplots for all channels
        self.fig, self.axes = plt.subplots(self.num_channels, 1, figsize=(12, 14), sharex=True)
        self.fig.suptitle("EEG Feature Visualization: EEG Filtered")

        # Generate colors for each channel using the viridis colormap
        colors = plt.cm.viridis(np.linspace(0, 1, self.num_channels))

        # Initialize plot lines for each channel
        self.lines = []
        for i, ax in enumerate(self.axes):
            line, = ax.plot(self.time_axis, np.zeros_like(self.time_axis), label=f"Channel {channel_names[i]}", color=colors[i])
            self.lines.append(line)
            ax.set_title(f"Channel {channel_names[i]}")
            ax.set_xlim(-self.time_window, 0)
            ax.legend(loc="upper right")
        self.axes[-1].set_xlabel("Time (s)")

    def update_data(self, features_array):
        """
        Update the data buffer with new eeg_filtered values.
        :param features_array: Numpy array containing EEG filtered features.
        """
        try:
            if not isinstance(features_array, np.ndarray):
                logger.error(f"[FeatureVisualizerEEGFiltered] Expected features_array to be a numpy array, but got {type(features_array)}")
                return  # Skip invalid data
            # Log the shape of the incoming data
            #logger.debug(f"[FeatureVisualizerEEGFiltered] Shape of eeg_filtered_: {features_array.shape}")
            # Dynamically determine the number of samples per channel
            samples_per_channel = features_array.size // self.num_channels

            # Reshape features_array into (num_channels, samples_per_channel)
            reshaped_features = features_array[:self.num_channels * samples_per_channel].reshape(self.num_channels, samples_per_channel)

            # Update the feature data buffer
            self.feature_data = np.roll(self.feature_data, -samples_per_channel, axis=1)
            self.feature_data[:, -samples_per_channel:] = reshaped_features

            #logger.info(f"[FeatureVisualizerEEGFiltered] EEG filtered data updated successfully, Shape: {self.feature_data.shape}")
        except Exception as e:
            logger.error(f"Error updating data in FeatureVisualizerEEGFiltered: {e}")

    def update_plot(self, frame):
        """
        Update the plots with the latest data.
        """
        try:
            # Update each channel's plot
            for ch_idx, line in enumerate(self.lines):
                y_data = self.feature_data[ch_idx]
                x_data = self.time_axis[-y_data.shape[0]:]  # Ensure x_data matches y_data length
                line.set_data(x_data, y_data)
                """
                # Dynamically adjust y-axis limits
                min_val = np.min(y_data)
                max_val = np.max(y_data)
                if min_val == max_val:  # Handle case where all values are the same
                    min_val -= 1
                    max_val += 1
                self.axes[ch_idx].set_ylim(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val))
                """
            # Fixed y-axis limits for visualizing filtering impact
            for ax, _ in zip(self.axes, self.feature_data):
                ax.set_ylim(-5, 5)

            # Force redraw
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            logger.error(f"Error updating plots in FeatureVisualizerEEGFiltered: {e}")

    def start_animation(self):
        """
        Start the animation for live plotting.
        """
        try:
            #logger.debug("[FeatureVisualizerEEGFiltered] Starting animation.")

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
            logger.error(f"Error starting animation in FeatureVisualizerEEGFiltered: {e}")
