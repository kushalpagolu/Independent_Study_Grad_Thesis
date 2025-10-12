import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureVisualizerFirstOrderDerivative:
    def __init__(self, channel_names, fs=256):
        """
        Initialize the FeatureVisualizerFirstOrderDerivative.
        :param channel_names: List of EEG channel names.
        :param fs: Sampling frequency of the EEG data.
        """
        self.channel_names = channel_names
        self.fs = fs
        self.num_channels = len(channel_names)
        self.samples_per_channel = 256  # Number of samples per channel (1 second of data)
        self.time_window = 10  # 10 seconds of data
        self.time_axis = np.linspace(-self.time_window, 0, self.time_window * self.fs)

        # Initialize data buffer for first-order derivatives
        self.featurefirst_order_derivatives_data = np.zeros((len(channel_names), self.time_window * self.fs))

        # Create a single figure with subplots for all 14 channels
        self.fig, self.axes = plt.subplots(len(channel_names), 1, figsize=(12, 14), sharex=True)
        self.fig.suptitle("EEG Feature Visualization: First-Order Derivatives")

        # Generate colors for each channel using the viridis colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(channel_names)))

        # Initialize plot lines for each channel
        self.lines = []
        for i, ax in enumerate(self.axes):
            line, = ax.plot(self.time_axis, np.zeros_like(self.time_axis), label=f"Channel {channel_names[i]}", color=colors[i])
            self.lines.append(line)
            ax.set_title(f"Channel {channel_names[i]}")
            ax.set_xlim(-self.time_window, 0)
            ax.legend(loc="upper right")
        self.axes[-1].set_xlabel("Time (s)")

    def update_data(self, first_order_derivatives):
        """
        Update the data buffer with new first-order derivative values.
        :param first_order_derivatives: Numpy array containing first-order derivatives.
        """
        try:
            if (first_order_derivatives is None or
                not isinstance(first_order_derivatives, np.ndarray)):
                logger.warning("[FeatureVisualizerFirstOrderDerivative] Invalid data received. Skipping update.")
                return  # Skip invalid data

            # Log the shape of the incoming data
            #logger.debug(f"[FeatureVisualizerFirstOrderDerivative] Shape of first_order_derivatives: {first_order_derivatives.shape}")

            # Dynamically determine the number of samples per channel
            samples_per_channel = first_order_derivatives.size // self.num_channels

            if samples_per_channel == 0:
                logger.warning("[FeatureVisualizerFirstOrderDerivative] Insufficient data for all channels. Skipping update.")
                return  # Skip if data is insufficient

            # Reshape first_order_derivatives into (num_channels, samples_per_channel)
            reshaped_features = first_order_derivatives[:self.num_channels * samples_per_channel].reshape(self.num_channels, samples_per_channel)

            # Handle rolling buffer
            self.featurefirst_order_derivatives_data = np.roll(self.featurefirst_order_derivatives_data, -samples_per_channel, axis=1)
            self.featurefirst_order_derivatives_data[:, -samples_per_channel:] = reshaped_features  # Add new data to the end

            #logger.info("[FeatureVisualizerFirstOrderDerivative] First-order derivative data buffers updated successfully.")
        except Exception as e:
            logger.error(f"Error updating data in FeatureVisualizerFirstOrderDerivative: {e}")

    def update_plot(self, frame):
        """
        Update the plots with the latest first-order derivative data.
        """
        try:
            #logger.debug("[FeatureVisualizerFirstOrderDerivative] update_plot method called.")

            # Update plots for each channel
            for ch_idx, line in enumerate(self.lines):
                y_data = self.featurefirst_order_derivatives_data[ch_idx]
                x_data = self.time_axis[-len(y_data):]  # Ensure x_data matches y_data length
                line.set_data(x_data, y_data)
            
            # Fixed y-axis limits for visualizing filtering impact
            # Set fixed y-axis limits for all channels
            # Set fixed y-axis limits per subplot using correct pairing
            for ax, _ in zip(self.axes, self.featurefirst_order_derivatives_data):
                ax.set_ylim(-25000, 25000)

            # Force redraw
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            #logger.info(f"[FeatureVisualizerFirstOrderDerivative] Plots updated successfully, Shape: {self.featurefirst_order_derivatives_data.shape}")
        except Exception as e:
            logger.error(f"Error updating plots in FeatureVisualizerFirstOrderDerivative: {e}")

    def start_animation(self):
        """
        Start the animation for live plotting.
        """
        try:
            #logger.debug("[FeatureVisualizerFirstOrderDerivative] Starting animation.")

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
            logger.error(f"Error starting animation in FeatureVisualizerFirstOrderDerivative: {e}")
