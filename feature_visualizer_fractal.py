import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FractalVisualizer:
    def __init__(self, channel_names, fs=256, time_window=10):
        """
        Initialize the FractalVisualizer with a heatmap for real-time streaming.
        :param channel_names: List of EEG channel names.
        :param fs: Sampling frequency of the EEG data.
        :param time_window: Time window (in seconds) for the rolling window.
        """
        self.channel_names = channel_names
        self.fs = fs
        self.num_channels = len(channel_names)
        self.time_window = time_window
        self.samples_per_window = time_window * fs

        # Initialize data buffer for fractal features (rolling buffer)
        self.fractal_data = np.zeros((self.num_channels, self.samples_per_window))
        #print(f"Size of fractal_data: {self.fractal_data.shape}")
        #print(f"Sample value of fractal_data: {self.fractal_data[0, 0]}")

        # Create a figure for the heatmap
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle("EEG Fractal Features Heatmap (Real-Time)", fontsize=16, fontweight="bold", color="navy")

        # Initialize the heatmap
        self.heatmap = self.ax.imshow(
            self.fractal_data,
            aspect="auto",
            cmap="coolwarm",  # Use a color range for visualization
            extent=[0, 50, 0, self.num_channels],  # X-axis: sample indices, Y-axis: channels
            origin="lower",
            interpolation="spline36"
        )

        # Configure the axes
        self.ax.set_xlabel("Sample Index", fontsize=12, color="white")
        self.ax.set_ylabel("EEG Channels", fontsize=12, color="white")
        self.ax.set_yticks(np.arange(self.num_channels))
        self.ax.set_yticklabels(self.channel_names, fontsize=10, color="white")
        self.ax.tick_params(axis="x", labelsize=10, colors="white")
        self.ax.tick_params(axis="y", labelsize=10, colors="white")

        # Add a colorbar to represent the fractal feature intensity
        self.colorbar = self.fig.colorbar(self.heatmap, ax=self.ax)
        self.colorbar.set_label("Fractal Feature Intensity", fontsize=12, color="white")

    def update_data(self, fractal_features):
        """
        Update the data buffer with new fractal feature values.
        :param fractal_features: Numpy array containing fractal features for all channels.
        """
        try:
            if fractal_features is None or not isinstance(fractal_features, np.ndarray):
                logger.warning("[FractalVisualizer] Invalid data received. Skipping update.")
                return

            # Ensure the data has the correct number of channels
            if fractal_features.shape[0] != self.num_channels:
                logger.error(f"[FractalVisualizer] Invalid shape for fractal_features: {fractal_features.shape}. Expected: ({self.num_channels},)")
                return

            # Shift the buffer to the left and add the new frame
            self.fractal_data = np.roll(self.fractal_data, -1, axis=1)
            self.fractal_data[:, -1] = fractal_features  # Add new data to the last column
            self.fractal_data = self.fractal_data * 1000  # Scale the data for better visualization


            #print(f"Updated fractal_data shape: {self.fractal_data.shape}")
            #print(f"Sample fractal_data values: {self.fractal_data[:, -5:]}")  # Print last 5 columns for debugging
        except Exception as e:
            logger.error(f"Error updating fractal data in FractalVisualizer: {e}")

    def update_plot(self, frame):
        """
        Update the heatmap with the latest fractal feature data.
        """
        try:
            # Update the heatmap data
            self.heatmap.set_array(self.fractal_data)
            #print(f"Fractal data shape in update_plot: {self.fractal_data.shape}")
            #print(f"Fractal data sample values in update_plot: {self.fractal_data[:, -5:]}")  # Debugging

            # Keep the extent fixed to the sample indices
            self.heatmap.set_extent([0, 50, 0, self.num_channels])  # X-axis: sample indices

            # Redraw the canvas to reflect the updated data
            self.fig.canvas.draw_idle()
        except Exception as e:
            logger.error(f"Error updating heatmap in FractalVisualizer: {e}")

    def start_animation(self):
        """
        Start the animation for live plotting.
        """
        try:
            self.animation = FuncAnimation(
                self.fig,
                self.update_plot,
                frames=np.arange(0, 1000),  # Provide a frame source
                interval=10,  # Update every 10 ms
                blit=False
            )
            plt.show()
        except Exception as e:
            logger.error(f"Error starting animation in FractalVisualizer: {e}")