import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FractalVisualizer:
    def __init__(self, channel_names, fs=256, time_window=10):
        """
        Real-time heatmap of EEG fractal features.
        New data appears at the top (frame index), scrolling downward off the bottom as new frames arrive.
        Y-axis tick labels dynamically show frame numbers (age) for each row.
        :param channel_names: List of EEG channel names.
        :param fs: Sampling rate (unused).
        :param time_window: Number of frames (rows) to display.
        """
        self.channel_names = channel_names
        self.num_channels = len(channel_names)
        self.time_window = time_window

        # Buffer: rows=frames, cols=channels
        # New frame at row 0 (top), older scroll down
        self.fractal_buffer = np.zeros((time_window, self.num_channels))
        self.frame_count = -1  # counts inserted frames

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle("EEG Fractal Features Heatmap", fontsize=16, fontweight="bold", color="navy")

        # Initialize heatmap: origin='upper' places index 0 at top
        self.heatmap = self.ax.imshow(
            self.fractal_buffer,
            aspect="auto",
            cmap="coolwarm",
            origin="upper",
            interpolation="nearest"
        )

        # Configure X-axis for channels
        self.ax.set_xlabel("Channels", fontsize=12, color="white")
        self.ax.set_xticks(np.arange(self.num_channels))
        self.ax.set_xticklabels(self.channel_names, rotation=45, ha="right", fontsize=10, color="white")
        self.ax.tick_params(axis="x", colors="white")

        # Configure Y-axis
        self.ax.set_ylabel("Frame Index", fontsize=12, color="white")
        # initial static limits
        self.ax.set_ylim(-0.5, self.time_window - 0.5)

        # Add colorbar
        self.colorbar = self.fig.colorbar(self.heatmap, ax=self.ax)
        self.colorbar.set_label("Fractal Intensity", fontsize=12, color="white")

    def update_data(self, fractal_features):
        """
        Insert a new fractal feature vector at the top, rolling older data downward.
        Updates the internal frame counter.
        :param fractal_features: 1D numpy array of length num_channels.
        """
        try:
            if fractal_features is None or not isinstance(fractal_features, np.ndarray):
                logger.warning("[FractalVisualizer] Invalid data received. Skipping update.")
                return
            if fractal_features.size != self.num_channels:
                logger.error(f"[FractalVisualizer] Expected {self.num_channels} values, got {fractal_features.size}")
                return

            # Increment frame count
            self.frame_count += 1

            # Shift buffer down by 1 row; row 0 becomes new frame
            self.fractal_buffer = np.roll(self.fractal_buffer, 1, axis=0)
            self.fractal_buffer[0, :] = fractal_features
        except Exception as e:
            logger.error(f"Error updating fractal buffer: {e}")

    def update_plot(self, frame=None):
        """
        Refresh the heatmap and dynamic Y-axis labels.
        """
        try:
            # Update heatmap data array
            self.heatmap.set_array(self.fractal_buffer)

            # Dynamic color scaling
            vmin, vmax = np.nanmin(self.fractal_buffer), np.nanmax(self.fractal_buffer)
            if vmax > vmin:
                self.heatmap.set_clim(vmin, vmax)

            # Dynamic Y-ticks: label each row with its frame number
            tick_positions = np.arange(self.time_window)
            # Newest at position 0, so label = frame_count - i
            labels = []
            for i in tick_positions:
                age = self.frame_count - i
                labels.append(str(age) if age >= 0 else "")
            self.ax.set_yticks(tick_positions)
            self.ax.set_yticklabels(labels, fontsize=10, color="white")

            # Redraw canvas
            self.fig.canvas.draw_idle()
        except Exception as e:
            logger.error(f"Error refreshing fractal heatmap: {e}")

    def start_animation(self):
        """
        Launch live updating of the heatmap.
        """
        try:
            self.animation = FuncAnimation(
                self.fig,
                self.update_plot,
                interval=100,  # milliseconds
                blit=False
            )
            plt.show()
        except Exception as e:
            logger.error(f"Error starting fractal animation: {e}")
