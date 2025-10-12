import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HjorthVisualizer:
    def __init__(self, channel_names, fs=256):
        """
        Initialize the HjorthVisualizer.
        :param channel_names: List of EEG channel names.
        :param fs: Sampling frequency of the EEG data.
        """
        self.channel_names = channel_names
        self.fs = fs
        self.time_window = 10  # 10 seconds of data
        self.time_axis = np.linspace(-self.time_window, 0, self.time_window * self.fs)

        # Initialize data buffers for mobility and complexity
        self.mobility_data = np.zeros((len(channel_names), self.time_window * self.fs))
        self.complexity_data = np.zeros((len(channel_names), self.time_window * self.fs))

        # Create a single figure with subplots for all channels
        self.fig, self.axes = plt.subplots(len(channel_names), 1, figsize=(16, 14), sharex=True)
        self.fig.suptitle("EEG Hjorth Parameters Visualization (Mobility and Complexity)")

        # Generate colors for each channel using the plasma colormap
        colors = plt.cm.plasma(np.linspace(0, 1, len(channel_names)))

        # Initialize bar plots for each channel
        self.mobility_bars = []
        self.complexity_bars = []
        for i, ax in enumerate(self.axes):
            # Mobility and Complexity bar plots
            bar_width = 0.4
            mobility_bar = ax.bar(0, 0, width=bar_width, label=f"Mobility ({channel_names[i]})", color=colors[i], align='center')
            complexity_bar = ax.bar(1, 0, width=bar_width, label=f"Complexity ({channel_names[i]})", color=colors[i], alpha=0.6, align='center')
            self.mobility_bars.append(mobility_bar)
            self.complexity_bars.append(complexity_bar)

            ax.set_title(f"Channel - {channel_names[i]}")
            ax.set_xlim(-0.5, 1.5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Mobility", "Complexity"])
            ax.legend(loc="upper right")

        self.axes[-1].set_xlabel("Features")

    def update_data(self, hjorth_features):
        """
        Update the data buffers with new Hjorth feature values.
        :param hjorth_features: Numpy array containing mobility and complexity for all channels.
                                Shape: (num_channels * 2,)
        """
        try:
            if not isinstance(hjorth_features, np.ndarray):
                logger.error(f"[HjorthVisualizer] Expected hjorth_features to be a numpy array, but got {type(hjorth_features)}")
                return  # Skip invalid data

            if hjorth_features.shape[0] != len(self.channel_names) * 2:
                logger.error(f"[HjorthVisualizer] Expected hjorth_features length {len(self.channel_names) * 2}, but got {hjorth_features.shape[0]}")
                return  # Skip invalid data

            #logger.info(f"[HjorthVisualizer] Updating data with hjorth_features of shape: {hjorth_features.shape}")

            # Split the features into mobility and complexity
            mobility_features = hjorth_features[:len(self.channel_names)]
            complexity_features = hjorth_features[len(self.channel_names):]

            # Update the data buffers
            self.mobility_data[:, -1] = mobility_features
            self.complexity_data[:, -1] = complexity_features

            #logger.info("[HjorthVisualizer] Hjorth data buffers updated successfully.")
        except Exception as e:
            logger.error(f"Error updating Hjorth data in HjorthVisualizer: {e}")

    def update_plot(self, frame):
        """
        Update the plots with the latest Hjorth feature data.
        """
        try:
            #logger.debug("[HjorthVisualizer] update_plot method called.")

            # Update each channel's mobility and complexity bar plots
            for ch_idx, (mobility_bar, complexity_bar) in enumerate(zip(self.mobility_bars, self.complexity_bars)):
                # Update mobility bar
                mobility_bar[0].set_height(self.mobility_data[ch_idx, -1])

                # Update complexity bar
                complexity_bar[0].set_height(self.complexity_data[ch_idx, -1])

                # Set fixed y-axis limits for better visual sense
                self.axes[ch_idx].set_ylim(-2, 2)

            # Force redraw
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            #logger.info("[HjorthVisualizer] Plots updated successfully.")
        except Exception as e:
            logger.error(f"Error updating plots in HjorthVisualizer: {e}")

    def start_animation(self):
        """
        Start the animation for live plotting.
        """
        try:
            #logger.debug("[HjorthVisualizer] Starting animation.")

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
            logger.error(f"Error starting animation in HjorthVisualizer: {e}")
