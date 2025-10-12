import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import mne
from mne.viz import Brain
import logging
import mne.datasets

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RealtimeEEGVisualizer:
    logger = logging.getLogger(__name__)

    def __init__(self, buffer_size=1000, num_channels=14):
        self.buffer_size = buffer_size
        self.num_channels = num_channels
        self.data_buffers = [deque(maxlen=buffer_size) for _ in range(num_channels)]
        self.gyro_x_buffer = deque(maxlen=buffer_size)
        self.gyro_y_buffer = deque(maxlen=buffer_size)

        self.channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
                              'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

        # MNE Brain setup
        self.subject = 'sample'
        self.data_path = mne.datasets.sample.data_path()
        self.subjects_dir = str(self.data_path / 'subjects')
        self.channel_to_label = {
            'AF3': 'lh.BA44.label', 'F7': 'lh.BA45.label', 'F3': 'lh.BA6.label',
            'FC5': 'lh.BA4a.label', 'T7': 'lh.BA4p.label', 'P7': 'lh.BA3b.label',
            'O1': 'lh.V1.label', 'O2': 'rh.V1.label', 'P8': 'rh.BA3b.label',
            'T8': 'rh.BA4p.label', 'FC6': 'rh.BA4a.label', 'F4': 'rh.BA6.label',
            'F8': 'rh.BA45.label', 'AF4': 'rh.BA44.label'
        }
        self.label_vertices = {ch: mne.read_label(f"{self.subjects_dir}/fsaverage/label/{label}").vertices
                               for ch, label in self.channel_to_label.items()}

        plt.style.use('dark_background')

        # Initialize 2D Figure for EEG and Gyro
        self.fig_2d, axes_2d = plt.subplots(15, 1, figsize=(20, 16), gridspec_kw={'height_ratios': [1] * 14 + [2]})
        self.ax_eeg = axes_2d[:14]
        self.ax_gyro = axes_2d[14]

        colors = plt.cm.viridis(np.linspace(0, 1, num_channels))
        self.lines = [self.ax_eeg[i].plot([], [], label=self.channel_names[i], color=colors[i])[0] for i in range(self.num_channels)]

        for i, ax in enumerate(self.ax_eeg):
            ax.set_ylabel(self.channel_names[i])
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.4)
            ax.set_ylim(-25000, 25000)

        self.ax_eeg[0].set_title("Real-time EEG Signals", fontsize=14)

        self.scatter_gyro, = self.ax_gyro.plot([], [], 'wo', markersize=4, alpha=0.6, label="Head Movement")
        self.line_gyro_traj, = self.ax_gyro.plot([], [], 'c-', linewidth=1, alpha=0.8, label="Trajectory")
        self.ax_gyro.set_title("Real-time Head Movement (Gyro)", fontsize=14)
        self.ax_gyro.set_xlim(0, 150)
        self.ax_gyro.set_ylim(0, 150)
        self.ax_gyro.legend()
        plt.subplots_adjust(hspace=0.2)

        # Initialize MNE Brain object
        brain_kwargs = dict(alpha=0.5, background="white", cortex="low_contrast")
        self.brain = Brain("sample", hemi="both", surf="pial", subjects_dir=self.subjects_dir, **brain_kwargs)
        self.brain.show_view(azimuth=190, elevation=70, distance=350, focalpoint=(0, 0, 20))


    def add_data(self, data):
        """Adds a raw data packet to the internal buffers."""
        if isinstance(data, dict):
            for i, channel in enumerate(self.channel_names):
                if channel in data:
                    self.data_buffers[i].append(data[channel])

            if 'gyro_x' in data and 'gyro_y' in data:
                self.gyro_x_buffer.append(data['gyro_x'])
                self.gyro_y_buffer.append(data['gyro_y'])
        else:
            self.logger.warning("Unsupported data format received.")


    def update_all_plots(self, frame):
        """
        A single method to update all visual components (2D and 3D).
        This will be called from the main animation loop in main_thread.py.
        """
        # --- Update 2D Plots ---
        for i, line in enumerate(self.lines):
            if self.data_buffers[i]:
                line.set_data(range(len(self.data_buffers[i])), self.data_buffers[i])
                self.ax_eeg[i].relim()
                self.ax_eeg[i].autoscale_view()

        if len(self.gyro_x_buffer) > 1:
            self.scatter_gyro.set_data(self.gyro_x_buffer, self.gyro_y_buffer)
            self.line_gyro_traj.set_data(self.gyro_x_buffer, self.gyro_y_buffer)
            self.ax_gyro.relim()
            self.ax_gyro.autoscale_view()

        # --- Update 3D Brain Plot ---
        try:
            lh_data = np.zeros(len(self.brain.geo['lh'].coords))
            rh_data = np.zeros(len(self.brain.geo['rh'].coords))

            for i, channel in enumerate(self.channel_names):
                if self.data_buffers[i]:
                    # Use a simple average of the buffer for visualization
                    avg_data = np.mean(self.data_buffers[i])
                    vertices = self.label_vertices[channel]

                    if 'lh' in self.channel_to_label[channel]:
                        valid_vertices = vertices[vertices < len(lh_data)]
                        lh_data[valid_vertices] = avg_data
                    else:
                        valid_vertices = vertices[vertices < len(rh_data)]
                        rh_data[valid_vertices] = avg_data

            self.brain.add_data(lh_data, hemi="lh", smoothing_steps=10)
            self.brain.add_data(rh_data, hemi="rh", smoothing_steps=10)
        except Exception as e:
            self.logger.error(f"Error updating 3D brain visualization: {e}")
