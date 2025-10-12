import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging
from feature_visualizer_band_power import FeatureVisualizer  # Import FeatureVisualizer
from feature_visualizer_first_order_derivative import FeatureVisualizerFirstOrderDerivative  # Import FeatureVisualizerAllChannels
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from feature_visualizer_fractal import FractalVisualizer  # Import FractalVisualizer
from feature_visualizer_entropy_features import FeatureVisualizerEntropyFeatures  # Import FeatureVisualizerEntropyFeatures
from feature_visualizer_second_order_derivatives import FeatureVisualizerSecondOrderDerivatives  # Import FeatureVisualizerSecondOrderDerivatives
from feature_visualizer_eeg_filtered import FeatureVisualizerEEGFiltered  # Import FeatureVisualizerEEGFiltered
from feature_visualizer_hjorth_params import HjorthVisualizer  # Import FeatureVisualizerHjorthParameters
import numpy as np
import threading
from shared_events import visualization_ready



def run_visualizations_on_main_thread(visualizer, visualization_queue, feature_visualizer, feature_queue, derivatives_visualizer, second_order_derivatives_visualizer, fractal_visualizer, entropy_visualizer, eeg_filtered_visualizer, hjorth_visualizer):
    """
    Run the 3D EEG, 2D EEG, and feature visualizations on the main thread.
    """
    def update_3d_visualization(frame):
        while not visualization_queue.empty():
            packet = visualization_queue.get()
            visualizer.add_data(packet)  # Add data to the 3D visualizer
        visualizer.update_3d(frame)  # Update the 3D visualization

    def update_2d_visualization(frame):
        while not visualization_queue.empty():
            packet = visualization_queue.get()
            visualizer.add_data(packet)  # Add data to the 2D visualizer
        visualizer.update_2d(frame)  # Update the 2D visualization

    def update_feature_visualization(frame):
        """
        Update the feature visualizer with band power and first-order derivative features.
        """
        if feature_queue.empty():
            #logging.debug("[update_feature_visualization] feature_queue is empty.")
            return  # Skip if the queue is empty
        else:
            try:
                features_array = feature_queue.get()
                logging.debug(f"[update_feature_visualization] Features array dequeued with shape: {features_array.shape}")

                # Extract band power features (first 70 values: 14 channels × 5 bands)
                band_power_features = features_array[:70]
                #logging.debug(f" Shape of band_power_features: {band_power_features.shape}")
                feature_visualizer.update_data(band_power_features)  # Update band power visualizer
                feature_visualizer.update_plot(frame)


                # Extract Hjorth parameters (next 28 values: 14 channels × 2 parameters)
                hjorth_parameters = features_array[70:98]
                #logging.debug(f" Shape of hjorth_parameters: {hjorth_parameters.shape}")
                hjorth_visualizer.update_data(hjorth_parameters)  # Update Hjorth parameters visualizer
                #logging.debug(f" Values of hjorth_visualizer: {hjorth_parameters}")
                hjorth_visualizer.update_plot(frame)


                # Extract entropy features (next 14 values: 1 per channel)
                entropy_features = features_array[98:112]
                #logging.debug(f" Shape of entropy_features: {entropy_features.shape}")
                entropy_visualizer.update_data(entropy_features)  # Update entropy visualizer
                entropy_visualizer.update_plot(frame)


                # Extract fractal features (next 14 values: 1 per channel)
                fractal_features = features_array[112:126]
                #logging.debug(f" Shape of fractal_features: {fractal_features.shape}")
                fractal_visualizer.update_data(fractal_features)  # Update fractal visualizer
                fractal_visualizer.update_plot(frame)


                # Extract first-order derivatives (next 3584 values: 14 channels × 256 samples)
                first_order_derivatives = features_array[126:3622]
                logging.debug(f" Shape of first_order_derivatives: {first_order_derivatives.shape}")
                derivatives_visualizer.update_data(first_order_derivatives)  # Update first-order derivatives visualizer
                derivatives_visualizer.update_plot(frame)

                # Extract second-order derivatives (next 3584 values: 14 channels × 256 samples)
                second_order_derivatives = features_array[3622:7118]
                #logging.debug(f" Shape of second_order_derivatives: {second_order_derivatives.shape}")
                second_order_derivatives_visualizer.update_data(second_order_derivatives)  # Update second-order derivatives visualizer
                second_order_derivatives_visualizer.update_plot(frame)


                # Extract EEG filtered data (last 3584 values: 14 channels × 256 samples)
                eeg_filtered_features = features_array[7118:]
                #logging.debug(f" Shape of eeg_filtered_features: {eeg_filtered_features.shape}")
                eeg_filtered_visualizer.update_data(eeg_filtered_features)  # Update EEG filtered visualizer
                eeg_filtered_visualizer.update_plot(frame)
            except Exception as e:
                logging.error(f"[update_feature_visualization] Error updating feature visualization: {e}")

    # Wait for the streaming and preprocessing threads to notify that data is ready
    visualization_ready.wait()

    # Create animations for the 3D EEG visualizer
    visualizer.anim_3d = FuncAnimation(
        visualizer.fig_2d,
        update_3d_visualization,
        interval=100,
        blit=False,
        cache_frame_data=False
    )

    # Create animations for the 2D EEG visualizer
    visualizer.anim_2d = FuncAnimation(
        visualizer.fig_2d,
        update_2d_visualization,
        interval=100,
        blit=False,
        cache_frame_data=False
    )

    # Create animations for the EEG filtered visualizer
    eeg_filtered_visualizer.anim_feature = FuncAnimation(
        eeg_filtered_visualizer.fig,
        update_feature_visualization,
        interval=100,
        blit=False,
        cache_frame_data=False
    )
    # Create animations for the feature visualizer
    feature_visualizer.anim_feature = FuncAnimation(
        feature_visualizer.fig,
        update_feature_visualization,
        interval=100,
        blit=False,
        cache_frame_data=False
    )
    # Create animations for the Hjorth visualizer
    hjorth_visualizer.anim_feature = FuncAnimation(
        hjorth_visualizer.fig,
        update_feature_visualization,
        interval=100,
        blit=False,
        cache_frame_data=False
    )

    # Create animations for the derivatives visualizer
    derivatives_visualizer.anim_feature = FuncAnimation(
        derivatives_visualizer.fig,
        update_feature_visualization,
        interval=100,
        blit=False,
        cache_frame_data=False
    )

    # Create animations for the combined feature visualizer
    second_order_derivatives_visualizer.anim_feature = FuncAnimation(
        second_order_derivatives_visualizer.fig,
        update_feature_visualization,
        interval=100,
        blit=False,
        cache_frame_data=False
    )
    # Create animations for the fractal visualizer
    fractal_visualizer.anim_feature = FuncAnimation(
        fractal_visualizer.fig,
        update_feature_visualization,
        interval=100,
        blit=False,
        cache_frame_data=False
    )
    # Create animations for the entropy visualizer
    entropy_visualizer.anim_feature = FuncAnimation(
        entropy_visualizer.fig,
        update_feature_visualization,
        interval=100,
        blit=False,
        cache_frame_data=False
    )

    # Start the combined EEG visualizer animation
    #logging.debug("[run_visualizations_on_main_thread] 3D EEG visualizer animation started.")
    second_order_derivatives_visualizer.start_animation()
    # Start the feature visualizer animation
    #logging.debug("[run_visualizations_on_main_thread] FeatureVisualizer animation started.")
    #feature_visualizer.start_animation()

    # Start the derivatives visualizer animation
    #logging.debug("[run_visualizations_on_main_thread] DerivativesVisualizer animation started.")
    derivatives_visualizer.start_animation()

    # Add debug log before starting entropy visualizer animation
    #logging.debug("[run_visualizations_on_main_thread] Starting EntropyVisualizer animation.")
    entropy_visualizer.start_animation()

    # Start the fractal visualizer animation
    #logging.debug("[run_visualizations_on_main_thread] FractalVisualizer animation started.")
    fractal_visualizer.start_animation()

    #entropy_visualizer.start_animation()
    # Start the fractal visualizer animation
    hjorth_visualizer.start_animation()

    #logging.debug("[run_visualizations_on_main_thread] FractalVisualizer animation started.")

    # Show all visualizations in a single call
    plt.show()



