import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging
import queue
from shared_events import visualization_ready

# Note: The other feature visualizer imports are no longer needed here
# because they are only passed as arguments, not used directly in this file.

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def run_visualizations_on_main_thread(visualizer, visualization_queue, feature_visualizer, feature_queue, derivatives_visualizer, second_order_derivatives_visualizer, fractal_visualizer, entropy_visualizer, eeg_filtered_visualizer, hjorth_visualizer):
    """
    Run all visualizations on the main thread using a single, coordinated update loop.
    """

    def master_update(frame):
        """
        This single function is called repeatedly by the animation timer.
        It's responsible for updating ALL plots.
        """
        # --- Part 1: Update the 2D EEG and 3D Brain plots ---
        # Drain the queue of any available raw EEG data packets
        while not visualization_queue.empty():
            try:
                packet = visualization_queue.get_nowait()
                # We will later create a single update method in the visualizer class
                # For now, we assume adding data here is enough, and the update will be called below.
                visualizer.add_data(packet)
            except queue.Empty:
                # This can happen in a multithreaded environment, it's safe to ignore.
                break
        
        # We will consolidate the 2D and 3D updates into a single call in the next step.
        # For now, let's call them as they are.
        visualizer.update_all_plots(frame)


        # --- Part 2: Update all the Feature Plots ---
        # Check if there is a new set of features to plot
        if not feature_queue.empty():
            try:
                # Get one comprehensive feature array from the queue
                features_array = feature_queue.get_nowait()
                logging.debug(f"[Master Update] Features array dequeued with shape: {features_array.shape}")

                # Distribute the relevant slices of the array to each visualizer
                feature_visualizer.update_data(features_array[:70])
                hjorth_visualizer.update_data(features_array[70:98])
                entropy_visualizer.update_data(features_array[98:112])
                fractal_visualizer.update_data(features_array[112:126])
                derivatives_visualizer.update_data(features_array[126:3622])
                second_order_derivatives_visualizer.update_data(features_array[3622:7118])
                eeg_filtered_visualizer.update_data(features_array[7118:])

                # Now, call the individual plot update methods for each visualizer
                feature_visualizer.update_plot(frame)
                hjorth_visualizer.update_plot(frame)
                entropy_visualizer.update_plot(frame)
                fractal_visualizer.update_plot(frame)
                derivatives_visualizer.update_plot(frame)
                second_order_derivatives_visualizer.update_plot(frame)
                eeg_filtered_visualizer.update_plot(frame)

            except queue.Empty:
                # This is okay, it means another part of the loop was faster.
                pass
            except Exception as e:
                logging.error(f"[Master Update] Error updating feature visualizations: {e}")


    # Wait for the producer threads to signal that they are ready to send data
    visualization_ready.wait()

    # --- Create ONLY ONE animation that calls our master_update function ---
    # This single animation will drive updates for ALL plots.
    # We must hold a reference to it to prevent it from being garbage collected.
    anim = FuncAnimation(
        visualizer.fig_2d,      # Attach the animation to one of the main figures
        master_update,          # The function to call for every frame
        interval=200,           # Update every 200ms (slower is more stable for many plots)
        blit=False,
        cache_frame_data=False
    )

    # Show all plots. Matplotlib will manage drawing all open figures.
    # This is a blocking call that starts the GUI event loop.
    plt.show()

    # This line is just to ensure the 'anim' variable is used, preventing some linters from complaining.
    # In reality, creating the FuncAnimation is what keeps it alive.
    _ = anim
