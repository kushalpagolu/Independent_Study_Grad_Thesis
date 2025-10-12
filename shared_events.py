import threading

# Shared threading events
stop_saving_thread = threading.Event()
stop_input_listener = threading.Event()
stop_main_loop = threading.Event()
visualization_ready = threading.Event()  # Event to signal when visualization can start
