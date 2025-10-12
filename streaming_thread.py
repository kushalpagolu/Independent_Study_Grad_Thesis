
import time
import threading
import logging
from shared_events import stop_saving_thread, stop_input_listener, stop_main_loop, visualization_ready

from signal_handler import signal_handler

def streaming_thread(emotiv, data_queue, visualization_queue, visualizer, data_store, data_lock):
    """
    Thread for streaming data from the Emotiv device.
    """
    
    valid_packet_count = 0
    empty_packet_count = 0
    loop_count = 0  # Initialize loop counter

    eeg_channels = emotiv.channel_names  # 14 EEG channels

    while not stop_main_loop.is_set():
        loop_count += 1
        try:
            packet = emotiv.read_emotiv_data()
            if packet and len(packet) > 0:
                valid_packet_count += 1
                data_queue.put(packet)
                visualization_queue.put(packet)
                visualization_ready.set()

                # Save raw EEG + gyro into data_store
                if data_store is not None:
                    try:
                        eeg_values = [packet.get(ch, 0.0) for ch in eeg_channels]
                        gyro_values = [packet.get('gyro_x', 0.0), packet.get('gyro_y', 0.0)]
                        with data_lock:
                            data_store.append(eeg_values + gyro_values)
                        logging.debug(f"[Streaming Thread] Appended data to data_store")
                    except Exception as e:
                        logging.warning(f"[Streaming Thread] Could not append to data_store: {e}")

                if 'gyro_x' in packet and 'gyro_y' in packet:
                    visualizer.update_gyro_data(packet['gyro_x'], packet['gyro_y'])
            else:
                empty_packet_count += 1
                if empty_packet_count > 250:
                    logging.error(f"[Streaming Thread] Too many empty packets. Stopping streaming and reconnecting... {empty_packet_count}")
                    emotiv.disconnect()
                    time.sleep(3)
                    if not emotiv.connect():
                        logging.error("[Streaming Thread] Failed to reconnect to Emotiv device.")
                    empty_packet_count = 0
                continue
        except KeyboardInterrupt:
            logging.info("[Streaming Thread] KeyboardInterrupt received. Stopping streaming thread.")
            stop_main_loop.set()
            break
        except Exception as e:
            logging.error(f"[Streaming Thread] Error while reading data: {e}")
        time.sleep(0.01)
