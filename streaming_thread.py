import time
import threading
import logging
from shared_events import stop_main_loop, visualization_ready

logger = logging.getLogger(__name__)

def streaming_thread(emotiv, data_queue, visualization_queue, data_store, data_lock):
    """
    Thread for streaming data from the Emotiv device.
    Its ONLY job is to read data and put it into the queues.
    """
    valid_packet_count = 0
    empty_packet_count = 0

    eeg_channels = emotiv.channel_names

    while not stop_main_loop.is_set():
        try:
            packet = emotiv.read_emotiv_data()
            if packet and len(packet) > 0:
                valid_packet_count += 1
                empty_packet_count = 0  # Reset counter on valid packet

                # Put the raw packet on the queues for other threads to process
                data_queue.put(packet)
                visualization_queue.put(packet)
                visualization_ready.set()

                # Save raw EEG + gyro into data_store
                if data_store is not None:
                    try:
                        with data_lock:
                            eeg_values = [packet.get(ch, 0.0) for ch in eeg_channels]
                            gyro_values = [packet.get('gyro_x', 0.0), packet.get('gyro_y', 0.0)]
                            data_store.append(eeg_values + gyro_values)
                    except Exception as e:
                        logger.warning(f"[Streaming Thread] Could not append to data_store: {e}")

            else:
                empty_packet_count += 1
                if empty_packet_count > 500:
                    logger.error(f"[Streaming Thread] Too many empty packets ({empty_packet_count}). Attempting to reconnect...")
                    emotiv.disconnect()
                    time.sleep(3)
                    if not emotiv.connect():
                        logger.error("[Streaming Thread] Failed to reconnect to Emotiv device.")
                    empty_packet_count = 0 # Reset after attempting reconnect

        except KeyboardInterrupt:
            logger.info("[Streaming Thread] KeyboardInterrupt received. Stopping.")
            stop_main_loop.set()
            break
        except Exception as e:
            logger.error(f"[Streaming Thread] Error while reading data: {e}")
        
        time.sleep(0.001) # Sleep for a very short duration to prevent busy-waiting
