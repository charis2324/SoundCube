import pyaudio
import datetime
import wave
import queue
import threading
import numpy as np
import time

import transcribe

# Settings
SAVE_ROLLING_WINDOW_TO_FILE = False

# Constants. Don't change.
RATE = 16000  # To fit whisper
BUFFER_SIZE = 1024
FORMAT = pyaudio.paInt16
ROLLING_WINDOW_SECONDS = 3
ROLLING_WINDOW_SAMPLES = RATE * ROLLING_WINDOW_SECONDS


p = pyaudio.PyAudio()
audio_queue = queue.Queue()
rolling_window = np.empty((0,), dtype=np.int16)


def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(np.frombuffer(in_data, dtype=np.int16))
    return (in_data, pyaudio.paContinue)


def process_audio():
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            global rolling_window
            rolling_window = np.concatenate((rolling_window, audio_data))
            if len(rolling_window) > ROLLING_WINDOW_SAMPLES:
                rolling_window = rolling_window[-ROLLING_WINDOW_SAMPLES:]
                print(rolling_window)
                # Start data processing here.
                transcribe_text = transcribe.transcribe(rolling_window)
                print(f"Transcript: {transcribe_text}")
                if SAVE_ROLLING_WINDOW_TO_FILE:
                    write_wave_file(
                        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".wav",
                        rolling_window,
                        RATE,
                    )


def write_wave_file(filename, audio_data, sample_rate):
    with wave.open(filename, "wb") as wave_file:
        # Set the wave file parameters
        wave_file.setnchannels(1)  # Mono audio
        wave_file.setsampwidth(2)  # 16-bit audio
        wave_file.setframerate(sample_rate)

        audio_data_bytes = audio_data.tobytes()
        wave_file.writeframes(audio_data_bytes)


# Start audio processing thread.
audio_processing_thread = threading.Thread(target=process_audio, daemon=True)
audio_processing_thread.start()

stream = p.open(
    format=FORMAT,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=BUFFER_SIZE,
    stream_callback=audio_callback,
)
stream.start_stream()

# Keep the main thread running.
while True:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        break

stream.stop_stream()
stream.close()
p.terminate()
