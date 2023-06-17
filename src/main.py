import time
import os
import numpy as np
import pyaudio
import tensorflow as tf
import speech_recognition as sr
from datetime import datetime
import wave
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ThreeCharacterClassicInference import ThreeCharacterClassicInference
from tts import text_to_speech

# Set seeds for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Constants
FRAMES_PER_BUFFER = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TAILING_DURATION = 1.5  # Tailing audio duration in seconds
KEYWORD = "你好"

# Global variables
stop_plotting_thread = False


# Load the model
interpreter = tf.lite.Interpreter(model_path="hey_ego_44100_obama_5.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
three_char_classic_model = ThreeCharacterClassicInference(
    model_path="3character.tflite", dictionary_path="3character_dict.pickle"
)


def get_spectrogram(waveform):
    """Convert the audio waveform to a spectrogram."""
    input_len = 66150
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([input_len] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=512, frame_step=256)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def preprocess_audiobuffer(waveform):
    """Preprocess the audio buffer for the model."""
    waveform = waveform / 32768
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    spectogram = get_spectrogram(waveform)
    spectogram = tf.expand_dims(spectogram, 0)
    return spectogram


def predict_mic(audio):
    """Predict the command from the audio."""
    start = time.time()
    spec = preprocess_audiobuffer(audio)
    interpreter.set_tensor(input_details[0]["index"], spec)
    interpreter.invoke()
    prediction = tf.nn.softmax(interpreter.get_tensor(output_details[0]["index"]))
    label_pred = np.argmax(prediction, axis=1)
    time_taken = time.time() - start
    print(prediction)
    print(label_pred)
    print(f"Predicted in: {time_taken}")
    return label_pred[0]


def save_audio_to_wav(audio_buffer, output_folder=None, rate=44100):
    """Save the audio buffer to a mono channel WAV file with a unique name."""
    output_folder = output_folder or os.getcwd()
    os.makedirs(output_folder, exist_ok=True)

    # Generate a unique name for the WAV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_{timestamp}.wav"
    output_file = os.path.join(output_folder, file_name)

    # Open a WAV file for writing
    with wave.open(output_file, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono channel
        wav_file.setsampwidth(
            pyaudio.get_sample_size(pyaudio.paInt16)
        )  # 16-bit samples
        wav_file.setframerate(rate)  # Set the frame rate
        wav_file.writeframes(audio_buffer.tobytes())  # Write the audio buffer data

    return output_file


# def plot_spectrogram(audio_buffer, spectrogram_func, stop_event):
#     while not stop_event.is_set():
#         # Set up the initial plot
#         fig, ax = plt.subplots()
#         spec = spectrogram_func(audio_buffer)
#         im = ax.imshow(
#             spec,
#             aspect="auto",
#             origin="lower",
#             cmap="viridis",
#             vmin=0.0,
#             vmax=1.0,
#         )
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Frequency")
#         plt.colorbar(im, ax=ax)
#         ax.set_title("Spectrogram")

#         # Add a text element to display the update frequency
#         freq_text = ax.text(
#             0.01, 0.95, "", transform=ax.transAxes, fontsize=10, color="white"
#         )

#         # Update function for the plot
#         def update(frame):
#             nonlocal audio_buffer
#             start_time = time.time()
#             spec = spectrogram_func(audio_buffer)
#             im.set_data(spec)
#             im.set_clim(vmin=0.0, vmax=1.0)
#             # Calculate and display the update frequency
#             end_time = time.time()
#             update_freq = 1 / (end_time - start_time)
#             freq_text.set_text(f"{update_freq:.2f} fps")

#             return [im, freq_text]

#         # Create the animation
#         ani = FuncAnimation(fig, update, blit=True, interval=RATE / FRAMES_PER_BUFFER)

#         # Show the plot
#         plt.show()
#         if stop_event.is_set():
#             plt.close()
#             break


def plot_spectrogram(audio_buffer, spectrogram_func, stop_event):
    # Initialize the stop_event flag
    stopped = [False]

    def handle_close(evt):
        stopped[0] = True

    while not stop_event.is_set():
        # Set up the initial plot
        fig, ax = plt.subplots()
        spec = spectrogram_func(audio_buffer)
        im = ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax)
        ax.set_title("Spectrogram")

        # Add a text element to display the update frequency
        freq_text = ax.text(
            0.01, 0.95, "", transform=ax.transAxes, fontsize=10, color="white"
        )

        # Update function for the plot
        def update(frame):
            nonlocal audio_buffer
            start_time = time.time()
            spec = spectrogram_func(audio_buffer)
            im.set_data(spec)
            im.set_clim(vmin=0.0, vmax=1.0)
            # Calculate and display the update frequency
            end_time = time.time()
            update_freq = 1 / (end_time - start_time)
            freq_text.set_text(f"{update_freq:.2f} fps")

            if stopped[0]:
                return []
            return [im, freq_text]

        # Create the animation
        ani = FuncAnimation(fig, update, blit=True, interval=RATE / FRAMES_PER_BUFFER)

        # Connect the event handler
        fig.canvas.mpl_connect("close_event", handle_close)

        # Show the plot
        plt.show(block=False)

        while not stop_event.is_set() and not stopped[0]:
            fig.canvas.flush_events()
            time.sleep(0.1)

        if stop_event.is_set() or stopped[0]:
            plt.close()
            break


def record_and_detect_keyword():
    """Continuously record audio and predict the command."""
    global stop_plotting_thread
    audio_buffer = np.zeros(int(TAILING_DURATION * RATE), dtype=np.int16)
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )
    stop_event = threading.Event()
    try:
        plot_thread = threading.Thread(
            target=plot_spectrogram,
            args=(
                audio_buffer,
                lambda buf: preprocess_audiobuffer(buf).numpy().squeeze(),
                stop_event,
            ),
            daemon=True,
        )
        plot_thread.start()
        is_awake = False
        recognizer = sr.Recognizer()
        while True and not is_awake:
            data = stream.read(FRAMES_PER_BUFFER)
            new_audio = np.frombuffer(data, dtype=np.int16)

            # Update the audio buffer
            audio_buffer[:-FRAMES_PER_BUFFER] = audio_buffer[FRAMES_PER_BUFFER:]
            audio_buffer[-FRAMES_PER_BUFFER:] = new_audio

            # Save the audio buffer to a WAV file
            # output_file = save_audio_to_wav(
            #     audio_buffer, output_folder="recorded_audio"
            # )

            # Predict using the tailing audio data
            if not is_awake:
                result = predict_mic(audio_buffer)

                if result == 0:
                    print(f"Obama model detected {KEYWORD}")
                    # is_awake = True
                    audio_data = sr.AudioData(
                        audio_buffer.tobytes(), sample_rate=RATE, sample_width=2
                    )
                    try:
                        text = recognizer.recognize_google(audio_data, language="zh-CN")
                        print("You said: ", text)
                        if KEYWORD in text:
                            is_awake = True
                    except sr.UnknownValueError:
                        print("Google Speech Recognition could not understand audio")
                    except sr.RequestError as e:
                        print(
                            "Could not request results from Google Speech Recognition service; {0}".format(
                                e
                            )
                        )

            if is_awake:
                p.terminate()
                stop_event.set()

    except Exception as e:
        print(e)
        p.terminate()
        stop_event.set()


def three_char_classic_reply():
    previous_sr_text = ""
    audio_buffer = np.zeros(int(5 * RATE), dtype=np.int16)
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )
    recognizer = sr.Recognizer()
    buffers_per_second = int(RATE / FRAMES_PER_BUFFER) * 2
    idel_start_time = time.time()
    while True:
        audio_data = np.empty((buffers_per_second, FRAMES_PER_BUFFER), dtype=np.int16)
        for i in range(buffers_per_second):
            audio_data[i] = np.frombuffer(
                stream.read(FRAMES_PER_BUFFER), dtype=np.int16
            )
        audio_data = audio_data.flatten()
        audio_buffer[: -audio_data.shape[0]] = audio_buffer[audio_data.shape[0] :]
        audio_buffer[-audio_data.shape[0] :] = audio_data
        audio_data = sr.AudioData(
            audio_buffer.tobytes(), sample_rate=RATE, sample_width=2
        )
        try:
            text = recognizer.recognize_google(audio_data, language="zh-CN")
            print("You said: ", text)
            if len(text) >= 3 and text != previous_sr_text:
                previous_sr_text = text
                reply = three_char_classic_model.predict_next_3(text[0:3])
                text_to_speech(reply, "zh")
                print(f"Model reply: {reply}")
            idel_start_time = time.time()
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(
                "Could not request results from Google Speech Recognition service; {0}".format(
                    e
                )
            )
        if (time.time() - idel_start_time) > 10:
            text_to_speech("晚安寶貝兒", "zh")
            return


if __name__ == "__main__":
    text_to_speech("開始了", "zh")
    while True:
        print("start...")
        record_and_detect_keyword()
        print("awake...")
        print("Your three char classic?..")
        text_to_speech("你好呀。 請讀出你的三字經典三連音。", "zh")
        three_char_classic_reply()
