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
from tqdm import tqdm
import wave
# importing speech recognition package from google api
import speech_recognition as sr
from gtts import gTTS  # google text to speech
import os  # to save/open files
from IPython.display import  Audio
from base64 import b64decode
import numpy as np
import wave
from scipy.io.wavfile import read as wav_read
import matplotlib.pyplot as plt
from itertools import chain
from gpiozero import RGBLED
from gpiozero import Button

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
press = Button(24, pull_up=False)
led = RGBLED(red=27, green=22, blue=23)

stop_plotting_thread = False


# Load the model
# interpreter = tf.lite.Interpreter(model_path="hey_ego_44100_obama_5_1.tflite")
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
three_char_classic_model = ThreeCharacterClassicInference(
    model_path="3character.tflite", dictionary_path="3character_dict.pickle"
)
vocabs = three_char_classic_model.get_dictionary().keys()


def check_triplets(text, vocabs):
    iter = list(chain(text, text))
    for i in range(len(text)):
        in_vocabs = []
        for j in range( i,  i + 3):
            if iter[j] in vocabs:
                in_vocabs.append(j)
        if len(in_vocabs) == 3:
            return "".join([iter[index] for index in in_vocabs])
    return ""

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

def play_audio(wave_input_path):
    p = pyaudio.PyAudio()
    wf = wave.open(wave_input_path, 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(FRAMES_PER_BUFFER)
    while len(data) > 0:
        if press.is_pressed:
            break
        stream.write(data)
        data = wf.readframes(FRAMES_PER_BUFFER)

    stream.stop_stream()
    stream.close()
    p.terminate()


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

                if True:
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
                                    print("Could not request results from Google Speech Recognition service; {0}".format(e))

            if is_awake:
                p.terminate()
                stop_event.set()

    except Exception as e:
        print(e)
        p.terminate()
        stop_event.set()

def get_text(human_sound_file):
    
    global language

    # initialize the recognizer
    asr = sr.Recognizer()

    # open the file
    try:
        with sr.AudioFile(human_sound_file) as source:
            # listen for the data (load audio to memory)
            audio_data = asr.record(source)
            # recognize (convert from speech to text)
            text = asr.recognize_google(
                audio_data, language='zh-hant', show_all=False)
            return text
    except:
        text = "Sorry, I can't understand. Please say again!"
        print(text)
        return text


def write_wav(f, sr, x, normalized=False):
    # 開檔
    f = wave.open(f, "wb")
    # 配置聲道数、量化位数和取樣频率
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sr)
    # 轉換為二進制，再寫入檔案
    wave_data = x.astype(np.short)
    f.writeframes(wave_data.tostring())
    f.close()

def robot_speaks(output): 
    global language
    global rnum
    # num to rename every audio file  
    # with different name to remove ambiguity 
    rnum += 1
    print("PerSon : ", output) 
  
    toSpeak = gTTS(text = output, lang = language, slow = False) 
    
    # saving the audio file given by google text to speech 
    robot_sound_file = "robot-"+str(rnum)+".mp3"
    toSpeak.save(robot_sound_file)
    
    return robot_sound_file

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio, language='zh-hant')
            print(said)
            return said

        except Exception as e:
            print("Exception: " + str(e))

def three_char_classic_reply():
    previous_sr_text = ""
    audio_buffer = np.zeros(int(5 * RATE), dtype=np.int16)
    p = None
    stream = None
    p = pyaudio.PyAudio()
    # stream = p.open(
    #     format=FORMAT,
    #     channels=CHANNELS,
    #     rate=RATE,
    #     input=True,
    #     frames_per_buffer=FRAMES_PER_BUFFER,
    # )
    recognizer = sr.Recognizer()
    buffers_per_second = int(RATE / FRAMES_PER_BUFFER) * 2
    idel_start_time = time.time()
    wake="你好"
    is_wake=False
    isReleased = True
    while True:
        # p = pyaudio.PyAudio()
        if isReleased == True:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=FRAMES_PER_BUFFER,
            )
            isReleased = False
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
        
        # p.terminate()
        try:
            #led.color = (0, 1, 0)
            text = recognizer.recognize_google(audio_data, language="zh-CN")
            print("You said: ", text)
            if wake in text:
                is_wake=True
                print("I'm wake")
            if is_wake==True:
                #led.color = (1,0, 0)
                string = recognizer.recognize_google(audio_data, language="zh-CN")
                filtered_string = check_triplets(string, vocabs)
                print(f"filtered_string: {filtered_string}")
                if  '周杰倫' in string:
                    led.color = (0, 0, 1)
                    print ("周杰倫")
                    stream.stop_stream()
                    stream.close()
                    isReleased = True
                    play_audio("music.wav")
                    idel_start_time = time.time()
                elif '蔡依林'in string:
                    #led.color = (0, 0, 1)
                    print ("蔡依林")
                    stream.stop_stream()
                    stream.close()
                    isReleased = True
                    play_audio("music_2.wav")
                    idel_start_time = time.time()
                elif len(filtered_string) == 3 and filtered_string != previous_sr_text:
                    #led.color = (0, 0, 1)
                    previous_sr_text = filtered_string
                    reply = three_char_classic_model.predict_next_3(filtered_string[0:3])
                    stream.stop_stream()
                    stream.close()
                    isReleased = True
                    text_to_speech(reply, "zh")
                    print(f"Model reply: {reply}")
                    
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(
                "Could not request results from Google Speech Recognition service; {0}".format(
                    e
                )
            )
        if (time.time() - idel_start_time) > 30:
            stream.stop_stream()
            stream.close()
            isReleased = True
            #text_to_speech("再見", "zh")
            
            #human_sound_file = get_audio()
            #text = get_text(human_sound_file)
            print(text)
            return


if __name__ == "__main__":
    #text_to_speech("開始了", "zh")
    #play_audio("getting started....wav")
    print("start")
    while True:
        print("start...")
        #record_and_detect_keyword()
        print("awake...")
        print("say something")
        #play_audio("say something....wav")
        #text_to_speech("説話", "zh")
        three_char_classic_reply()
