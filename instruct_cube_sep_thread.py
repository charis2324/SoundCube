import speech_recognition as sr
import queue
import pyaudio
import wave
import time

recognized_texts = queue.Queue()

def playAudio(filename, chunk = 1024):
    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True)
    data = wf.readframes(chunk)
    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)
    stream.close()
    p.terminate()

def callback(recognizer, audio):
    global recognized_text
    print("New chunk received. Recogning...")
    try:
        recognized_text = recognizer.recognize_google(audio)
        print("Google Speech Recognition thinks you said " + recognized_text)
        recognized_texts.put(recognized_text)
    except:
        print("Recognition failed.")

for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

r = sr.Recognizer()
m = sr.Microphone()

with m as source:
    r.adjust_for_ambient_noise(source)
    print("calibrated to background noise.")
stop_listening = r.listen_in_background(source, callback)

while True:
    if recognized_texts.empty():
        print('No text recignized.')
    else:
        text = recognized_texts.get()
        if '周杰倫' in text:
            print("周杰倫 in text.")
            print("Playing his song.")
            playAudio(r'\path to his song')
        elif '蔡依林' in text:
            print("蔡依林 in text.")
            print("Playing her song.")
            playAudio(r'\path to her song')
    time.sleep(0.5)