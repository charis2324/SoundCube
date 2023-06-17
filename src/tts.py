import os
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play


def text_to_speech(text, lang="en"):
    # Convert text to speech using gTTS
    tts = gTTS(text=text, lang=lang)
    audio_file = "temp_audio.mp3"
    tts.save(audio_file)

    # Play the audio using pydub and pyaudio
    audio = AudioSegment.from_file(audio_file, format="mp3")
    play(audio)

    # Remove the temporary audio file
    os.remove(audio_file)


if __name__ == "__main__":
    # Example usage:
    text = "Hello, I am a text to speech function using gTTS and PyAudio."
    text_to_audio(text)
