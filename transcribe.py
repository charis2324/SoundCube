from faster_whisper import WhisperModel
import numpy as np
import time

model_size = "base"
model = WhisperModel(model_size, device="cuda", compute_type="float16")
print(f"Whisper model: {model_size} loaded.")


def transcribe(audio_data, lang=None):
    if isinstance(audio_data, np.ndarray) and audio_data.dtype == np.int16:
        print("Converting int16 to float32 to fit the whisper model.")
        audio_data = audio_data.astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio_data, language=lang)
    print("Transcribing.")
    start_time = time.time()
    transcribe_text = "".join(list(segment.text for segment in segments))
    print(f"Transcribed in {((time.time() - start_time)*1000) :.2f} ms.")
    return transcribe_text
