import soundfile as sf
from kokoro_onnx import Kokoro
import os
import onnxruntime as ort
import re
import time
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"

_KOKORO_INSTANCE = None

def clean_text_for_tts(text: str) -> str:
    text = re.sub(r'(\*\*|__|\*|_)', '', text)
    text = re.sub(r'#+\s?', '', text)
    text = re.sub(r'`', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[\[\]]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_kokoro():
    global _KOKORO_INSTANCE
    if _KOKORO_INSTANCE is None:
        model_path = "models/kokoro-v1.0.onnx"
        voices_path = "voices-v1.0.bin"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Kokoro model not found at {model_path}")
        if not os.path.exists(voices_path):
            raise FileNotFoundError(f"Voices file not found at {voices_path}")
            
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
        _KOKORO_INSTANCE = Kokoro.from_session(session, voices_path)
        
    return _KOKORO_INSTANCE

def generate_audio_response(text: str, output_filepath: str) -> str:
    kokoro = get_kokoro()
    
    clean_text = clean_text_for_tts(text)
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    all_samples = []
    sample_rate = 24000
    
    print(f"Generating audio for {len(sentences)} sentences...")
    start_time = time.time()
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        samples, sr = kokoro.create(
            sentence, 
            voice="am_adam", 
            speed=1.0, 
            lang="en-us"
        )
        all_samples.append(samples)
        sample_rate = sr
    
    if not all_samples:
        raise ValueError("No audio samples were generated.")
        
    final_samples = np.concatenate(all_samples)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Audio generation took {duration:.2f} seconds.")
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    sf.write(output_filepath, final_samples, sample_rate)
    return output_filepath
