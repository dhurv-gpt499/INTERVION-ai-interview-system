import os
import torch
from faster_whisper import WhisperModel


def load_models():
    # "Loading Whisper" 
    whisper = WhisperModel("small", device="cuda", compute_type="int8_float16") 
    # "Load VAD" 
    silero_vad ,_ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad' , trust_repo=True)
    return whisper, silero_vad