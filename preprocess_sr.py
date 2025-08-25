import os
import argparse
import torch
import librosa
from glob import glob
from tqdm import tqdm
import utils
from wavlm import WavLM, WavLMConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import soundfile as sf

def process(filename):
    wav, _ = librosa.load(filename, sr=16000)
    sf.write(filename, wav, 16000)


if __name__ == "__main__":    
    filenames = []
    filenames = glob("/home/jovyan/voice-chung/vcvc/FreeVC/data/ML/mls_portuguese/train/audio/*/*/*.flac")
    num_threads = 64 
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process, filename) for filename in filenames]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # To catch exceptions raised in threads  
