import os
import argparse
import torch
import librosa
from glob import glob
from tqdm import tqdm
import numpy as np
import pyworld as pw
from concurrent.futures import ThreadPoolExecutor, as_completed
import parselmouth


# pyworld
def calculate_f0_raw(filename: str, 
                 fs: int = 16000, 
                 frame_period: int = 20, 
                 f0min: int = 50, 
                 f0max: int = 550, 
                ) -> torch.Tensor:
    wav, sr = librosa.load(filename, sr=16000, dtype='float64')
    _f0, t = pw.dio(wav, fs, frame_period=frame_period, f0_floor=f0min, f0_ceil=f0max)
    f0 = pw.stonemask(wav, _f0, t, fs)
    f0 = f0[:torch.load(filename.replace(".wav", ".pt")).shape[2]]
    print(f0.shape)
    # save_name = filename.replace(".wav", ".pitch_raw.npy")
    # np.save(save_name, f0)
    # return f0

# praat
def calculate_f0_praat(filename):
    audio, fs = librosa.load(filename, sr=16000)
    sound = parselmouth.Sound(audio, fs)
    time_step = 0.02
    pitch = sound.to_pitch(pitch_floor=50.0, pitch_ceiling=800.0, time_step=time_step)
    f0 = np.array([p[0] for p in pitch.selected_array])
    #pad f0 and fix ensure shape the same as ssl
    f0 = np.pad(f0, (0, 3), mode='constant', constant_values=0) 
    f0 = f0[:torch.load(filename.replace(".flac", ".pt")).shape[2]]
    save_name = filename.replace(".wav", ".pitch_raw.npy").replace(".flac", ".pitch_raw.npy")
    np.save(save_name, f0)
    return f0


if __name__ == "__main__":    
    filenames = []
    filenames = glob("/home/jovyan/voice-chung/vcvc/FreeVC/data/ML/mls_portuguese/train/audio/*/*/*.flac")
    num_threads = 64  # Adjust the number of threads as needed    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(calculate_f0_praat, filename) for filename in filenames]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()
