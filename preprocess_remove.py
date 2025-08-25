import os
import argparse
import torch
import librosa
from glob import glob
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import parselmouth


# if bug when training, some specpt fail => bug => remove all rerun, or try extract mel first
def removee(filename):
    to_remove = filename.replace(".flac", ".spec.pt")
    if os.path.exists(to_remove):
        os.remove(to_remove)


if __name__ == "__main__":    
    filenames = []
    filenames = glob("/home/jovyan/voice-chung/vcvc/FreeVC/data/ML/mls_portuguese/train/audio/*/*/*.flac")
    num_threads = 64  # Adjust the number of threads as needed    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(removee, filename) for filename in filenames]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()
