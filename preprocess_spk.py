import os, sys
from speaker_encoder.voice_encoder import SpeakerEncoder
from speaker_encoder.audio import preprocess_wav
from pathlib import Path
import numpy as np
from os.path import join, basename, split
from tqdm import tqdm
from multiprocessing import cpu_count
from functools import partial
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

def _compute_spkEmbed(wav_path):
    fpath = Path(wav_path)
    wav = preprocess_wav(fpath)
    embed = encoder.embed_utterance(wav)
    fname_save = wav_path.replace(".wav", ".freevc.npy").replace(".flac", ".freevc.npy")
    np.save(fname_save, embed, allow_pickle=False)

if __name__ == "__main__":
    encoder = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')
    # filenames = glob("/home/jovyan/voice-chung/stbase/vits/data/librispeech/*/*/*/*.flac")
    filenames = glob("/home/jovyan/voice-chung/vcvc/FreeVC/data/ML/mls_portuguese/train/audio/*/*/*.flac")
    num_threads = 64  
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(_compute_spkEmbed, filename) for filename in filenames]

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()