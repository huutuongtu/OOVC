import os
import argparse
import torch
import librosa
from glob import glob
from tqdm import tqdm

import utils
from wavlm import WavLM, WavLMConfig
from concurrent.futures import ThreadPoolExecutor, as_completed



def process(filename):
    wav, _ = librosa.load(filename, sr=16000)
    wav = torch.from_numpy(wav).unsqueeze(0).cuda()
    c = utils.get_content(cmodel, wav)
    save_name = filename.replace(".flac", ".pt").replace(".wav", ".pt")
    torch.save(c.detach().cpu(), save_name)


if __name__ == "__main__":    

    print("Loading WavLM for content...")
    checkpoint = torch.load('wavlm/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg).cuda()
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()

    filenames = glob("/home/jovyan/voice-chung/vcvc/FreeVC/data/ML/mls_portuguese/train/audio/*/*/*.flac")
    
    num_threads = 32  # Adjust the number of threads as needed    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process, filename) for filename in filenames]

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # To catch exceptions raised in threads
    