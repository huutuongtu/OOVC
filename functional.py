import os
import torch
import librosa
import json
from tqdm import tqdm
from scipy.io import wavfile
import utils
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from transformers import Wav2Vec2Processor, HubertForCTC
import os
from glob import glob
from jiwer import wer, cer
import re
from tqdm import tqdm
import numpy as np
import random
import hifigan
import soundfile as sf


def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

seed_everything(0)


def get_vocoder(rank):
    with open("hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load("hifigan/generator_v1")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.cuda(rank)
    return vocoder