import os
import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm
import parselmouth
import numpy as np
import utils
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import re
from models_f0 import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import logging
import einops
from scipy.io.wavfile import write
from tqdm import tqdm
import torchaudio
from transformers import Wav2Vec2Processor, HubertForCTC
import os
from glob import glob
from jiwer import wer, cer

def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    
hps = utils.get_hparams_from_file("configs/freevc_f0.json")

print("Loading model...")
net_g = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()
print("Loading checkpoint...")
_ = utils.load_checkpoint("/home/jovyan/voice-chung/vcvc/OOVC/logs/oovc_w_f0/G_1470000.pth", net_g), None, True

print("Loading WavLM for content...")
cmodel = utils.get_cmodel(0)

if hps.model.use_spk:
    print("Loading speaker encoder...")
    smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

def calculate_f0_praat(filename):
    audio, fs = librosa.load(filename, sr=16000)
    sound = parselmouth.Sound(audio, fs)
    time_step = 0.02
    pitch = sound.to_pitch(pitch_floor=50.0, pitch_ceiling=550.0, time_step=time_step)
    f0 = np.array([p[0] for p in pitch.selected_array])
    f0 = np.pad(f0, (0, 3), mode='constant', constant_values=0) #here we add f0 to ensure shape the same as content
    return f0


encoder = VoiceEncoder(device=torch.device("cuda"), weights_fpath='./pretrained.pt')

    
seed_everything(0)
net_g.eval()
cmodel.eval()
smodel.eval()


def conversion(source, target):
    F0_src = calculate_f0_praat(src)
    F0_tgt = calculate_f0_praat(tgt)
    F0_src = torch.from_numpy(F0_src)
    F0_tgt = torch.from_numpy(F0_tgt)        
    voiced_F0_src = F0_src[F0_src > 1]
    voiced_F0_tgt = F0_tgt[F0_tgt > 1]
    log_f0_src = torch.log(F0_src + 1e-5)
    voiced_log_f0_src = torch.log(voiced_F0_src + 1e-5)
    voiced_log_f0_tgt = torch.log(voiced_F0_tgt + 1e-5)
    median_log_f0_src = torch.median(voiced_log_f0_src)
    median_log_f0_tgt = torch.median(voiced_log_f0_tgt)
    median_f0_src = torch.exp(median_log_f0_src)
    # F0 source to target speaker f0 by median
    shifted_log_f0_tgt = log_f0_src.clone()
    # shift semitone
    shifted_log_f0_tgt[F0_src > 1] = log_f0_src[F0_src > 1] - median_log_f0_src + median_log_f0_tgt
    shifted_f0_tgt = torch.exp(shifted_log_f0_tgt).cuda().unsqueeze(0)
    
    wav_tgt, _ = librosa.load(target, sr=hps.data.sampling_rate)
    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
    g_tgt = smodel.embed_utterance(wav_tgt)
    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
    wav_src, _ = librosa.load(source, sr=hps.data.sampling_rate)
    wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
    c = utils.get_content(cmodel, wav_src)
    audio = net_g.infer(c, g=g_tgt, f0=shifted_f0_tgt)
    audio = audio[0][0].data
    return audio.cpu().float().numpy()

if __name__ == '__main__':  
    src = "./sample/8230-279154-0028.flac"
    tgt = "./sample/4970-29095-0008.flac"
    out = "./sample/test.wav"
    audio = conversion(src, tgt)
    write(out, hps.data.sampling_rate, audio)