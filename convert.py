import os
import argparse
import torch
import librosa
import numpy as np
import parselmouth
from scipy.io.wavfile import write
from resemblyzer import VoiceEncoder
from speaker_encoder.voice_encoder import SpeakerEncoder
from models_f0 import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
import utils

def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def calculate_f0_praat(filename):
    """Calculate F0 contour using Praat via parselmouth."""
    audio, fs = librosa.load(filename, sr=16000)
    sound = parselmouth.Sound(audio, fs)
    time_step = 0.02
    pitch = sound.to_pitch(pitch_floor=50.0, pitch_ceiling=550.0, time_step=time_step)
    f0 = pitch.selected_array['frequency']
    f0 = np.pad(f0, (0, 3), mode='constant', constant_values=0)
    return f0


def conversion(source, target, net_g, cmodel, smodel, hps):
    """Main conversion pipeline."""
    # Calculate F0
    F0_src = calculate_f0_praat(source)
    F0_tgt = calculate_f0_praat(target)
    F0_src = torch.from_numpy(F0_src)
    F0_tgt = torch.from_numpy(F0_tgt)

    # Median-based F0 shift
    voiced_F0_src = F0_src[F0_src > 1]
    voiced_F0_tgt = F0_tgt[F0_tgt > 1]
    log_f0_src = torch.log(F0_src + 1e-5)
    voiced_log_f0_src = torch.log(voiced_F0_src + 1e-5)
    voiced_log_f0_tgt = torch.log(voiced_F0_tgt + 1e-5)
    median_log_f0_src = torch.median(voiced_log_f0_src)
    median_log_f0_tgt = torch.median(voiced_log_f0_tgt)
    shifted_log_f0_tgt = log_f0_src.clone()
    shifted_log_f0_tgt[F0_src > 1] = log_f0_src[F0_src > 1] - median_log_f0_src + median_log_f0_tgt
    shifted_f0_tgt = torch.exp(shifted_log_f0_tgt).cuda().unsqueeze(0)

    # Load target speaker embedding
    wav_tgt, _ = librosa.load(target, sr=hps.data.sampling_rate)
    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
    g_tgt = smodel.embed_utterance(wav_tgt)
    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()

    # Get source content representation
    wav_src, _ = librosa.load(source, sr=hps.data.sampling_rate)
    wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
    c = utils.get_content(cmodel, wav_src)

    # Generate converted audio
    with torch.no_grad():
        audio = net_g.infer(c, g=g_tgt, f0=shifted_f0_tgt)[0][0].data
    return audio.cpu().float().numpy()


def main():
    parser = argparse.ArgumentParser(description="Voice conversion using OOVC with F0 shifting")
    parser.add_argument("--source", "-s", required=True, help="Path to source audio file")
    parser.add_argument("--target", "-t", required=True, help="Path to target audio file")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint (G_xxx.pth)")
    parser.add_argument("--output", "-o", default="./converted.wav", help="Path to save converted audio")
    args = parser.parse_args()

    # Load configs and models
    hps = utils.get_hparams_from_file("configs/freevc_f0.json")

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()

    print(f"Loading checkpoint: {args.checkpoint}")
    _ = utils.load_checkpoint(args.checkpoint, net_g)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)

    print("Loading speaker encoder...")
    smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    # Seed
    seed_everything(0)

    # Convert
    print("Running conversion...")
    audio = conversion(args.source, args.target, net_g, cmodel, smodel, hps)
    write(args.output, hps.data.sampling_rate, audio)
    print(f"Conversion done. Saved to: {args.output}")


if __name__ == "__main__":
    main()
