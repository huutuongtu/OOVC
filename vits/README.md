# [EMNLP 2025 Findings] O_O-VC: Synthetic Data-Driven One-to-One Alignment for Any-to-Any Voice Conversion  


## 🛠 Setup & Dependencies  

### Clone the repository  
```bash
git clone https://github.com/huutuongtu/OOVC
cd OOVC
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```


## 🎧 Model Downloads

### 1️⃣ Download WavLM Model

Download **[WavLM-Large](https://github.com/microsoft/unilm/tree/master/wavlm)** and place it under the `wavlm/` directory.

### 2️⃣ Download Pretrained Generator Checkpoint

Download the **[pretrained checkpoint](https://drive.google.com/drive/folders/120CWHi3L2C-cw4AYcxwCSl6b-Th9PmJA?usp=sharing)**
and place it under your logs folder (e.g., `logs/oovc_w_f0/`).


## 🚀 Inference


```bash
python convert.py \
  --source sample/8230-279154-0028.flac \
  --target sample/4970-29095-0008.flac \
  --checkpoint logs/oovc_w_f0/G_1470000.pth \
  --output sample/test_converted.wav
```

## 📂 Repository Structure

```
OOVC/
├── convert.py               # Main inference script
├── models_f0.py             # Generator model definition
├── mel_processing.py        # Mel spectrogram utilities
├── utils.py                 # Helper functions
├── wavlm/                   # WavLM model files
├── speaker_encoder/         # Speaker encoder files
├── configs/
│   └── freevc_f0.json       # Configuration file
├── sample/
│   ├── source_audio.flac
│   ├── target_audio.flac
│   └── test_converted.wav
└── logs/
    └── oovc_w_f0/
        └── G_1470000.pth
```

---

## 📘 Citation

If you use this code, please cite our paper:

```
@inproceedings{tu-2025_oovc,
  author    = {Huu Tuong Tu and Huan Vu and Cuong Tien Nguyen and Dien Hy Ngo and Nguyen Thi Thu Trang},
  title     = {O\_O-VC: Synthetic Data-Driven One-to-One Alignment for Any-to-Any Voice Conversion},
  booktitle = {Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)},
  year      = {2025},
}
```

---

## 🧠 Acknowledgements

This implementation builds upon **FreeVC** 

```