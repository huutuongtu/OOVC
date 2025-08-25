import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import commons 
from mel_processing import spectrogram_torch, spec_to_mel_torch
from utils import load_wav_to_torch, load_filepaths_and_text, transform


class TextAudioSpeakerLoader(torch.utils.data.Dataset):

    def __init__(self, audiopaths, hparams):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length  = hparams.data.filter_length
        self.hop_length     = hparams.data.hop_length
        self.win_length     = hparams.data.win_length
        self.sampling_rate  = hparams.data.sampling_rate
        self.use_spk = hparams.model.use_spk
        self.spec_len = hparams.train.max_speclen

        random.seed(1234)
        random.shuffle(self.audiopaths)
        self._filter()

    def _filter(self):
        lengths = []
        for audiopath in self.audiopaths:
            lengths.append(os.path.getsize(audiopath[0]) // (2 * self.hop_length))
        self.lengths = lengths

    def get_audio(self, src, tgt):
        audio, sampling_rate = load_wav_to_torch(tgt)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio.unsqueeze(0)
        spec_filename = tgt.replace(".flac", ".spec.pt").replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
            
        # spk embedding
        if self.use_spk:
            spk_filename = tgt.replace(".flac", ".freevc.npy").replace(".wav", ".freevc.npy")
            spk = torch.from_numpy(np.load(spk_filename))
        
        # f0_tgt for condition
        f0_src = src.replace(".flac", ".pitch_raw.npy").replace(".wav", ".pitch_raw.npy")
        f0_src = torch.from_numpy(np.load(f0_src))
        f0_tgt = tgt.replace(".flac", ".pitch_raw.npy").replace(".wav", ".pitch_raw.npy")
        f0_tgt = torch.from_numpy(np.load(f0_tgt))
        
        # wavlm 
        c_filename = src.replace(".flac", ".pt").replace(".wav", ".pt")
        c = torch.load(c_filename).squeeze(0)

        if self.use_spk:
            return c, spec, audio_norm, spk, f0_src, f0_tgt
        else:
            return c, spec, audio_norm, f0_src, f0_tgt

    def __getitem__(self, index):
        return self.get_audio(self.audiopaths[index][0], self.audiopaths[index][1])

    def __len__(self):
        return len(self.audiopaths)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, hps):
        self.hps = hps
        self.use_sr = hps.train.use_sr
        self.use_spk = hps.model.use_spk

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]),
            dim=0, descending=True)

        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        if self.use_spk:
            spks = torch.FloatTensor(len(batch), batch[0][3].size(0))
        else:
            spks = None
        
        c_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        f0_src_padded = torch.FloatTensor(len(batch), max_spec_len)
        f0_tgt_padded = torch.FloatTensor(len(batch), max_spec_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        f0_src_padded.zero_()
        f0_tgt_padded.zero_()
        c_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            
            c = row[0]
            c_padded[i, :, :c.size(1)] = c

            f0_src = row[4]
            f0_tgt = row[5]
            
            f0_src_padded[i, :f0_src.size(0)] = f0_src

            f0_tgt_padded[i, :f0_tgt.size(0)] = f0_tgt
            
            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
            
            if self.use_spk:
                spks[i] = row[3]
        
        spec_seglen = spec_lengths[-1] if spec_lengths[-1] < self.hps.train.max_speclen + 1 else self.hps.train.max_speclen + 1
        wav_seglen = spec_seglen * self.hps.data.hop_length 

        spec_padded, ids_slice = commons.rand_spec_segments(spec_padded, spec_lengths, spec_seglen)
        wav_padded = commons.slice_segments(wav_padded, ids_slice * self.hps.data.hop_length, wav_seglen)
        
        c_padded = commons.slice_segments(c_padded, ids_slice, spec_seglen)[:,:,:-1]

        f0_src_padded = commons.slice_segments(f0_src_padded.unsqueeze(1), ids_slice, spec_seglen)[:,:,:-1]
        f0_tgt_padded = commons.slice_segments(f0_tgt_padded.unsqueeze(1), ids_slice, spec_seglen)[:,:,:-1]
    
        spec_padded = spec_padded[:,:,:-1]
        wav_padded = wav_padded[:,:,:-self.hps.data.hop_length]

        if self.use_spk:
          return c_padded, spec_padded, wav_padded, spks, f0_src_padded, f0_tgt_padded
        else:
          return c_padded, spec_padded, wav_padded, f0_src_padded, f0_tgt_padded


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):

      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
