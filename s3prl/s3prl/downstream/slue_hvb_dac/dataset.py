import os
import random
from pathlib import Path

import torch
import torchaudio
import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import Dataset

SAMPLE_RATE = 16000
EXAMPLE_WAV_MAX_SEC = 10


class SlueHvbDacDataset(Dataset):
    def __init__(self, data_split, df, base_path, label_map):
        self.data_split = data_split
        self.df = df
        self.base_path = base_path
        self.max_length = SAMPLE_RATE * EXAMPLE_WAV_MAX_SEC
        self.label_map = label_map
        self.num_classes = len(label_map) // 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        issue_id = self.df.issue_id[idx]
        start_ms = self.df.start_ms[idx]
        end_ms = start_ms + self.df.duration_ms[idx]
        fname = f'{issue_id}_{start_ms}_{end_ms}.wav'
        wav_path = os.path.join(self.base_path, self.data_split, fname)
        wav, sr = torchaudio.load(wav_path)

        assert sr == SAMPLE_RATE
        
        wav = wav.squeeze(0)
        str_labels = eval(self.df.dialog_acts[idx])
        idx_labels = [self.label_map[label] for label in str_labels]
        y_onehot = np.eye(self.num_classes)[idx_labels]
        y_multihot = y_onehot.sum(axis=0)
        return wav.numpy(), y_multihot, Path(wav_path).stem

    def collate_fn(self, samples):
        return zip(*samples)
