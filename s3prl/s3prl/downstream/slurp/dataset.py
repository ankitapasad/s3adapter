import random
import json
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data.dataset import Dataset

import pandas as pd

SAMPLE_RATE = 16000
EXAMPLE_WAV_MIN_SEC = 5
EXAMPLE_WAV_MAX_SEC = 20
EXAMPLE_DATASET_SIZE = 200


class SlurpDataset(Dataset):
    def __init__(self, csv_fn, class_name='scenario', i2s=None):
        df = pd.read_csv(csv_fn)
        self.ids = df['ID'].tolist()
        self.wav_fns = df['wav'].tolist()
        s2i = dict([(s, i) for i, s in enumerate(i2s)])
        self.i2s, self.s2i = i2s, s2i
        self.labels = []
        semantic_types = []
        assert class_name in {'scenario', 'action'}
        durations = df['duration'].tolist()
        for semantic in df['semantics']:
            items = semantic.split('|')
            if class_name == 'scenario':
                label = items[0].split(':')[1].strip()[1:-1]
            elif class_name == 'action':
                label = items[1].split(':')[1].strip()[1:-1]
            self.labels.append(label)
        print(f"{len(self.wav_fns)} wav files, "
              f"min duration (sec): {min(durations)} "
              f"max duration (sec): {max(durations)}")


    def __len__(self):
        return len(self.wav_fns)
        
    def __getitem__(self, idx):
        data_id = self.ids[idx]
        wav_fn, label = self.wav_fns[idx], self.labels[idx]
        wav, sr = torchaudio.load(wav_fn)
        wav = wav.squeeze(0)

        assert sr == SAMPLE_RATE
        label = self.s2i[label]
        return wav.numpy(), label, data_id

    def collate_fn(self, samples):
        return zip(*samples)

class RandomDataset(Dataset):
    def __init__(self, **kwargs):
        self.class_num = 48

    def __getitem__(self, idx):
        samples = random.randint(EXAMPLE_WAV_MIN_SEC * SAMPLE_RATE, EXAMPLE_WAV_MAX_SEC * SAMPLE_RATE)
        wav = torch.randn(samples)
        label = random.randint(0, self.class_num - 1)
        return wav, label

    def __len__(self):
        return EXAMPLE_DATASET_SIZE

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels


def test():
    csv_fn = "/share/data/speech/hackathon_2022/data/slurp/csvs/train-type=direct.csv"
    dict_fn = "downstream/slurp/class_dict.json"
    classname = 'scenario'
    i2s = json.load(open(dict_fn))[classname]
    dataset = SlurpDataset(csv_fn, class_name=classname, i2s=i2s)
    s1, s2 = dataset[0], dataset[10]
    data = dataset.collate_fn([s1, s2])
    print(data)
    return


if __name__ == '__main__':
    test()
