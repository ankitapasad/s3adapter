# copy of asr/dataset.py


###############
# IMPORTATION #
###############
import logging
import os
import random
#-------------#
import pandas as pd
from tqdm import tqdm
from pathlib import Path
#-------------#
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
#-------------#
import torchaudio
#-------------#
from .dictionary import Dictionary

SAMPLE_RATE = 16000
HALF_BATCHSIZE_TIME = 2000


####################
# Sequence Dataset #
####################
class SequenceDataset(Dataset):
    
    def __init__(self, split, bucket_size, dictionary, vp_data, bucket_file, **kwargs):
        super(SequenceDataset, self).__init__()
        
        self.dictionary = dictionary
        self.vp_data = vp_data
        self.sample_rate = SAMPLE_RATE
        self.split_sets = kwargs[split]
        self.split = self.split_sets[0]
        self.vp_manifest = kwargs["vp_manifest"]

        # Read table for bucketing
        assert os.path.isdir(bucket_file), 'Please first run `python3 preprocess/generate_len_for_bucket.py -h` to get bucket file.'

        # Wavs
        table_list = []
        for item in self.split_sets:
            file_path = os.path.join(bucket_file, item + ".csv")
            if os.path.exists(file_path):
                table_list.append(
                    pd.read_csv(file_path)
                )
            else:
                logging.warning(f'{item} is not found in bucket_file: {bucket_file}, skipping it.')

        table_list = pd.concat(table_list)
        table_list = table_list.sort_values(by=['length'], ascending=False)

        X = table_list['file_path'].tolist()
        X_lens = table_list['length'].tolist()

        assert len(X) != 0, f"0 data found for {split}"

        # Transcripts
        Y = self._load_transcript(X)

        x_names = set([self._parse_x_name(x) for x in X])
        y_names = set(Y.keys())
        usage_list = list(x_names & y_names)
        assert len(usage_list) != 0, f"0 data found for {split}"

        Y = {key: Y[key] for key in usage_list}

        self.Y = {
            k: self.dictionary.encode_line(
                v, line_tokenizer=lambda x: x.split()
            ).long() 
            for k, v in Y.items()
        }

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in tqdm(zip(X, X_lens), total=len(X), desc=f'ASR dataset {split}', dynamic_ncols=True):
            if self._parse_x_name(x) in usage_list:
                batch_x.append(x)
                batch_len.append(x_len)
                
                # Fill in batch_x until batch is full
                if len(batch_x) == bucket_size:
                    # Half the batch size if seq too long
                    if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                        self.X.append(batch_x[:bucket_size//2])
                        self.X.append(batch_x[bucket_size//2:])
                    else:
                        self.X.append(batch_x)
                    batch_x, batch_len = [], []
        
        # Gather the last batch
        if len(batch_x) > 1:
            if self._parse_x_name(x) in usage_list:
                self.X.append(batch_x)

    def _parse_x_name(self, x):
        return x.split('/')[-1]

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(os.path.join(self.vp_data, wav_path))
        assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)

    def _load_transcript(self, x_list):
        """Load the transcripts for Librispeech"""

        with open(os.path.join(self.vp_manifest, f"{self.split}.combined.ltr"), "r") as f:
            transcripts = [line.strip() for line in f.readlines()]
        with open(os.path.join(self.vp_manifest, f"{self.split}.tsv"), "r") as f:
            file_paths = [line.strip() for line in f.readlines()]
        trsp_sequences = {}
        for idx, transcript in enumerate(transcripts):
            trsp_sequences[file_paths[idx+1].split("\t")[0]] = transcript
        
        return trsp_sequences

    def _build_dictionary(self, transcripts, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = Dictionary()
        transcript_list = list(transcripts.values())
        Dictionary.add_transcripts_to_dictionary(
            transcript_list, d, workers
        )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav_batch = [self._load_wav(x_file).numpy() for x_file in self.X[index]]
        label_batch = [self.Y[self._parse_x_name(x_file)].numpy() for x_file in self.X[index]]
        filename_batch = [Path(x_file).stem for x_file in self.X[index]]
        return wav_batch, label_batch, filename_batch # bucketing, return ((wavs, labels))

    def collate_fn(self, items):
        assert len(items) == 1
        return items[0][0], items[0][1], items[0][2] # hack bucketing, return (wavs, labels, filenames)
