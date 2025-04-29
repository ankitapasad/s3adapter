import os
import math
import numpy as np
import sys
import torch
import random
import shutil
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from ..model import *
from .dataset import SlueHvbDacDataset


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        # import pdb; pdb.set_trace()
        self.objrc = downstream_expert['objrc']

        self.get_dataset()

        self.train_dataset = SlueHvbDacDataset('fine-tune', self.train_df, self.base_path, self.label_map)
        self.dev_dataset = SlueHvbDacDataset('dev', self.valid_df, self.base_path, self.label_map)
        self.test_dataset = SlueHvbDacDataset('test', self.test_df, self.base_path, self.label_map)

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        if self.modelrc['use_proj']:
            self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
            self.model = model_cls(
                input_dim = self.modelrc['projector_dim'],
                output_dim = self.num_classes,
                **model_conf,
            )
        else:
            self.model = model_cls(
                input_dim = upstream_dim,
                output_dim = self.num_classes,
                **model_conf,
            )

        # self.objective = F.binary_cross_entropy_with_logits(reduction='none')

        if self.objrc['pos_wt'] is None or self.objrc['pos_wt'] == 'None':
            self.objective = nn.BCEWithLogitsLoss(reduction='none')
        elif self.objrc['pos_wt'] == 'ones':
            self.objective = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.ones([self.num_classes]))
        elif self.objrc['pos_wt'] == 'balanced':
            self.objective = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.balanced_pos_weight)
        else:
            self.objective = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([self.objrc['pos_wt']]))
        self.expdir = expdir
        self.register_buffer('best_score', torch.zeros(1))
        self.register_buffer('best_score_acc', torch.zeros(1))
        self.register_buffer('best_score_f1', torch.zeros(1))

    def get_dataset(self):
        self.base_path = self.datarc['file_path']
        train_df = pd.read_csv(os.path.join(self.base_path, "slue-hvb_fine-tune.tsv"), sep='\t')
        valid_df = pd.read_csv(os.path.join(self.base_path, "slue-hvb_dev.tsv"), sep='\t')
        test_df = pd.read_csv(os.path.join(self.base_path, "slue-hvb_test.tsv"), sep='\t')


        # Create a mapping from labels to indices
        label_map = {}
        all_labels = [item for item_label in train_df.dialog_acts for item in eval(item_label)]
        label_cnt = Counter(all_labels)
        for index, label in enumerate(label_cnt):
            label_map[label] = index
            label_map[index] = label

        self.label_map = label_map
        self.num_classes = len(self.label_map)//2
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

        # Creating pos_weight vector for based on counts in the train set
        if self.objrc['pos_wt'] == 'balanced':
            y_labels = []
            for idx in range(len(train_df)):
                str_labels = eval(train_df.dialog_acts[idx])
                idx_labels = [self.label_map[label] for label in str_labels]
                y_onehot = np.eye(self.num_classes)[idx_labels]
                y_labels.append(y_onehot.sum(axis=0))
            y_labels = torch.tensor(np.array(y_labels))
            self.balanced_pos_weight = (y_labels==0.).sum(axis=0)/y_labels.sum(axis=0)
            self.balanced_pos_weight = self.balanced_pos_weight/self.balanced_pos_weight.sum()

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    def safe_divide(self, numerator, denominator):
        numerator = np.array(numerator)
        denominator = np.array(denominator)
        mask = denominator == 0.0
        denominator = denominator.copy()
        denominator[mask] = 1  # avoid infs/nans
        return numerator / denominator

    def eval_f1(self, tp, fp, fn, avg='macro'):
        tp = np.sum(np.reshape(np.array(tp), (-1, self.num_classes)), axis=0)
        fp = np.sum(np.reshape(np.array(fp), (-1, self.num_classes)), axis=0)
        fn = np.sum(np.reshape(np.array(fn), (-1, self.num_classes)), axis=0)
        if avg == 'macro':
            p, r = self.safe_divide(tp, tp + fp), self.safe_divide(tp, tp + fn)
            f1 = np.mean(self.safe_divide(2 * p * r, p + r))
        elif avg == 'micro':
            tp = np.sum(tp)
            fp = np.sum(fp)
            fn = np.sum(fn)
            p, r = self.safe_divide(tp, tp + fp), self.safe_divide(tp, tp + fn)
            f1 = self.safe_divide(2 * p * r, p + r)
        return f1
    
    def convert_multihot_to_labels(self, multihot):
        """
        convert b x num_classes array to list of labels
        """
        indices = torch.nonzero(multihot, as_tuple=False)

        actions_str = [[] for _ in range(multihot.size(0))]
        for idx in indices:
            batch_idx, class_idx = idx[0].item(), idx[1].item()
            actions_str[batch_idx].append(self.label_map[class_idx])
        return actions_str
    
    # Interface
    def forward(self, mode, features, labels, filenames, records, **kwargs):
        labels = [torch.tensor(label, dtype=torch.float32) for label in labels]
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=features[0].device)
        features = pad_sequence(features, batch_first=True)

        if self.modelrc['use_proj']:
            features = self.projector(features)
        action_logits, _ = self.model(features, features_len)
        labels = torch.stack(labels).to(features.device)

        # bce
        # m = torch.nn.Sigmoid()
        # action_logits = m(action_logits)
        # action_loss = torch.nn.functional.binary_cross_entropy(
        #     action_logits, labels, reduction='sum'
        # )

        action_loss = self.objective(action_logits, labels) # batch_size x num_classes
        action_loss = action_loss.mean(dim=0) # num_classes
        
        threshold = 0.5
        # pred = action_logits > threshold
        pred = action_logits.sigmoid() > threshold
        acc = [((pred == labels).long().sum()/pred.numel()).cpu().item()] # accuracy
        if mode == 'dev' or mode == 'test':
            bool_labels = labels.bool()
            records['tp'] += [(pred & bool_labels).long().sum(axis=0).cpu().numpy()] # true positive
            records['fn'] += [(~pred & bool_labels).long().sum(axis=0).cpu().numpy()] # false negative
            records['tn'] += [(~pred & ~bool_labels).long().sum(axis=0).cpu().numpy()] # true negative
            records['fp'] += [(pred & ~bool_labels).long().sum(axis=0).cpu().numpy()] # false positive
            
        records['acc'] += acc
        records['action_loss'].append(action_loss.mean().cpu().item())

        records["filename"] += filenames
        records['predict'] += self.convert_multihot_to_labels(pred)
        records['truth'] += self.convert_multihot_to_labels(labels)

        return action_loss.mean()

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "action_loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'fluent_commands/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score_acc:
                        self.best_score_acc = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best-acc.ckpt')
                if key == 'action_loss':
                    f.write(f'{mode} loss at step {global_step}: {average}\n')
        if mode == 'dev' or mode == 'test':
            macro_f1 = self.eval_f1(records['tp'], records['fp'], records['fn'], avg='macro')
            micro_f1 = self.eval_f1(records['tp'], records['fp'], records['fn'], avg='micro')
            with open(Path(self.expdir) / "log.log", 'a') as f:
                print(f"{mode} macro F1: {macro_f1}")
                print(f"{mode} micro F1: {micro_f1}")
                f.write(f'{mode} macro F1 at step {global_step}: {macro_f1}\n')
                f.write(f'{mode} micro F1 at step {global_step}: {micro_f1}\n')
                if mode == 'dev' and macro_f1 > self.best_score_f1:
                    self.best_score_f1 = torch.ones(1) * macro_f1
                    f.write(f'New best macro F1 on {mode} at step {global_step}: {macro_f1}\n')
                    save_names.append(f'{mode}-best-macro-f1.ckpt')
            if global_step >= 20000 and macro_f1 == 0.0:
                sys.exit()


        with open(Path(self.expdir) / f"{mode}_predict_{global_step}.csv", "w") as file:
            for f, list_of_labels in zip(records["filename"], records["predict"]):
                file.write(f"{f},{','.join(list_of_labels)}\n")

        with open(Path(self.expdir) / f"{mode}_truth_{global_step}.csv", "w") as file:
            for f, list_of_labels in zip(records["filename"], records["truth"]):
                file.write(f"{f},{','.join(list_of_labels)}\n")
        

        return save_names
