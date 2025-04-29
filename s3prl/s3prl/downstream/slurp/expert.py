import os
import json
import math
import torch
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

# from .model import Model
from ..model import *
from .dataset import SlurpDataset # RandomDataset
from pathlib import Path


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        """
        Args:
            upstream_dim: int
                Different upstream will give different representation dimension
                You might want to first project them to the same dimension

            upstream_rate: int
                160: for upstream with 10 ms per frame
                320: for upstream with 20 ms per frame
            
            downstream_expert: dict
                The 'downstream_expert' field specified in your downstream config file
                eg. downstream/example/config.yaml

            expdir: string
                The expdir from command-line argument, you should save all results into
                this directory, like some logging files.

            **kwargs: dict
                All the arguments specified by the argparser in run_downstream.py
                and all the other fields in config.yaml, in case you need it.
                
                Note1. Feel free to add new argument for __init__ as long as it is
                a command-line argument or a config field. You can check the constructor
                code in downstream/runner.py
        """

        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        print(f"[Downstream expert] - class- {self.datarc['classname']}")
        base_path = os.path.join(self.datarc['file_path'], "csvs")
        train_fn, dev_fn, test_fn = f"{base_path}/train-type=direct.csv", f"{base_path}/devel-type=direct.csv", f"{base_path}/test-type=direct.csv"
        dict_path = self.datarc['dict_path']
        cls2label_dict = json.load(open(dict_path))
        i2s = cls2label_dict[self.datarc['classname']]
        self.i2s = i2s

        self.train_dataset = SlurpDataset(train_fn, class_name=self.datarc['classname'], i2s=i2s)
        self.dev_dataset = SlurpDataset(dev_fn, class_name=self.datarc['classname'], i2s=i2s)
        self.test_dataset = SlurpDataset(test_fn, class_name=self.datarc['classname'], i2s=i2s)

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = len(self.i2s),
            **model_conf,
        )

        self.objective = nn.CrossEntropyLoss()
        self.register_buffer('best_score', torch.zeros(1))
        self.expdir = expdir

    # Interface
    def get_dataloader(self, split, epoch: int = 0):
        """
        Args:
            split: string
                'train'
                    will always be called before the training loop

                'dev', 'test', or more
                    defined by the 'eval_dataloaders' field in your downstream config
                    these will be called before the evaluation loops during the training loop

        Return:
            a torch.utils.data.DataLoader returning each batch in the format of:

            [wav1, wav2, ...], your_other_contents1, your_other_contents2, ...

            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with:
                1. dim() == 1
                2. sample_rate == 16000
                3. directly loaded by torchaudio
        """

        if split == 'train':
            return self._get_train_dataloader(self.train_dataset, epoch)
        elif split == 'dev':
            return self._get_eval_dataloader(self.dev_dataset)
        elif split == 'test':
            return self._get_eval_dataloader(self.test_dataset)


    def _get_train_dataloader(self, dataset, epoch: int):
        from s3prl.utility.data import get_ddp_sampler
        sampler = get_ddp_sampler(dataset, epoch)
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )


    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )


    # Interface
    def forward(self, split, features, labels, filenames, records, **kwargs):
        """
        Args:
            split: string
                'train'
                    when the forward is inside the training loop

                'dev', 'test' or more
                    when the forward is inside the evaluation loop

            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            your_other_contents1, ... :
                in the order defined by your dataloader (dataset + collate_fn)
                these are all in cpu, and you can move them to the same device
                as features

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records (also customized by you)

                Note1. downstream/runner.py will call self.log_records
                    1. every `log_step` during training
                    2. once after evalute the whole dev/test dataloader

                Note2. `log_step` is defined in your downstream config
                eg. downstream/example/config.yaml

        Return:
            loss:
                the loss to be optimized, should not be detached
                a single scalar in torch.FloatTensor
        """
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)

        # utterance_labels = labels
        labels = torch.LongTensor(labels).to(device)

        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices

        records['loss'].append(loss.item())
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records["filename"] += filenames
        records["predict"] += [self.i2s[idx] for idx in predicted_classid.cpu().tolist()]
        records["truth"] += [self.i2s[idx] for idx in labels.cpu().tolist()]
        return loss


    # interface
    def log_records(self, split, records, logger, global_step, batch_ids, total_batch_num, **kwargs):
        """
        Args:
            split: string
                'train':
                    records and batchids contain contents for `log_step` batches
                    `log_step` is defined in your downstream config
                    eg. downstream/example/config.yaml

                'dev', 'test' or more:
                    records and batchids contain contents for the entire evaluation dataset

            records:
                defaultdict(list), contents already prepared by self.forward

            logger:
                Tensorboard SummaryWriter
                please use f'{your_task_name}/{split}-{key}' as key name to log your contents,
                preventing conflict with the logging of other tasks

            global_step:
                The global_step when training, which is helpful for Tensorboard logging

            batch_ids:
                The batches contained in records when enumerating over the dataloader

            total_batch_num:
                The total amount of batches in the dataloader
        
        Return:
            a list of string
                Each string is a filename we wish to use to save the current model
                according to the evaluation result, like the best.ckpt on the dev set
                You can return nothing or an empty list when no need to save the checkpoint
        """
        save_names = []
        for key in ["acc", "loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'slurp/{split}-{key}',
                average,
                global_step=global_step
            )

            with open(f"{self.expdir}/log.log", 'a') as fo:
                if key == 'acc':
                    print(f"{split} {key}: {average}")
                    fo.write(f'{split} at step {global_step}: {average}\n')
                    if split == 'dev' and average > self.best_score:
                        fo.write(f'New best on {split} at step {global_step}: {average}\n')

            if split == 'dev' and key == 'acc' and average > self.best_score:
                self.best_score = torch.ones(1) * average
                save_names.append(f'{split}-best.ckpt')
        with open(Path(self.expdir) / f"{split}_predict.csv", "w") as file:
            lines = [f"{f},{a}\n" for f, a in zip(records["filename"], records["predict"])]
            file.writelines(lines)

        with open(Path(self.expdir) / f"{split}_truth.csv", "w") as file:
            lines = [f"{f},{a}\n" for f, a in zip(records["filename"], records["truth"])]
            file.writelines(lines)

        return save_names
