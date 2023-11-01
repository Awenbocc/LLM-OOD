import torch
import random
import numpy as np
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataclasses import dataclass

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class DataCollatorForSeq2Cls:
    tokenizer: None
    padding: bool= True
    max_length: Optional[int] = None
    pad_to_multiple_of: int = 8
    def __call__(self, features):
        self.tokenizer.padding_side = "left" 
        input_ids, attention_mask, labels = tuple([instance[key] for instance in features]
                                  for key in ("input_ids", "attention_mask", "labels"))

        features = self.tokenizer.pad(
            dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels),
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        # features['labels'] = labels
        return features
    

@dataclass
class DataCollatorForSeq2Match:
    tokenizer: None
    padding: bool= True
    max_length: Optional[int] = None
    pad_to_multiple_of: int = 8
    def __call__(self, features):
        self.tokenizer.padding_side = "right" 
        input_ids, attention_mask, labels = tuple([instance[key] for instance in features]
                                  for key in ("input_ids", "attention_mask", "labels"))

        features = self.tokenizer.pad(
            dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels),
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        # features['labels'] = labels
        return features
    

@dataclass
class DataCollatorForSeq2Seq:
    tokenizer: None
    padding: bool= True
    max_length: Optional[int] = None
    label_pad_token_id: int = -100
    pad_to_multiple_of: int = 8
    def __call__(self, features):
        
        self.tokenizer.padding_side = "right" 
        input_ids, attention_mask, labels = tuple([instance[key] for instance in features]
                                  for key in ("input_ids", "attention_mask", "labels"))
        new_labels = []
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            padding_side = self.tokenizer.padding_side
            for label in labels:
                remainder = [self.label_pad_token_id] * (max_label_length - len(label))
                if isinstance(label, list):
                    label = (
                        label + remainder if padding_side == "right" else remainder + label
                    )
                elif padding_side == "right":
                    label = np.concatenate([label, remainder]).astype(np.int64)
                else:
                    label = np.concatenate([remainder, label]).astype(np.int64)
                new_labels.append(label)

        features = self.tokenizer.pad(
            dict(input_ids=input_ids, attention_mask=attention_mask, labels = new_labels),
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )     
        
        return features

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    outputs = {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "labels": labels,
    }
    return outputs

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0]

