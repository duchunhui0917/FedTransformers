import os
import random

from datasets.load import load_dataset, load_metric
from transformers import AutoTokenizer
import numpy as np
import logging
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    mean_squared_error
from .utils import add_space_for_punctuation
import torch

logger = logging.getLogger(__name__)
base_dir = os.path.expanduser('~/FedTransformers')

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


class GlueDataset:
    def __init__(self, data_args, model_args):
        super().__init__()
        self.model_name = model_args.model_name
        self.vocab_file = model_args.vocab_file
        cache_dir = os.path.join(base_dir, 'data')
        raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=cache_dir)
        self.ignored_columns = set(raw_datasets['train'].column_names) - {'label'}

        self.max_seq_length = data_args.max_seq_length

        self.data_args = data_args
        # labels
        self.is_regression = data_args.task_name == "stsb"
        if not self.is_regression:
            self.label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {i: l for l, i in self.label2id.items()}
            logger.info(self.label2id)
            logger.info(self.id2label)
        else:
            self.num_labels = 1

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[data_args.task_name]

        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            sampled_idxes = random.sample(range(len(train_dataset)), k=data_args.max_train_samples)
            train_dataset = train_dataset.select(sampled_idxes)
        logger.info('tokenizing train dataset')
        self.train_dataset = train_dataset.map(self.process_fn, batched=True, batch_size=len(train_dataset))

        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            sampled_idxes = random.sample(range(len(eval_dataset)), k=data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(sampled_idxes)
        logger.info('tokenizing eval dataset')
        self.eval_dataset = eval_dataset.map(self.process_fn, batched=True, batch_size=len(eval_dataset))

        test_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            sampled_idxes = random.sample(range(len(test_dataset)), k=data_args.max_test_samples)
            test_dataset = test_dataset.select(sampled_idxes)
        logger.info('tokenizing test dataset')
        self.test_dataset = test_dataset.map(self.process_fn, batched=True, batch_size=len(test_dataset))

        self.metric = load_metric("glue", data_args.task_name)

    def process_fn(self, examples):
        if self.model_name == 'LSTM':
            inputs = {}
            with open(self.vocab_file, 'r') as f:
                vocab = json.load(f)
            input_ids = []

            if self.sentence2_key is None:
                t = tqdm(examples[self.sentence1_key])

                for s in t:
                    input_id = []
                    s = add_space_for_punctuation(s)
                    ls = s.split()
                    for token in ls:
                        token = token.lower()
                        if token in vocab:
                            input_id.append(vocab[token])
                        else:
                            input_id.append(vocab['[OOV]'])
                    input_ids.append(input_id)
            else:
                t = tqdm(zip(examples[self.sentence1_key], examples[self.sentence2_key]))
                for s1, s2 in t:
                    input_id = []
                    s1, s2 = add_space_for_punctuation(s1), add_space_for_punctuation(s2)
                    ls1, ls2 = s1.split(), s2.split()
                    for token in ls1:
                        token = token.lower()
                        if token in vocab:
                            input_id.append(vocab[token])
                        else:
                            input_id.append(vocab['[OOV]'])
                    input_id.append('[PAD]')
                    for token in ls2:
                        token = token.lower()
                        if token in vocab:
                            input_id.append(vocab[token])
                        else:
                            input_id.append(vocab['[OOV]'])

                    input_ids.append(input_id)

            for i, input_id in enumerate(input_ids):
                if len(input_id) > self.max_seq_length:
                    input_ids[i] = input_id[:self.max_seq_length]
                else:
                    input_ids[i] += [vocab['[PAD]']] * (self.max_seq_length - len(input_id))
                    inputs.update({'input_ids': input_ids})
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            max_seq_length = min(self.max_seq_length, tokenizer.model_max_length)
            logger.info(f'max sequence length: {max_seq_length}')

            args = (
                (examples[self.sentence1_key],) if self.sentence2_key is None else
                (examples[self.sentence1_key], examples[self.sentence2_key])
            )
            inputs = tokenizer(*args, padding=True, truncation=True, max_length=max_seq_length)

        return inputs

    def collate_fn(self, batch):
        keys = batch[0].keys() - self.ignored_columns
        d = {key: [] for key in keys}
        for x in batch:
            for key in keys:
                d[key].append(x[key])
        for key in d.keys():
            d[key] = torch.LongTensor(d[key])
        return d

    def compute_metrics(self, labels, logits, metrics):
        res = {}
        pred_labels = np.squeeze(logits) if self.is_regression else np.argmax(logits, axis=1)

        if self.is_regression:
            mse = mean_squared_error(labels, pred_labels)
            res.update({'mse': mse})
        else:
            if 'acc' in metrics:
                acc = accuracy_score(labels, pred_labels)
                res.update({'acc': acc})
            if 'f1' in metrics:
                precision = precision_score(labels, pred_labels, average='macro')
                recall = recall_score(labels, pred_labels, average='macro')
                f1 = f1_score(labels, pred_labels, average='macro')
                res.update({'precision': precision, 'recall': recall, 'f1': f1})
            if 'confusion matrix' in metrics:
                cf = confusion_matrix(labels, pred_labels)
                cf = json.dumps(cf.tolist())
                logger.info(f'confusion matrix\n{cf}')

        return res
