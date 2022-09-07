import json
import numpy as np
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import logging
import os
import h5py
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(os.path.basename(__file__))


class MultipleChoiceDataset:
    def __init__(self, data_args, model_args):
        self.task_name = data_args.task_name
        self.dataset_name = data_args.task_name
        self.split = data_args.split
        self.dataset_path = data_args.dataset_path
        self.model_name = model_args.model_name
        self.vocab_file = model_args.vocab_file

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) if self.model_name != 'LSTM' else None
        self.max_seq_length = data_args.max_seq_length

        logger.info('loading metric')
        self.eval_metric = 'f1'
        self.accuracy = load_metric('accuracy')
        self.f1 = load_metric('f1')

        logger.info('loading dataset')
        if self.dataset_path:
            with h5py.File(self.dataset_path, 'r+') as df:
                attributes = json.loads(df["attributes"][()])

                self.label_vocab = attributes['label_vocab']
                index_list = attributes['index_list']
                train_idx = attributes['train_index_list']
                test_idx = attributes['test_index_list']
                eval_idx = attributes['eval_index_list'] if 'eval_index_list' in attributes else test_idx

                if 'doc_index' in attributes:
                    doc_index = attributes['doc_index']
                else:
                    doc_index = {str(i): 0 for i in index_list}
                self.num_labels = attributes['num_labels']

                unique_docs = set()

                train_dict, eval_dict, test_dict = ({'text': [], 'label': [], 'doc': []} for _ in range(3))
                for idx in train_idx:
                    text = df['X'][str(idx)][()].decode('UTF-8')
                    train_dict['text'].append(text)
                    label = df['Y'][str(idx)][()].decode('UTF-8')
                    train_dict['label'].append(self.label_vocab[label])
                    doc = doc_index[str(idx)]
                    train_dict['doc'].append(doc)
                    unique_docs.add(doc)
                for idx in eval_idx:
                    text = df['X'][str(idx)][()].decode('UTF-8')
                    eval_dict['text'].append(text)
                    label = df['Y'][str(idx)][()].decode('UTF-8')
                    eval_dict['label'].append(self.label_vocab[label])
                    doc = doc_index[str(idx)]
                    eval_dict['doc'].append(doc)
                    unique_docs.add(doc)
                for idx in test_idx:
                    text = df['X'][str(idx)][()].decode('UTF-8')
                    test_dict['text'].append(text)
                    label = df['Y'][str(idx)][()].decode('UTF-8')
                    test_dict['label'].append(self.label_vocab[label])
                    doc = doc_index[str(idx)]
                    test_dict['doc'].append(doc)
                    unique_docs.add(doc)

            train_dataset = Dataset.from_dict(train_dict)
            eval_dataset = Dataset.from_dict(eval_dict)
            test_dataset = Dataset.from_dict(test_dict)
        else:
            raw_datasets = load_dataset(self.task_name, self.dataset_name, split=self.split)
            label_list = raw_datasets['train'].features['label'].names
            self.num_labels = len(label_list)
            self.label_vocab = {l: i for i, l in enumerate(label_list)}

            train_dataset = raw_datasets["train"]
            eval_dataset = raw_datasets["validation"]
            test_dataset = raw_datasets["test"]

        self.remove_columns = train_dataset.column_names
        logger.info('processing train dataset')
        if data_args.max_train_samples is not None:
            sampled_idxes = random.sample(range(len(train_dataset)), k=data_args.max_train_samples)
            train_dataset = train_dataset.select(sampled_idxes)
        self.train_dataset = train_dataset.map(self.process_fn, batched=True)

        logger.info('processing eval dataset')
        if data_args.max_eval_samples is not None:
            sampled_idxes = random.sample(range(len(eval_dataset)), k=data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(sampled_idxes)
        self.eval_dataset = eval_dataset.map(self.process_fn, batched=True)

        logger.info('processing test dataset')
        if data_args.max_test_samples is not None:
            sampled_idxes = random.sample(range(len(test_dataset)), k=data_args.max_test_samples)
            test_dataset = test_dataset.select(sampled_idxes)
        self.test_dataset = test_dataset.map(self.process_fn, batched=True)

    def process_fn(self, examples):
        text = examples['text']
        if self.model_name == 'LSTM':
            raise NotImplementedError
        else:
            ending_names = ["ending0", "ending1", "ending2", "ending3"]
            first_sentences = [[context] * 4 for context in examples["sent1"]]
            question_headers = examples["sent2"]
            second_sentences = [
                [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
            ]

            first_sentences = sum(first_sentences, [])
            second_sentences = sum(second_sentences, [])

            tokenized_examples = self.tokenizer(first_sentences, second_sentences, truncation=True)
            return {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


def compute_metrics(self, labels, logits):
    labels, logits = labels[0], logits[0]
    pred_labels = np.argmax(logits, axis=-1)

    accuracy = self.accuracy.compute(predictions=pred_labels, references=labels)
    f1 = self.f1.compute(predictions=pred_labels, references=labels, average='macro')
    res = {'accuracy': accuracy['accuracy'], 'f1': f1['f1']}
    return res


def get_loader(self, dataset, batch_size, shuffle=False):
    if len(dataset) == 0:
        return None
    data_collator = DataCollatorForMultipleChoice(tokenizer=self.tokenizer)
    dataset = dataset.remove_columns(self.remove_columns)
    loader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=shuffle)
    return loader


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
