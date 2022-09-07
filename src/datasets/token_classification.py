import json
import numpy as np
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import logging
import os
from transformers import AutoTokenizer, DataCollatorForTokenClassification
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader
import random

from .utils import add_space_for_punctuation

logger = logging.getLogger(os.path.basename(__file__))


class TokenClassificationDataset:
    def __init__(self, data_args, model_args):
        self.task_name = data_args.task_name
        self.dataset_name = data_args.dataset_name
        self.split = data_args.split
        self.dataset_path = data_args.dataset_path
        self.model_name = model_args.model_name
        self.vocab_file = model_args.vocab_file

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) if self.model_name != 'LSTM' else None
        self.max_seq_length = data_args.max_seq_length

        logger.info('loading metric')
        self.eval_metric = 'f1'
        self.seqeval = load_metric("seqeval")

        logger.info('loading dataset')
        if self.dataset_path:
            with h5py.File(self.dataset_path, 'r+') as df:
                attributes = json.loads(df["attributes"][()])

                self.label2id = attributes['label_vocab']
                self.id2label = {v: k for k, v in self.label2id.items()}
                self.label_names = list(self.label2id.keys())

                index_list = attributes['index_list']
                train_idx = attributes['train_index_list']
                eval_idx = attributes['test_index_list']
                test_idx = attributes['test_index_list']
                if 'doc_index' in attributes:
                    doc_index = attributes['doc_index']
                else:
                    doc_index = {str(i): 0 for i in index_list}
                self.num_labels = attributes['num_labels']
                self.unique_docs = set()

                train_dict, eval_dict, test_dict = ({'text': [], 'label': [], 'doc': []} for _ in range(3))
                for idx in train_idx:
                    text = df['X'][str(idx)][()]
                    text = [i.decode('UTF-8') for i in text]
                    train_dict['text'].append(text)

                    label = df['Y'][str(idx)][()]
                    label = [self.label2id[i.decode('UTF-8')] for i in label]
                    train_dict['label'].append(label)
                    doc = doc_index[str(idx)]
                    train_dict['doc'].append(doc)
                    self.unique_docs.add(doc)
                for idx in eval_idx:
                    text = df['X'][str(idx)][()]
                    text = [i.decode('UTF-8') for i in text]
                    eval_dict['text'].append(text)

                    label = df['Y'][str(idx)][()]
                    label = [self.label2id[i.decode('UTF-8')] for i in label]
                    eval_dict['label'].append(label)
                    doc = doc_index[str(idx)]
                    eval_dict['doc'].append(doc)
                    self.unique_docs.add(doc)
                for idx in test_idx:
                    text = df['X'][str(idx)][()]
                    text = [i.decode('UTF-8') for i in text]
                    test_dict['text'].append(text)

                    label = df['Y'][str(idx)][()]
                    label = [self.label2id[i.decode('UTF-8')] for i in label]
                    test_dict['label'].append(label)
                    doc = doc_index[str(idx)]
                    test_dict['doc'].append(doc)
                    self.unique_docs.add(doc)

            train_dataset = Dataset.from_dict(train_dict)
            eval_dataset = Dataset.from_dict(eval_dict)
            test_dataset = Dataset.from_dict(test_dict)
        else:
            raw_datasets = load_dataset(self.task_name, self.dataset_name, split=self.split)

            train_dataset = raw_datasets["train"]
            eval_dataset = raw_datasets["validation"]
            test_dataset = raw_datasets["test"]

            ner_feature = raw_datasets["train"].features["ner_tags"]
            self.label_names = ner_feature.feature.names
            self.id2label = {str(i): label for i, label in enumerate(self.label_names)}
            self.label2id = {v: k for k, v in self.id2label.items()}

        self.remove_columns = train_dataset.column_names
        logger.info('processing train dataset')
        if data_args.max_train_samples is not None:
            sampled_idxes = random.sample(range(len(train_dataset)), k=data_args.max_train_samples)
            train_dataset = train_dataset.select(sampled_idxes)
        self.train_dataset = train_dataset.map(self.process_fn, batched=True, load_from_cache_file=False)
        logger.info('processing eval dataset')
        if data_args.max_eval_samples is not None:
            sampled_idxes = random.sample(range(len(eval_dataset)), k=data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(sampled_idxes)
        self.eval_dataset = eval_dataset.map(self.process_fn, batched=True, load_from_cache_file=False)
        logger.info('processing test dataset')
        if data_args.max_test_samples is not None:
            sampled_idxes = random.sample(range(len(test_dataset)), k=data_args.max_test_samples)
            test_dataset = test_dataset.select(sampled_idxes)
        self.test_dataset = test_dataset.map(self.process_fn, batched=True, load_from_cache_file=False)

    def process_fn(self, examples):
        text = examples['text']
        label = examples['label']

        if self.model_name == 'LSTM':
            logger.info(f'max sequence length: {self.max_seq_length}')
            with open(self.vocab_file, 'r') as f:
                vocab = json.load(f)
            embeddings = []
            t = tqdm(examples)

            for i, s in enumerate(t):
                embedding = []
                s = add_space_for_punctuation(s)
                ls = s.split()
                for token in ls:
                    token = token.lower()
                    if token in vocab:
                        embedding.append(vocab[token])
                    else:
                        embedding.append(vocab['[OOV]'])
                embeddings.append(embedding)

            lengths = [len(embedding) for embedding in embeddings]
            max_seq_length = max(lengths)
            logger.info(f'max sequence length: {max_seq_length}')
            for i, embedding in enumerate(embeddings):
                if len(embedding) < max_seq_length:
                    embeddings[i] += [vocab['[PAD]']] * (max_seq_length - len(embedding))
            inputs = {'embeddings': embeddings}
            label = [x + [self.label2id['O']] * (max_seq_length - len(x)) for x in label]
            inputs.update({'label': label})
            return inputs
        else:
            if self.task_name in ["conll2003"]:
                text = examples["tokens"]
                tags = examples["ner_tags"]
            else:
                text = examples["text"]
                tags = examples['label']

            tokenized_inputs = self.tokenizer(text, truncation=True, is_split_into_words=True)

            labels = []
            for i, label in enumerate(tags):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

    def compute_metrics(self, labels, logits):
        '''

        Args:
            labels[0]: (N, L)
            logits[0]: (N, L)

        Returns:

        '''
        labels, logits = labels[0], logits[0]
        if isinstance(logits, list):
            labels = [label for batch_labels in labels for label in batch_labels]
            predictions = [np.argmax(logit, axis=-1) for batch_logits in logits for logit in batch_logits]
        else:
            predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[self.label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = self.seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    def get_loader(self, dataset, batch_size, shuffle=False):
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        dataset = dataset.remove_columns(self.remove_columns)
        loader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=shuffle)
        return loader
