import json
import numpy as np
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import logging
import os
from transformers import AutoTokenizer
import h5py
from tqdm import tqdm

logger = logging.getLogger(os.path.basename(__file__))


class ImageClassificationDataset(Dataset):
    def __init__(self, data_args, model_args):
        super().__init__()
        self.dataset_name = data_args.task_name
        self.dataset_path = data_args.dataset_path
        self.model_name = model_args.model_name
        self.vocab_file = model_args.vocab_file
        self.max_seq_length = model_args.max_seq_length

        if self.dataset_path != 'None':
            with h5py.File(self.dataset_name, 'r+') as df:
                attributes = json.loads(df["attributes"][()])

                label_vocab = attributes['label_vocab']
                index_list = attributes['index_list']
                train_idx = attributes['train_index_list']
                validation_idx = attributes['validation_index_list'] if 'validation_index_list' in attributes else []
                test_idx = attributes['test_index_list']
                task_type = attributes['task_type']
                if 'doc_index' in attributes:
                    doc_index = attributes['doc_index']
                else:
                    doc_index = {str(i): 0 for i in index_list}
                num_labels = attributes['num_labels']

                unique_docs = set()

                train_dict, validation_dict, test_dict = ({'text': [], 'label': [], 'doc': []} for _ in range(3))
                for idx in train_idx:
                    text = df['X'][str(idx)][()].decode('UTF-8')
                    train_dict['text'].append(text)
                    label = df['Y'][str(idx)][()].decode('UTF-8')
                    train_dict['label'].append(label_vocab[label])
                    doc = doc_index[str(idx)]
                    train_dict['doc'].append(doc)
                    unique_docs.add(doc)

            train_dataset = Dataset.from_dict(train_dict)
            eval_dataset = Dataset.from_dict(validation_dict)
            test_dataset = Dataset.from_dict(test_dict)

            self.train_dataset = train_dataset.map(self.token)
            self.eval_dataset = eval_dataset.map(self.token)
            self.test_dataset = test_dataset.map(self.token)
        else:
            raw_datasets = load_dataset(self.dataset_name)
            raw_datasets = raw_datasets.map(self.token)

            self.train_dataset = raw_datasets["train"]
            self.eval_dataset = raw_datasets["validation"]
            self.predict_dataset = raw_datasets["test"]

    def token(self, examples):
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

        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            inputs = tokenizer(examples, padding=True, truncation=True)
            max_seq_length = len(inputs['input_ids'][0])
            logger.info(f'max sequence length: {max_seq_length}')

        label = [x + [self.label_vocab['O']] * (max_seq_length - len(x)) for x in label]
        inputs.update({'label': label})
        return inputs
