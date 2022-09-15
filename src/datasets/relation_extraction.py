import copy
import json
import numpy as np
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import torch
import logging
import os
from transformers import AutoTokenizer, DataCollatorWithPadding
import h5py
from tqdm import tqdm
import random
from .utils import add_space_for_punctuation
from torch.utils.data import DataLoader

logger = logging.getLogger(os.path.basename(__file__))
base_dir = os.path.expanduser('~/FedTransformers')


def update_dict(idxes, doc_index, unique_docs, d, df, label_vocab):
    t = tqdm(idxes)
    for idx in t:
        if 'e_text' in df and 'label' in df:
            text = df['e_text'][str(idx)][()].decode('UTF-8')
            label = df['label'][str(idx)][()].decode('UTF-8')
        else:
            text = df['X'][str(idx)][()].decode('UTF-8')
            label = df['Y'][str(idx)][()].decode('UTF-8')

        d['text'].append(text)
        d['label'].append(label_vocab[label])
        doc = doc_index[str(idx)]
        d['doc'].append(doc)
        unique_docs.add(doc)


def parse(texts, examples, tokenizer):
    examples.update({
        "input_ids": [],
        "attention_mask": [],
        "e1_mask": [],
        "e2_mask": [],
        "labels": examples["label"]
    })

    for text in texts:
        text_ls = text.split(' ')

        # add [CLS] token
        tokens = ["[CLS]"]
        e1_mask = [0]
        e2_mask = [0]
        e1_mask_val = 0
        e2_mask_val = 0
        for i, word in enumerate(text_ls):
            if word in ["<e1>", "</e1>", "<e2>", "</e2>"]:
                if word in ["<e1>"]:
                    e1_mask_val = 1
                elif word in ["</e1>"]:
                    e1_mask_val = 0
                if word in ["<e2>"]:
                    e2_mask_val = 1
                elif word in ["</e2>"]:
                    e2_mask_val = 0
                continue

            token = tokenizer.tokenize(word)

            tokens.extend(token)
            e1_mask.extend([e1_mask_val] * len(token))
            e2_mask.extend([e2_mask_val] * len(token))

        # add [SEP] token
        tokens.append("[SEP]")
        e1_mask.append(0)
        e2_mask.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        examples["input_ids"].append(input_ids)
        examples["attention_mask"].append(attention_mask)
        examples["e1_mask"].append(e1_mask)
        examples["e2_mask"].append(e2_mask)

    max_length = max([len(token) for token in examples["input_ids"]])
    # logger.info(f'max sequence length: {max_length}')
    ls = zip(examples["input_ids"], examples["attention_mask"],
             examples["e1_mask"], examples["e2_mask"])

    for i, (input_ids, attention_mask, e1_mask, e2_mask) in enumerate(ls):
        # zero-pad up to the sequence length
        padding = [0] * (max_length - len(input_ids))

        examples['input_ids'][i] = input_ids + padding
        examples['attention_mask'][i] = attention_mask + padding
        examples['e1_mask'][i] = e1_mask + padding
        examples['e2_mask'][i] = e2_mask + padding

        assert len(examples['input_ids'][i]) == max_length
        assert len(examples['attention_mask'][i]) == max_length
        assert len(examples['e1_mask'][i]) == max_length
        assert len(examples['e2_mask'][i]) == max_length

    return examples


class RelationExtractionDataset:
    def __init__(self, data_args, model_args):
        self.task_name = data_args.task_name
        self.dataset_name = data_args.dataset_name
        self.split = data_args.split
        self.dataset_path = data_args.dataset_path
        self.model_name = model_args.model_name
        self.vocab_file = model_args.vocab_file

        self.augment = model_args.augment

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
                update_dict(train_idx, doc_index, unique_docs, train_dict, df, self.label_vocab)
                update_dict(eval_idx, doc_index, unique_docs, eval_dict, df, self.label_vocab)
                update_dict(test_idx, doc_index, unique_docs, test_dict, df, self.label_vocab)

                self.unique_docs = unique_docs

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
        self.train_dataset = train_dataset.map(self.process_fn, batched=True, batch_size=len(train_dataset))

        logger.info('processing eval dataset')
        if data_args.max_eval_samples is not None:
            sampled_idxes = random.sample(range(len(eval_dataset)), k=data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(sampled_idxes)
        self.eval_dataset = eval_dataset.map(self.process_fn, batched=True, batch_size=len(eval_dataset))

        logger.info('processing test dataset')
        if data_args.max_test_samples is not None:
            sampled_idxes = random.sample(range(len(test_dataset)), k=data_args.max_test_samples)
            test_dataset = test_dataset.select(sampled_idxes)
        self.test_dataset = test_dataset.map(self.process_fn, batched=True, batch_size=len(test_dataset))

    def process_fn(self, examples):
        if self.model_name == 'LSTM':
            pass
        else:
            texts = examples['text']
            examples = parse(texts, examples, self.tokenizer)

            if self.augment == 'gradient_aug':
                path = os.path.join(base_dir, f'data/weight/{self.task_name}_s233_800.json')
                logger.info(path)

                with open(path, 'r') as f:
                    weight_dict = json.load(f)
                positive_mask_input_ids = copy.deepcopy(examples['input_ids'])
                negative_mask_input_ids = copy.deepcopy(examples['input_ids'])
                z = zip(examples['input_ids'], examples['attention_mask'], examples['e1_mask'], examples['e2_mask'])
                for i, (input_id, attention_mask, e1_mask, e2_mask) in enumerate(z):
                    k = sum(attention_mask) // 10
                    weight = torch.FloatTensor([weight_dict[str(x)] for x in input_id])
                    locs = torch.argsort(weight, descending=True)
                    locs = locs[:k]
                    for l in locs:
                        if e1_mask[l] != 1 and e2_mask[l] != 1:
                            negative_mask_input_ids[i][l] = self.tokenizer.mask_token_id

                    weight = torch.FloatTensor(
                        [weight_dict[str(x)] if x not in self.tokenizer.all_special_ids else float('inf')
                         for x in input_id]
                    )
                    locs = torch.argsort(weight)
                    locs = locs[:k]
                    for l in locs:
                        if e1_mask[l] != 1 and e2_mask[l] != 1:
                            positive_mask_input_ids[i][l] = self.tokenizer.mask_token_id
                examples['positive_mask_input_ids'] = positive_mask_input_ids
                examples['negative_mask_input_ids'] = negative_mask_input_ids

            return examples

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
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataset = dataset.remove_columns(self.remove_columns)
        loader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=shuffle)
        return loader
