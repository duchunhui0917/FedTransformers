import copy
import json
import numpy as np
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import torch
import logging
import os
from transformers import (
    AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, AutoModelForMaskedLM
)
from ..modules.augment import sst2_augment, sentiment_augment
from ..modules.verbalizers import (
    ManualVerbalizer,
    SoftVerbalizer,
    ProtoVerbalizer,
    AnchorVerbalizer,
)
from ..modules.templates import Template
from ..modules.common_modules import DataCollatorForAnchor
import h5py
from tqdm import tqdm
import random
from .utils import add_space_for_punctuation, sst_label_fn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

base_dir = os.path.expanduser('~/FedTransformers')
logger = logging.getLogger(os.path.basename(__file__))

VERBALIZER_MAP = {
    'manual': ManualVerbalizer,
    'soft': SoftVerbalizer,
    'proto': ProtoVerbalizer,
    'anchor': AnchorVerbalizer
}


class SequenceClassificationDataset:
    def __init__(self, data_args, model_args):
        self.augment = model_args.augment
        self.tokenized_anchor_texts = None
        self.task_name = data_args.task_name
        self.dataset_name = data_args.dataset_name

        self.split = data_args.split
        self.dataset_path = data_args.dataset_path
        self.model_type = model_args.model_type
        self.model_name = model_args.model_name

        self.vocab_file = model_args.vocab_file
        self.tunning_method = model_args.tunning_method
        self.prompt_method = model_args.prompt_method
        self.template_text = model_args.template_text
        self.anchor_texts = json.loads(model_args.anchor_texts) if model_args.anchor_texts else None
        self.label_words = json.loads(model_args.label_words) if model_args.label_words else None

        self.max_seq_length = data_args.max_seq_length
        self.decoder_max_length = data_args.decoder_max_length

        self.max_train_samples = data_args.max_train_samples
        self.max_eval_samples = data_args.max_eval_samples
        self.max_test_samples = data_args.max_test_samples

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        pad_token_id = self.tokenizer.pad_token_id
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id

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
            if '*' in self.task_name:
                self.prompt = False
                self.verbalizer = None
                self.process()
                self.remove_columns = ['text', 'label', 'doc']
                return

            raw_datasets = load_dataset(self.task_name, self.dataset_name, split=self.split)
            if self.task_name == 'web_of_science':
                raw_datasets = raw_datasets["train"].train_test_split(test_size=0.2)
                train_dataset = raw_datasets["train"]
                eval_dataset = raw_datasets["test"]
                test_dataset = raw_datasets["test"]
                label_list = set(train_dataset['label'])
                self.num_labels = len(label_list)
                self.label_vocab = {l: i for i, l in enumerate(label_list)}
            elif self.task_name == 'sst5':
                train_dataset = raw_datasets["train"]
                eval_dataset = raw_datasets["validation"]
                test_dataset = raw_datasets["test"]

                label_list = [0, 1, 2, 3, 4]
                self.num_labels = len(label_list)
                self.label_vocab = {l: i for i, l in enumerate(label_list)}
            elif self.task_name == 'sst2':
                raw_datasets.rename_column_('sentence', 'text')
                train_dataset = raw_datasets["train"]
                eval_dataset = raw_datasets["validation"]
                test_dataset = raw_datasets["validation"]

                label_list = [0, 1]
                self.num_labels = len(label_list)
                self.label_vocab = {l: i for i, l in enumerate(label_list)}
            elif self.task_name == 'imdb':
                train_dataset = raw_datasets["train"]
                eval_dataset = raw_datasets["test"]
                test_dataset = raw_datasets["test"]

                label_list = [0, 1]
                self.num_labels = len(label_list)
                self.label_vocab = {l: i for i, l in enumerate(label_list)}
            elif self.task_name == 'rotten_tomatoes':
                train_dataset = raw_datasets["train"]
                eval_dataset = raw_datasets["validation"]
                test_dataset = raw_datasets["test"]

                label_list = [0, 1]
                self.num_labels = len(label_list)
                self.label_vocab = {l: i for i, l in enumerate(label_list)}
            elif self.task_name == 'yelp_polarity':
                train_dataset = raw_datasets["train"]
                eval_dataset = raw_datasets["test"]
                test_dataset = raw_datasets["test"]

                label_list = [0, 1]
                self.num_labels = len(label_list)
                self.label_vocab = {l: i for i, l in enumerate(label_list)}

            elif self.dataset_name == 'qnli':
                train_dataset = raw_datasets["train"]
                eval_dataset = raw_datasets["validation"]
                test_dataset = raw_datasets["test"]

                label_list = [0, 1]
                self.num_labels = len(label_list)
                self.label_vocab = {l: i for i, l in enumerate(label_list)}
            elif self.dataset_name == 'mrpc':
                train_dataset = raw_datasets["train"]
                eval_dataset = raw_datasets["validation"]
                test_dataset = raw_datasets["test"]

                label_list = [0, 1]
                self.num_labels = len(label_list)
                self.label_vocab = {l: i for i, l in enumerate(label_list)}
            else:
                label_list = raw_datasets['train'].features['label'].names
                self.num_labels = len(label_list)
                self.label_vocab = {l: i for i, l in enumerate(label_list)}

                train_dataset = raw_datasets["train"]
                eval_dataset = raw_datasets["validation"]
                test_dataset = raw_datasets["test"]

        if self.prompt_method in VERBALIZER_MAP:
            inputs = [label_word[0] for label_word in self.label_words]
            tokenized_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
            outputs = self.model(**tokenized_inputs, output_hidden_states=True)
            input_ids = tokenized_inputs['input_ids']
            h = outputs.hidden_states[0]
            label_vectors = torch.empty((h.shape[0], h.shape[2]))
            for i in range(len(label_vectors)):
                label_vectors[i] = h[i][
                    (input_ids[i] != pad_token_id) & (input_ids[i] != cls_token_id) & (
                            input_ids[i] != sep_token_id)].mean(dim=0)
            self.prompt = True
            self.verbalizer = VERBALIZER_MAP[self.prompt_method](
                model=AutoModelForMaskedLM.from_pretrained(self.model_name),
                tokenizer=self.tokenizer,
                label_words=self.label_words,
                num_classes=self.num_labels,
                label_vectors=label_vectors
            )
        else:
            self.prompt = False
            self.verbalizer = None

        self.remove_columns = train_dataset.column_names
        logger.info('processing train dataset')
        if self.max_train_samples is not None:
            sampled_idxes = random.sample(range(len(train_dataset)), k=self.max_train_samples)
            train_dataset = train_dataset.select(sampled_idxes)
            self.train_sampled_idxes = sampled_idxes
            self.train_samples = data_args.max_train_samples
        else:
            self.train_samples = len(train_dataset)
            self.train_sampled_idxes = None

        self.train_dataset = train_dataset.map(self.process_fn, batched=True, load_from_cache_file=False)

        logger.info('processing eval dataset')
        if self.max_eval_samples is not None:
            sampled_idxes = random.sample(range(len(eval_dataset)), k=self.max_eval_samples)
            eval_dataset = eval_dataset.select(sampled_idxes)
            self.eval_samples = self.max_eval_samples
        else:
            self.eval_samples = len(eval_dataset)

        self.eval_dataset = eval_dataset.map(self.process_fn, batched=True, load_from_cache_file=False)

        logger.info('processing test dataset')
        if self.max_test_samples is not None:
            sampled_idxes = random.sample(range(len(test_dataset)), k=self.max_test_samples)
            test_dataset = test_dataset.select(sampled_idxes)
            self.test_samples = self.max_test_samples
        else:
            self.test_samples = len(test_dataset)

        self.test_dataset = test_dataset.map(self.process_fn, batched=True, load_from_cache_file=False)

    def process_fn(self, examples):
        label = examples['label']
        if self.task_name == 'sst5':
            text = examples['sentence']
            ls = [{'text_a': t} for t in text]
            label = sst_label_fn(label)
        elif self.dataset_name == 'qnli':
            question = examples['question']
            sentence = examples['sentence']
            text = [q + ' [SEP] ' + s for q, s in zip(question, sentence)]
            ls = [{'text_a': q,
                   'text_b': s} for q, s in zip(question, sentence)]
        elif self.dataset_name == 'mrpc':
            sentence1 = examples['sentence1']
            sentence2 = examples['sentence2']
            text = [s1 + ' [SEP] ' + s2 for s1, s2 in zip(sentence1, sentence2)]
            ls = [{'text_a': s1,
                   'text_b': s2} for s1, s2 in zip(sentence1, sentence2)]
        else:
            text = examples['text']
            ls = [{'text_a': t} for t in text]

        if self.model_name == 'LSTM':
            logger.info(f'max sequence length: {self.max_seq_length}')
            with open(self.vocab_file, 'r') as f:
                vocab = json.load(f)
            embeddings = []
            t = tqdm(text)

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

            for i, embedding in enumerate(embeddings):
                if len(embedding) > self.max_seq_length:
                    embeddings[i] = embedding[:self.max_seq_length]
                else:
                    embeddings[i] += [vocab['[PAD]']] * (self.max_seq_length - len(embedding))
            return {'embeddings': embeddings}

        else:
            if self.prompt:
                text = Template(ls, self.template_text).text
                if self.anchor_texts:
                    template_anchor_texts = Template(self.anchor_texts, self.template_text).text
                    tokenized_anchor_texts = self.tokenizer(template_anchor_texts,
                                                            truncation=True,
                                                            padding=True,
                                                            max_length=self.max_seq_length,
                                                            return_tensors='pt')
                    self.tokenized_anchor_texts = tokenized_anchor_texts
                    # keys = tokenized_anchor_texts.keys()
                    # num = self.num_labels
                    # self.tokenized_anchor_texts = [
                    #     {key: tokenized_anchor_texts[key][i] for key in keys} for i in range(num)
                    # ]

            inputs = self.tokenizer(text, truncation=True, padding=True, max_length=self.max_seq_length)
            inputs['labels'] = label
            if self.task_name == 'sst':
                inputs['label'] = label

            if self.augment == 'manual_aug':
                positive_text, positive_mask_text, negative_text, negative_mask_text = sentiment_augment(
                    self.train_sampled_idxes)

                assert len(positive_text) == len(positive_mask_text) == len(negative_text) == len(negative_mask_text)
                positive_inputs = self.tokenizer(positive_text, truncation=True, padding=True,
                                                 max_length=self.max_seq_length)
                positive_input_ids = positive_inputs['input_ids']

                positive_mask_inputs = self.tokenizer(positive_mask_text, truncation=True, padding=True,
                                                      max_length=self.max_seq_length)
                positive_mask_input_ids = positive_mask_inputs['input_ids']

                negative_inputs = self.tokenizer(negative_text, truncation=True, padding=True,
                                                 max_length=self.max_seq_length)
                negative_input_ids = negative_inputs['input_ids']

                negative_mask_inputs = self.tokenizer(negative_mask_text, truncation=True, padding=True,
                                                      max_length=self.max_seq_length)
                negative_mask_input_ids = negative_mask_inputs['input_ids']

                inputs['positive_input_ids'] = positive_input_ids
                inputs['positive_mask_input_ids'] = positive_mask_input_ids
                inputs['negative_input_ids'] = negative_input_ids
                inputs['negative_mask_input_ids'] = negative_mask_input_ids
            elif self.augment == 'gradient_aug':
                with open(os.path.join(base_dir, f'data/weight/{self.task_name}_s233_800.json'), 'r') as f:
                    weight_dict = json.load(f)
                positive_mask_input_ids = copy.deepcopy(inputs['input_ids'])
                negative_mask_input_ids = copy.deepcopy(inputs['input_ids'])
                for i, (input_id, attention_mask) in enumerate(zip(inputs['input_ids'], inputs['attention_mask'])):
                    k = sum(attention_mask) // 10
                    weight = torch.FloatTensor([weight_dict[str(x)] for x in input_id])
                    locs = torch.argsort(weight, descending=True)
                    locs = locs[:k]
                    for l in locs:
                        negative_mask_input_ids[i][l] = self.tokenizer.mask_token_id

                    weight = torch.FloatTensor(
                        [weight_dict[str(x)] if x not in self.tokenizer.all_special_ids else float('inf')
                         for x in input_id]
                    )
                    locs = torch.argsort(weight)
                    locs = locs[:k]
                    for l in locs:
                        positive_mask_input_ids[i][l] = self.tokenizer.mask_token_id
                inputs['positive_mask_input_ids'] = positive_mask_input_ids
                inputs['negative_mask_input_ids'] = negative_mask_input_ids

            return inputs

    def compute_metrics(self, labels, logits):
        labels, logits = labels[0], logits[0]
        pred_labels = np.argmax(logits, axis=-1)

        labels = labels[:len(pred_labels)]

        accuracy = self.accuracy.compute(predictions=pred_labels, references=labels)
        f1 = self.f1.compute(predictions=pred_labels, references=labels, average='macro')
        res = {'accuracy': accuracy['accuracy'], 'f1': f1['f1']}
        if len(labels) == self.train_samples or len(labels) == self.eval_samples or len(labels) == self.test_samples:
            cf = confusion_matrix(y_true=labels, y_pred=pred_labels)
            logger.info('confusion matrix')
            logger.info('\n' + '\n'.join([str(_) for _ in cf]))

            logits_exp = np.exp(logits)
            logits_sum = np.sum(logits_exp, axis=1, keepdims=True)
            logits_exp /= logits_sum
            logger.info('logits histogram')
            s = '\n'
            for i in range(self.num_labels):
                ls = logits_exp[labels == i, i]
                ls = np.histogram(ls, bins=np.arange(0, 1.1, 0.1))[0]
                s += str(ls) + '\n'
            logger.info(s)

        return res

    def get_loader(self, dataset, batch_size, shuffle=False):
        if len(dataset) == 0:
            return None

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataset = dataset.remove_columns(self.remove_columns)
        loader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=shuffle)
        return loader

    def process(self):
        task_names = self.task_name.split('*')

        logger.info('processing train dataset')
        train_datasets = {'text': [], 'label': [], 'doc': []}
        for i, task_name in enumerate(task_names):
            raw_datasets = load_dataset(task_name)
            train_dataset = raw_datasets['train']
            if self.max_train_samples is not None:
                sampled_idxes = random.sample(range(len(train_dataset)), k=self.max_train_samples)
                train_dataset = train_dataset.select(sampled_idxes)

            if task_name == 'sst2':
                train_datasets['text'].extend(train_dataset['sentence'])
            else:
                train_datasets['text'].extend(train_dataset['text'])
            train_datasets['label'].extend(train_dataset['label'])
            train_datasets['doc'].extend([i] * len(train_dataset))
        self.train_dataset = Dataset.from_dict(train_datasets).map(
            self.process_fn, batched=True, batch_size=len(train_datasets['text']), load_from_cache_file=False)

        logger.info('processing eval dataset')
        eval_datasets = {'text': [], 'label': [], 'doc': []}
        for i, task_name in enumerate(task_names):
            raw_datasets = load_dataset(task_name)
            if task_name == 'imdb' or task_name == 'yelp_polarity':
                eval_dataset = raw_datasets['test']
            else:
                eval_dataset = raw_datasets['validation']

            if self.max_eval_samples is not None:
                sampled_idxes = random.sample(range(len(eval_dataset)), k=self.max_eval_samples)
                eval_dataset = eval_dataset.select(sampled_idxes)

            if task_name == 'sst2':
                eval_datasets['text'].extend(eval_dataset['sentence'])
            else:
                eval_datasets['text'].extend(eval_dataset['text'])
            eval_datasets['label'].extend(eval_dataset['label'])
            eval_datasets['doc'].extend([i] * len(eval_dataset))

        self.eval_dataset = Dataset.from_dict(eval_datasets).map(
            self.process_fn, batched=True, batch_size=len(eval_datasets['text']), load_from_cache_file=False)

        logger.info('processing test dataset')
        test_datasets = {'text': [], 'label': [], 'doc': []}
        for i, task_name in enumerate(task_names):
            raw_datasets = load_dataset(task_name)
            test_dataset = raw_datasets['test']

            if self.max_test_samples is not None:
                sampled_idxes = random.sample(range(len(test_dataset)), k=self.max_test_samples)
                test_dataset = test_dataset.select(sampled_idxes)

            if task_name == 'sst2':
                test_datasets['text'].extend(test_dataset['sentence'])
            else:
                test_datasets['text'].extend(test_dataset['text'])
            test_datasets['label'].extend(test_dataset['label'])
            test_datasets['doc'].extend([i] * len(test_dataset))

        self.test_dataset = Dataset.from_dict(test_datasets).map(
            self.process_fn, batched=True, batch_size=len(test_datasets['text']), load_from_cache_file=False)

        self.train_samples = len(self.train_dataset)
        self.eval_samples = len(self.eval_dataset)
        self.test_samples = len(self.test_dataset)
        label_list = [0, 1]
        self.num_labels = len(label_list)
        self.label_vocab = {l: i for i, l in enumerate(label_list)}
        self.unique_docs = set([i for i in range(len(task_names))])
