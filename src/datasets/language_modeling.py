import json
import numpy as np
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import logging
import os
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import h5py
import tqdm
import random
from torch.utils.data import DataLoader

logger = logging.getLogger(os.path.basename(__file__))


class CasualLanguageModelingDataset:
    def __init__(self, data_args, model_args):
        self.task_name = data_args.task_name
        self.dataset_name = data_args.dataset_name
        self.split = data_args.split
        self.dataset_path = data_args.dataset_path
        self.max_seq_length = data_args.max_seq_length
        self.model_name = model_args.model_name
        self.vocab_file = model_args.vocab_file

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) if self.model_name != 'LSTM' else None
        self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info('loading metric')
        self.eval_metric = 'accuracy'
        self.accuracy = load_metric('accuracy')
        self.num_labels = 1

        logger.info('loading dataset')
        if self.dataset_path is not None:
            with h5py.File(self.dataset_path, 'r+') as f:
                attributes = json.loads(f['attributes'][()])
                users = attributes['users']
                train_texts = []
                train_docs = []
                test_texts = []
                test_docs = []
                self.unique_docs = list(range(len(users)))
                t = tqdm.tqdm(users)
                for doc, user in enumerate(t):
                    train_text = f['user_text'][user]['train'][()].decode('UTF-8')
                    train_texts.append(train_text)
                    train_docs.append(doc)

                    test_text = f['user_text'][user]['test'][()].decode('UTF-8')
                    test_texts.append(test_text)
                    test_docs.append(doc)

                train_dict = {'text': train_texts, 'doc': train_docs}
                test_dict = {'text': test_texts, 'doc': test_docs}
                eval_dict = test_dict.copy()

            train_dataset = Dataset.from_dict(train_dict)
            eval_dataset = Dataset.from_dict(eval_dict)
            test_dataset = Dataset.from_dict(test_dict)
        else:
            raw_datasets = load_dataset(self.task_name, self.dataset_name, split=self.split)

            if "test" not in raw_datasets:
                raw_datasets = raw_datasets.train_test_split(test_size=0.2)

            if self.task_name == 'eli5':
                raw_datasets = raw_datasets.flatten()

            train_dataset = raw_datasets["train"]
            eval_dataset = raw_datasets["validation"] if "validation" in raw_datasets else raw_datasets["test"]
            test_dataset = raw_datasets["test"]

        self.remove_columns = train_dataset.column_names
        logger.info('processing train dataset')
        self.train_dataset = train_dataset.map(self.process_fn, batched=True, remove_columns=self.remove_columns)
        if data_args.max_train_samples is not None:
            sampled_idxes = random.sample(range(len(self.train_dataset)), k=data_args.max_train_samples)
            self.train_dataset = self.train_dataset.select(sampled_idxes)

        logger.info('processing eval dataset')
        self.eval_dataset = eval_dataset.map(self.process_fn, batched=True, remove_columns=self.remove_columns)
        if data_args.max_eval_samples is not None:
            sampled_idxes = random.sample(range(len(self.eval_dataset)), k=data_args.max_eval_samples)
            self.eval_dataset = self.eval_dataset.select(sampled_idxes)

        logger.info('processing test dataset')
        self.test_dataset = test_dataset.map(self.process_fn, batched=True, remove_columns=self.remove_columns)
        if data_args.max_test_samples is not None:
            sampled_idxes = random.sample(range(len(self.test_dataset)), k=data_args.max_test_samples)
            self.test_dataset = self.test_dataset.select(sampled_idxes)

    def process_fn(self, examples):
        if self.model_name == 'LSTM':
            pass
        else:
            if self.task_name == 'eli5':
                text = examples["answers.text"]
                outputs = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_overflowing_tokens=True,
                    return_length=True,
                )
                input_batch = []
                for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                    if length == self.max_seq_length:
                        input_batch.append(input_ids)
                return {"input_ids": input_batch}
            else:
                text = examples["text"]
                outputs = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_overflowing_tokens=True,
                    return_length=True,
                )
                input_batch = []
                docs = []
                z = zip(outputs["length"], outputs["input_ids"], outputs["overflow_to_sample_mapping"])
                for length, input_ids, example_id in z:
                    if length == self.max_seq_length:
                        input_batch.append(input_ids)
                        docs.append(examples["doc"][example_id])
                return {"input_ids": input_batch, "doc": docs}

    def compute_metrics(self, labels, logits):
        labels, logits = labels[0], logits[0]

        shift_labels = labels[..., 1:]
        decode_shift_labels = self.tokenizer.batch_decode(shift_labels, skip_special_tokens=True)
        shift_labels = shift_labels.flatten()

        shift_logits = logits[..., :-1, :]
        pred_shift_labels = np.argmax(shift_logits, axis=-1)
        decode_pred_shift_labels = self.tokenizer.batch_decode(pred_shift_labels, skip_special_tokens=True)
        pred_shift_labels = pred_shift_labels.flatten()

        result = self.accuracy.compute(predictions=pred_shift_labels, references=shift_labels)

        if len(decode_shift_labels) == len(self.eval_dataset):
            idxes = random.sample(range(len(decode_shift_labels)), k=5)
            for idx in idxes:
                print('label:', decode_shift_labels[idx])
                print('pred:', decode_pred_shift_labels[idx])

        return result

    def get_loader(self, dataset, batch_size, shuffle=False):
        if len(dataset) == 0:
            return None
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        if "doc" in dataset.column_names:
            dataset = dataset.remove_columns(["doc"])
        loader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=shuffle)
        return loader
