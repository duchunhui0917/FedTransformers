import json
import numpy as np
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import logging
import os
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
import h5py
import tqdm
import random
from torch.utils.data import DataLoader

logger = logging.getLogger(os.path.basename(__file__))


class Sequence2SequenceDataset:
    def __init__(self, data_args, model_args):
        self.task_name = data_args.task_name
        self.dataset_name = data_args.dataset_name
        self.split = data_args.split
        self.dataset_path = data_args.dataset_path
        self.model_name = model_args.model_name
        self.vocab_file = model_args.vocab_file

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) if self.model_name != 'LSTM' else None
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name) if self.model_name != 'LSTM' else None

        self.max_seq_length = data_args.max_seq_length
        self.max_target_length = data_args.decoder_max_length

        logger.info('loading metric')
        self.eval_metric = 'rouge1'
        self.sacrebleu = load_metric('sacrebleu')
        self.rouge = load_metric('rouge')
        # self.perplexity = load_metric('perplexity')
        self.num_labels = 1
        self.unique_docs = [0]

        logger.info('loading dataset')
        if self.dataset_path:
            with h5py.File(self.dataset_path, 'r+') as df:
                attributes = json.loads(df["attributes"][()])

                index_list = attributes['index_list']
                train_idx = attributes['train_index_list']
                test_idx = attributes['test_index_list']
                eval_idx = attributes['eval_index_list'] if 'eval_index_list' in attributes else test_idx

                if 'doc_index' in attributes:
                    doc_index = attributes['doc_index']
                else:
                    doc_index = {str(i): 0 for i in index_list}

                unique_docs = set()
                train_dict, eval_dict, test_dict = ({'text': [], 'label': [], 'doc': []} for _ in range(3))
                t = tqdm.tqdm(train_idx)
                for idx in t:
                    text = df['X'][str(idx)][()].decode('UTF-8')
                    train_dict['text'].append(text)
                    label = df['Y'][str(idx)][()].decode('UTF-8')
                    train_dict['label'].append(label)
                    doc = doc_index[str(idx)]
                    train_dict['doc'].append(doc)
                    unique_docs.add(doc)
                t = tqdm.tqdm(eval_idx)
                for idx in t:
                    text = df['X'][str(idx)][()].decode('UTF-8')
                    eval_dict['text'].append(text)
                    label = df['Y'][str(idx)][()].decode('UTF-8')
                    eval_dict['label'].append(label)
                    doc = doc_index[str(idx)]
                    eval_dict['doc'].append(doc)
                    unique_docs.add(doc)
                t = tqdm.tqdm(test_idx)
                for idx in t:
                    text = df['X'][str(idx)][()].decode('UTF-8')
                    test_dict['text'].append(text)
                    label = df['Y'][str(idx)][()].decode('UTF-8')
                    test_dict['label'].append(label)
                    doc = doc_index[str(idx)]
                    test_dict['doc'].append(doc)
                    unique_docs.add(doc)
            self.unique_docs = unique_docs
            self.num_docs = len(self.unique_docs)

            train_dataset = Dataset.from_dict(train_dict)
            eval_dataset = Dataset.from_dict(eval_dict)
            test_dataset = Dataset.from_dict(test_dict)
        else:
            raw_datasets = load_dataset(self.task_name, self.dataset_name, split=self.split)

            train_dataset = raw_datasets["train"]
            eval_dataset = raw_datasets["validation"]
            test_dataset = raw_datasets["test"]

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
        if self.model_name == 'LSTM':
            pass
        else:
            if self.task_name == 'opus_books':
                source_lang, target_lang = self.dataset_name.split('-')
                inputs = [example[source_lang] for example in examples['translation']]
                targets = [example[target_lang] for example in examples['translation']]
            elif self.task_name == 'cnn_dailymail':
                inputs = examples['article']
                targets = examples['highlights']
            else:
                inputs = [example for example in examples['text']]
                targets = [example for example in examples['label']]

            model_inputs = self.tokenizer(inputs, max_length=self.max_seq_length, truncation=True, padding=True)
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=self.max_target_length, truncation=True, padding=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

    def compute_metrics(self, labels, logits):
        pad_token_id = self.tokenizer.pad_token_id
        labels, logits = labels[0], logits[0]
        # computing predict labels
        if not isinstance(logits, list):
            training = True
            pred_labels = np.argmax(logits, axis=-1)
        else:
            training = False
            ml = labels.shape[1]

            pred_labels = logits
            for i, x in enumerate(pred_labels):
                b, l = x.shape[0], x.shape[1]
                pad = np.ones((b, ml - l), dtype=np.int) * pad_token_id
                pred_labels[i] = np.concatenate([x, pad], axis=1)
            pred_labels = np.concatenate(pred_labels)

        decoded_pred_labels = self.tokenizer.batch_decode(pred_labels, skip_special_tokens=True)
        decoded_pred_labels = [pred.strip() for pred in decoded_pred_labels]

        res = {}
        # computing labels
        labels = np.where(labels != -100, labels, pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [label.strip() for label in decoded_labels]

        # computing rouge metric
        rouge_results = self.rouge.compute(predictions=decoded_pred_labels,
                                           references=decoded_labels)
        res.update({
            "rouge1": rouge_results["rouge1"].mid.fmeasure,
            "rouge2": rouge_results["rouge2"].mid.fmeasure,
            "rougeL": rouge_results["rougeL"].mid.fmeasure,
        })
        # computing bleu metric
        bleu_results = self.sacrebleu.compute(predictions=decoded_pred_labels,
                                              references=[[label] for label in decoded_labels])
        res.update({
            "bleu": bleu_results["score"],
        })
        # computing perplexity
        # if not training:
        #     idxes = random.sample(range(len(decoded_labels)), k=5)
        #     for idx in idxes:
        #         print('label:', decoded_labels[idx])
        #         print('pred:', decoded_pred_labels[idx])
        #
        #     decoded_pred_labels = [s.strip() for s in decoded_pred_labels if s != '']
        #     perplexity_results = self.perplexity.compute(model_id='gpt2', input_texts=decoded_pred_labels)
        #     res.update(
        #         {"perplexity": perplexity_results["mean_perplexity"]}
        #     )
        return res

    def get_loader(self, dataset, batch_size, shuffle=False):
        if len(dataset) == 0:
            return None
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        dataset = dataset.remove_columns(self.remove_columns)
        loader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=shuffle)
        return loader
