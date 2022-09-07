import copy

from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    EvalPrediction,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import numpy as np
import logging
import os
import random
from torch.utils.data import DataLoader

base_dir = os.path.expanduser('~/FedTransformers')
logger = logging.getLogger(__name__)
langs = {
    'ner': ['en', 'de', 'es', 'nl'],
    'paws-x': ['en', 'de', 'es', 'fr'],
    'ntg': ['en', 'de', 'es', 'fr', 'ru'],
}


class XGlueDataset:
    def __init__(self, data_args, model_args):
        super().__init__()
        self.task_name = data_args.task_name
        self.dataset_name = data_args.dataset_name
        self.signature_columns = {'label'}
        self.data_args = data_args
        self.model_name = model_args.model_name

        logger.info('loading dataset')
        raw_datasets = load_dataset("xglue", self.dataset_name)
        logger.info('loading metric')
        if self.dataset_name in ["ner", "pos"]:
            self.seqeval = load_metric("seqeval")
        elif self.dataset_name in ['qg', 'ntg']:
            self.sacrebleu = load_metric('sacrebleu')
            self.rouge = load_metric('rouge')
            self.perplexity = load_metric('perplexity')
        else:
            self.accuracy = load_metric('accuracy')
            self.f1 = load_metric('f1')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.max_seq_length = data_args.max_seq_length
        self.max_target_length = data_args.max_target_length

        if self.dataset_name in ["ner", "pos"]:
            self.ner_feature = raw_datasets["train"].features["ner"]
            self.label_names = self.ner_feature.feature.names
            self.num_labels = len(self.label_names)
            self.id2label = {str(i): label for i, label in enumerate(self.label_names)}
            self.label2id = {v: k for k, v in self.id2label.items()}
            self.eval_metric = 'f1'
            logger.info(self.label2id)
        elif self.dataset_name in ["qg", "ntg", "mlqa"]:
            self.num_labels = 1
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.eval_metric = 'rouge1'
        else:
            self.label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
            self.eval_metric = 'f1'

        self.unique_docs = list(range(len(langs[self.dataset_name])))

        logger.info('processing train dataset')
        train_dict = {'doc': []}
        train_langs = [f'validation.{lang}' for lang in langs[self.dataset_name]]
        for i, train_lang in enumerate(train_langs):
            train_dataset = raw_datasets[train_lang]
            d = train_dataset.to_dict()
            for key, val in d.items():
                l = len(val)
                if key in train_dict:
                    train_dict[key] += val
                else:
                    train_dict.update({key: val})
            train_dict['doc'] += [i for _ in range(l)]
        train_dataset = Dataset.from_dict(train_dict)

        if data_args.max_train_samples is not None:
            sampled_idxes = random.sample(range(len(train_dataset)), k=data_args.max_train_samples)
            train_dataset = train_dataset.select(sampled_idxes)
        self.train_dataset = train_dataset.map(self.process_fn, batched=True, load_from_cache_file=False)

        logger.info('processing eval dataset')
        eval_dict = {'doc': []}
        eval_langs = [f'test.{lang}' for lang in langs[self.dataset_name]]
        for i, eval_lang in enumerate(eval_langs):
            eval_dataset = raw_datasets[eval_lang]
            d = eval_dataset.to_dict()
            for key, val in d.items():
                l = len(val)
                if key in eval_dict:
                    eval_dict[key] += val
                else:
                    eval_dict.update({key: val})
            eval_dict['doc'] += [i for _ in range(l)]
        eval_dataset = Dataset.from_dict(eval_dict)

        if data_args.max_eval_samples is not None:
            sampled_idxes = random.sample(range(len(eval_dataset)), k=data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(sampled_idxes)
        self.eval_dataset = eval_dataset.map(self.process_fn, batched=True, load_from_cache_file=False)

        logger.info('processing test dataset')
        test_dataset = copy.deepcopy(eval_dataset)
        if data_args.max_test_samples is not None:
            sampled_idxes = random.sample(range(len(test_dataset)), k=data_args.max_test_samples)
            test_dataset = test_dataset.select(sampled_idxes)
        self.test_dataset = test_dataset.map(self.process_fn, batched=True, load_from_cache_file=False)

    def process_fn(self, examples):
        if self.dataset_name in ["ner", 'pos']:
            text = examples["words"]
            tags = examples["ner"] if self.dataset_name == "ner" else examples["pos"]

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
            tokenized_inputs["label"] = tags

            self.signature_columns.update(tokenized_inputs.keys())
            self.signature_columns.remove('label')
            return tokenized_inputs
        if self.dataset_name == "mlqa":
            pass
        if self.dataset_name == "paws-x":
            args = (
                (examples["sentence1"], examples["sentence2"])
            )
            results = self.tokenizer(*args, padding=True, truncation=True, max_length=self.max_seq_length)
            self.signature_columns.update(results.keys())
            return results

        if self.dataset_name in ['qg', 'ntg']:
            inputs = examples['answer_passage'] if self.dataset_name == 'qg' else examples['news_body']
            targets = examples['question'] if self.dataset_name == 'ntg' else examples['news_title']

            model_inputs = self.tokenizer(inputs, max_length=self.max_seq_length, truncation=True, padding=True)
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=self.max_target_length, truncation=True, padding=True)

            model_inputs["labels"] = labels["input_ids"]
            self.signature_columns.update(model_inputs.keys())
            return model_inputs

    def compute_metrics(self, labels, logits):
        if self.dataset_name in ["ner", "pos"]:
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
        elif self.dataset_name in ['qg', 'ntg']:

            labels, logits = labels[0], logits[0]
            # computing predict labels
            if len(logits.shape) == 3:
                training = True
                pred_labels = np.argmax(logits, axis=-1)
            else:
                training = False
                pred_labels = logits
            decoded_pred_labels = self.tokenizer.batch_decode(pred_labels, skip_special_tokens=True)
            decoded_pred_labels = [pred.strip() for pred in decoded_pred_labels]

            res = {}
            # computing labels
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
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
            if not training:
                idxes = random.sample(range(len(decoded_labels)), k=5)
                for idx in idxes:
                    print('label:', decoded_labels[idx])
                    print('pred:', decoded_pred_labels[idx])

                decoded_pred_labels = [s.strip() for s in decoded_pred_labels if s != '']
                perplexity_results = self.perplexity.compute(model_id='gpt2', input_texts=decoded_pred_labels)
                res.update(
                    {"perplexity": perplexity_results["mean_perplexity"]}
                )
            return res
        else:

            labels, logits = labels[0], logits[0]
            pred_labels = np.argmax(logits, axis=-1)

            accuracy = self.accuracy.compute(predictions=pred_labels, references=labels)
            f1 = self.f1.compute(predictions=pred_labels, references=labels, average='macro')
            res = {'accuracy': accuracy['accuracy'], 'f1': f1['f1']}
            return res

    def get_loader(self, dataset, batch_size, shuffle=False):
        if len(dataset) == 0:
            return None
        if self.dataset_name in ['ner', 'pos']:
            data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        elif self.dataset_name in ['qg', 'ntg']:
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        else:
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        remove_columns = list(set(dataset.column_names) - self.signature_columns)
        dataset = dataset.remove_columns(remove_columns)

        loader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=shuffle)
        return loader
