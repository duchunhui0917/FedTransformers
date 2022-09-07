import json
import numpy as np
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import torch
import logging
import os
from transformers import AutoTokenizer, DefaultDataCollator
import h5py
from tqdm import tqdm
import random
from .utils import add_space_for_punctuation
from torch.utils.data import DataLoader
import collections

logger = logging.getLogger(os.path.basename(__file__))


class QuestionAnsweringDataset:
    def __init__(self, data_args, model_args):
        self.task_name = data_args.task_name
        self.dataset_name = data_args.dataset_name
        self.split = data_args.split
        self.dataset_path = data_args.dataset_path
        self.model_name = model_args.model_name
        self.vocab_file = model_args.vocab_file

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) if self.model_name != 'LSTM' else None
        self.max_seq_length = data_args.max_seq_length
        self.max_answer_length = data_args.max_target_length

        self.stride = data_args.stride
        self.n_best = data_args.n_best

        logger.info('loading metric')
        self.eval_metric = 'f1'
        self.squad = load_metric('squad')
        self.num_labels = 1

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
            self.raw_datasets = DatasetDict({"train": train_dataset,
                                             "validation": eval_dataset,
                                             "test": test_dataset})
        else:
            self.raw_datasets = load_dataset(self.task_name, self.dataset_name, split=self.split)

            if self.task_name == 'squad':
                train_dataset = self.raw_datasets["train"]
                eval_dataset = self.raw_datasets["validation"]
                test_dataset = self.raw_datasets["validation"]
            else:
                train_dataset = self.raw_datasets["train"]
                eval_dataset = self.raw_datasets["validation"]
                test_dataset = self.raw_datasets["test"]

        logger.info('processing train dataset')
        if data_args.max_train_samples is not None:
            sampled_idxes = random.sample(range(len(train_dataset)), k=data_args.max_train_samples)
            train_dataset = train_dataset.select(sampled_idxes)
        self.train_dataset = train_dataset.map(self.process_training_fn,
                                               batched=True,
                                               remove_columns=self.raw_datasets['train'].column_names,
                                               load_from_cache_file=False
                                               )

        logger.info('processing eval dataset')
        if data_args.max_eval_samples is not None:
            sampled_idxes = random.sample(range(len(eval_dataset)), k=data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(sampled_idxes)
        self.eval_dataset = eval_dataset.map(self.preprocess_validation_fn,
                                             batched=True,
                                             remove_columns=self.raw_datasets['validation'].column_names,
                                             load_from_cache_file=False)

        logger.info('processing test dataset')
        if data_args.max_test_samples is not None:
            sampled_idxes = random.sample(range(len(test_dataset)), k=data_args.max_test_samples)
            test_dataset = test_dataset.select(sampled_idxes)
        self.test_dataset = test_dataset.map(self.preprocess_validation_fn,
                                             batched=True,
                                             remove_columns=self.raw_datasets['validation'].column_names,
                                             load_from_cache_file=False)

    def process_training_fn(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_seq_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_fn(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_seq_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    def compute_metrics(self, labels, logits):
        start_logits, end_logits = logits[0], logits[1]

        features = self.eval_dataset
        examples = self.raw_datasets["validation"]
        if len(start_logits) != len(features.data):
            return {'f1': 0}

        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1: -self.n_best - 1: -1].tolist()
                end_indexes = np.argsort(end_logit)[-1: -self.n_best - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                                end_index < start_index
                                or end_index - start_index + 1 > self.max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        results = self.squad.compute(predictions=predicted_answers, references=theoretical_answers)
        return results

    def get_loader(self, dataset, batch_size, shuffle=False):
        if len(dataset) == 0:
            return None
        data_collator = DefaultDataCollator()
        if 'example_id' in dataset.column_names:
            dataset = dataset.remove_columns(['example_id', 'offset_mapping'])
        loader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=shuffle)
        return loader
