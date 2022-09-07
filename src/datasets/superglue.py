from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM
)
from ..modules.verbalizers import (
    ManualVerbalizer,
    SoftVerbalizer,
    ProtoVerbalizer,
    AnchorVerbalizer,
)

import numpy as np
import logging
import json
from tqdm import tqdm
import os
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    mean_squared_error
import random
from collections import defaultdict
from torch.utils.data import DataLoader
from .utils import f1_score, exact_match_score, metric_max_over_ground_truths
from .multiple_choice import DataCollatorForMultipleChoice

base_dir = os.path.expanduser('~/FedTransformers')
logger = logging.getLogger(__name__)

VERBALIZER_MAP = {
    'manual': ManualVerbalizer,
    'soft': SoftVerbalizer,
    'proto': ProtoVerbalizer,
    'anchor': AnchorVerbalizer
}

task_to_keys = {
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "rte": ("premise", "hypothesis"),
    "wic": ("processed_sentence", None),
    "wsc": ("span2_word_text", "span1_text"),
    "copa": (None, None),
    "record": (None, None),
    "multirc": ("paragraph", "question_answer")
}


class SuperGlueDataset:
    def __init__(self, data_args, model_args):
        super().__init__()
        self.task_name = data_args.task_name
        self.dataset_name = data_args.dataset_name
        self.signature_columns = {'label'}
        self.data_args = data_args
        self.model_name = model_args.model_name

        self.vocab_file = model_args.vocab_file
        self.tunning_method = model_args.tunning_method
        self.prompt_method = model_args.prompt_method
        self.template_text = model_args.template_text
        self.anchor_texts = json.loads(model_args.anchor_texts) if model_args.anchor_texts else None
        self.label_words = json.loads(model_args.label_words) if model_args.label_words else None

        logger.info('loading dataset')
        raw_datasets = load_dataset("super_glue", self.dataset_name)
        logger.info('loading metric')
        self.metric = load_metric("super_glue", self.dataset_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        pad_token_id = self.tokenizer.pad_token_id
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id

        self.max_seq_length = data_args.max_seq_length

        if self.dataset_name == "record":
            self.num_labels = 2
            self.label_list = ["0", "1"]
        elif self.dataset_name == "copa":
            self.num_labels = 1
        else:
            self.label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[self.dataset_name]

        if self.num_labels != 1:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}
            logger.info(self.label2id)
            logger.info(self.id2label)

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

        remove_columns = raw_datasets["train"].column_names
        logger.info('processing train dataset')
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            sampled_idxes = random.sample(range(len(train_dataset)), k=data_args.max_train_samples)
            train_dataset = train_dataset.select(sampled_idxes)
        self.train_dataset = train_dataset.map(self.process_fn, batched=True, load_from_cache_file=False)

        logger.info('processing eval dataset')
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            sampled_idxes = random.sample(range(len(eval_dataset)), k=data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(sampled_idxes)
        self.eval_dataset = eval_dataset.map(self.process_fn, batched=True, load_from_cache_file=False)

        logger.info('processing test dataset')
        test_dataset = raw_datasets["test"]
        if data_args.max_test_samples is not None:
            sampled_idxes = random.sample(range(len(test_dataset)), k=data_args.max_test_samples)
            test_dataset = test_dataset.select(sampled_idxes)
        self.test_dataset = test_dataset.map(self.process_fn, batched=True, load_from_cache_file=False)

        self.eval_metric = "accuracy" if self.dataset_name not in ["record", "multirc"] else "f1"

    def process_fn(self, examples):
        # the winograd schema challenge
        if self.dataset_name == "wsc":
            examples["span2_word_text"] = []
            for text, span2_index, span2_word in zip(examples["text"], examples["span2_index"], examples["span2_text"]):
                if self.data_args.template_id == 0:
                    examples["span2_word_text"].append(span2_word + ": " + text)
                elif self.data_args.template_id == 1:
                    words_a = text.split()
                    words_a[span2_index] = "*" + words_a[span2_index] + "*"
                    examples["span2_word_text"].append(' '.join(words_a))

        # words in context
        if self.dataset_name == "wic":
            examples["processed_sentence"] = []
            z = zip(examples["sentence1"],
                    examples["sentence2"],
                    examples["word"],
                    examples["start1"],
                    examples["end1"],
                    examples["start2"],
                    examples["end2"])
            for sentence1, sentence2, word, start1, end1, start2, end2 in z:
                examples["processed_sentence"].append(
                    f"{sentence1} {sentence2} Does {word} have the same meaning in both sentences?"
                )

        # multi-sentence reading comprehension
        if self.dataset_name == "multirc":
            examples["question_answer"] = []
            for question, answer in zip(examples["question"], examples["answer"]):
                examples["question_answer"].append(f"{question} {answer}")

        # choice of plausible alternatives
        if self.dataset_name == "copa":
            examples["text_a"] = []
            for premise, question in zip(examples["premise"], examples["question"]):
                joiner = "because" if question == "cause" else "so"
                text_a = f"{premise} {joiner}"
                examples["text_a"].append(text_a)

            result1 = self.tokenizer(examples["text_a"], examples["choice1"], padding=True, truncation=True,
                                     max_length=self.max_seq_length)
            result2 = self.tokenizer(examples["text_a"], examples["choice2"], padding=True, truncation=True,
                                     max_length=self.max_seq_length)
            results = {}
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key in result1 and key in result2:
                    results[key] = []
                    for value1, value2 in zip(result1[key], result2[key]):
                        results[key].append([value1, value2])
            self.signature_columns.update({"input_ids", "attention_mask", "token_type_ids"})
            return results
        # reading comprehension with commonsense reasoning
        if self.dataset_name == 'record':
            results = {
                "passage": list(),
                "query": list(),
                "entities": list(),
                "answers": list(),
                "idx": list(),
                "question_id": list(),
                "entity": list(),
                "label": list(),
                "input_ids": list(),
                "attention_mask": list(),
                "token_type_ids": list(),
            }
            for idx, passage in enumerate(examples["passage"]):
                query, entities, answers = examples["query"][idx], examples["entities"][idx], examples["answers"][idx]
                index = examples["idx"][idx]
                passage = passage.replace("@highlight\n", "- ")

                for ent_idx, ent in enumerate(entities):
                    question = query.replace("@placeholder", ent)
                    result = self.tokenizer(passage, question, padding=True, truncation=True,
                                            max_length=self.max_seq_length)
                    label = 1 if ent in answers else 0

                    results["passage"].append(passage)
                    results["query"].append(query)
                    results["entities"].append(entities)
                    results["answers"].append(answers)
                    results["idx"].append(index)
                    results["question_id"].append(index["query"])
                    results["entity"].append(ent)
                    results["label"].append(label)

                    results["input_ids"].append(result["input_ids"])
                    results["attention_mask"].append(result["attention_mask"])
                    if "token_type_ids" in result:
                        results["token_type_ids"].append(result["token_type_ids"])
            self.signature_columns.update({"input_ids", "attention_mask", "token_type_ids"})
            return results

        args = (
            (examples[self.sentence1_key],) if self.sentence2_key is None else (
                examples[self.sentence1_key], examples[self.sentence2_key])
        )
        results = self.tokenizer(*args, padding=True, truncation=True, max_length=self.max_seq_length)
        self.signature_columns.update(results.keys())
        return results

    def compute_metrics(self, labels, logits):
        labels, logits = labels[0], logits[0]
        pred_labels = np.argmax(logits, axis=-1)

        if self.dataset_name == "record":
            return self.record_compute_metrics(logits)
        elif self.dataset_name == "multirc":
            from sklearn.metrics import f1_score
            return {"f1": f1_score(pred_labels, labels)}
        else:
            result = self.metric.compute(predictions=pred_labels, references=labels)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result

    def record_compute_metrics(self, logits):
        examples = self.eval_dataset
        qid2pred = defaultdict(list)
        qid2ans = {}
        for prob, example in zip(logits, examples):
            qid = example['question_id']
            qid2pred[qid].append((prob[1], example['entity']))
            if qid not in qid2ans:
                qid2ans[qid] = example['answers']
        n_correct, n_total = 0, 0
        f1, em = 0, 0
        for qid in qid2pred:
            preds = sorted(qid2pred[qid], reverse=True)
            entity = preds[0][1]
            n_total += 1
            n_correct += (entity in qid2ans[qid])
            f1 += metric_max_over_ground_truths(f1_score, entity, qid2ans[qid])
            em += metric_max_over_ground_truths(exact_match_score, entity, qid2ans[qid])
        acc = n_correct / n_total
        f1 = f1 / n_total
        em = em / n_total
        return {'acc': acc, 'f1': f1, 'exact_match': em}

    def get_loader(self, dataset, batch_size, shuffle=False):
        if len(dataset) == 0:
            return None
        if self.dataset_name == 'copa':
            data_collator = DataCollatorForMultipleChoice(tokenizer=self.tokenizer)
        else:
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        remove_columns = list(set(dataset.column_names) - set(self.signature_columns))
        dataset = dataset.remove_columns(remove_columns)

        loader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=shuffle)
        return loader
