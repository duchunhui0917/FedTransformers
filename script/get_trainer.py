import logging
import random
from enum import Enum
from collections import OrderedDict
from typing import Dict

from transformers import (
    AutoTokenizer,
    Trainer
)
from get_dataset import SuperGlueDataset, GlueDataset

from src.models.token_classification import (
    BertPrefixForTokenClassification,
    RobertaPrefixForTokenClassification,
)

from src.models.sequence_classification import (
    BertPrefixForSequenceClassification,
    BertPromptForSequenceClassification,
    RobertaPrefixForSequenceClassification,
    RobertaPromptForSequenceClassification,
)

from src.models.question_answering import (
    BertPrefixForQuestionAnswering,
    RobertaPrefixModelForQuestionAnswering,
)

from src.models.multiple_choice import (
    BertPrefixForMultipleChoice,
    BertPromptForMultipleChoice,
    RobertaPrefixForMultipleChoice,
    RobertaPromptForMultipleChoice
)

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice
)

logger = logging.getLogger(__name__)


class TaskType(Enum):
    TOKEN_CLASSIFICATION = 1,
    SEQUENCE_CLASSIFICATION = 2,
    QUESTION_ANSWERING = 3,
    MULTIPLE_CHOICE = 4


PREFIX_MODELS = {
    "bert": {
        TaskType.TOKEN_CLASSIFICATION: BertPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: BertPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: BertPrefixForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: BertPrefixForMultipleChoice
    },
    "roberta": {
        TaskType.TOKEN_CLASSIFICATION: RobertaPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: RobertaPrefixModelForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: RobertaPrefixForMultipleChoice,
    },
}

PROMPT_MODELS = {
    "bert": {
        TaskType.SEQUENCE_CLASSIFICATION: BertPromptForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: BertPromptForMultipleChoice
    },
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPromptForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: RobertaPromptForMultipleChoice
    }
}

AUTO_MODELS = {
    TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
    TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
    TaskType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice,
}


def get_model(model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False):
    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size

        model_class = PREFIX_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name,
            config=config,
            revision=model_args.model_revision,
        )
    elif model_args.prompt:
        config.pre_seq_len = model_args.pre_seq_len
        model_class = PROMPT_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name,
            config=config,
            revision=model_args.model_revision,
        )
    else:
        model_class = AUTO_MODELS[task_type]
        model = model_class.from_pretrained(
            model_args.model_name,
            config=config,
            revision=model_args.model_revision,
        )

        bert_param = 0
        if fix_bert:
            if config.model_type == "bert":
                for param in model.bert.parameters():
                    param.requires_grad = False
                for _, param in model.bert.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "deberta":
                for param in model.deberta.parameters():
                    param.requires_grad = False
                for _, param in model.deberta.named_parameters():
                    bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('***** total param is {} *****'.format(total_param))
    return model


class BaseTrainer(Trainer):
    def __init__(self, *args, predict_dataset=None, test_key="accuracy", **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })

    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, eval_metrics)

            if eval_metrics["eval_" + self.test_key] > self.best_metrics["best_eval_" + self.test_key]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_" + self.test_key] = eval_metrics["eval_" + self.test_key]

                if self.predict_dataset is not None:
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(dataset, metric_key_prefix="test")
                            self.best_metrics[f"best_test_{dataset_name}_{self.test_key}"] = test_metrics[
                                "test_" + self.test_key]
                    else:
                        _, _, test_metrics = self.predict(self.predict_dataset, metric_key_prefix="test")
                        self.best_metrics["best_test_" + self.test_key] = test_metrics["test_" + self.test_key]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


def get_superglue_trainer(args):
    model_args, data_args, training_args, _ = args

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    dataset = SuperGlueDataset(tokenizer, data_args, training_args)

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    if not dataset.multiple_choice:
        config = AutoConfig.from_pretrained(
            model_args.model_name,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.task_name,
            revision=model_args.model_revision,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.task_name,
            revision=model_args.model_revision,
        )

    if not dataset.multiple_choice:
        model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)
    else:
        model = get_model(model_args, TaskType.MULTIPLE_CHOICE, config, fix_bert=True)

    # Initialize our Trainer
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        test_key=dataset.test_key
    )

    return trainer, None


def get_glue_trainer(args):
    model_args, data_args, training_args, _ = args

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    dataset = GlueDataset(tokenizer, data_args, training_args)

    if not dataset.is_regression:
        config = AutoConfig.from_pretrained(
            model_args.model_name,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.task_name,
            revision=model_args.model_revision,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.task_name,
            revision=model_args.model_revision,
        )

    model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)

    # Initialize our Trainer
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
    )

    return trainer, None


def get_ner_trainer(args):
    model_args, data_args, training_args, qa_args = args

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    model_type = AutoConfig.from_pretrained(model_args.model_name).model_type

    add_prefix_space = ADD_PREFIX_SPACE[model_type]

    use_fast = USE_FAST[model_type]

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        use_fast=use_fast,
        revision=model_args.model_revision,
        add_prefix_space=add_prefix_space,
    )

    dataset = NERDataset(tokenizer, data_args, training_args)

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    if data_args.task_name == "conll2003":
        config = AutoConfig.from_pretrained(
            model_args.model_name,
            num_labels=dataset.num_labels,
            label2id=dataset.label_to_id,
            id2label={i: l for l, i in dataset.label_to_id.items()},
            revision=model_args.model_revision,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name,
            num_labels=dataset.num_labels,
            label2id=dataset.label_to_id,
            id2label={i: l for l, i in dataset.label_to_id.items()},
            revision=model_args.model_revision,
        )

    model = get_model(model_args, TaskType.TOKEN_CLASSIFICATION, config, fix_bert=True)

    trainer = ExponentialTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        predict_dataset=dataset.test_dataset if training_args.do_predict else None,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        compute_metrics=dataset.compute_metrics,
        test_key="f1"
    )
    return trainer, dataset.test_dataset


def get_qa_trainer(args):
    model_args, data_args, training_args, qa_args = args

    config = AutoConfig.from_pretrained(
        model_args.model_name,
        num_labels=2,
        revision=model_args.model_revision,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        revision=model_args.model_revision,
        use_fast=True,
    )

    model = get_model(model_args, TaskType.QUESTION_ANSWERING, config, fix_bert=True)

    dataset = SQuAD(tokenizer, data_args, training_args, qa_args)

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        eval_examples=dataset.eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        post_process_function=dataset.post_processing_function,
        compute_metrics=dataset.compute_metrics,
    )

    return trainer, dataset.test_dataset


def get_srl_trainer(args):
    model_args, data_args, training_args, _ = args

    model_type = AutoConfig.from_pretrained(model_args.model_name).model_type

    add_prefix_space = ADD_PREFIX_SPACE[model_type]

    use_fast = USE_FAST[model_type]

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        use_fast=use_fast,
        revision=model_args.model_revision,
        add_prefix_space=add_prefix_space,
    )

    dataset = SRLDataset(tokenizer, data_args, training_args)

    config = AutoConfig.from_pretrained(
        model_args.model_name,
        num_labels=dataset.num_labels,
        revision=model_args.model_revision,
    )

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    model = get_model(model_args, TaskType.TOKEN_CLASSIFICATION, config, fix_bert=False)

    trainer = ExponentialTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        predict_dataset=dataset.test_dataset if training_args.do_predict else None,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        compute_metrics=dataset.compute_metrics,
        test_key="f1"
    )

    return trainer, dataset.test_dataset
