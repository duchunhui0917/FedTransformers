import logging
import os
import sys
import numpy as np
from typing import Dict
import datasets
import transformers
from transformers import set_seed, Trainer
from transformers import HfArgumentParser, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from src.arguments import (
    SUPERGLUE_DATASETS, GLUE_DATASETS, NER_DATASETS, SRL_DATASETS, QA_DATASETS,
    DataArguments, ModelArguments, QuestionAnsweringArguments
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)


def train(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    # trainer.save_model()

    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.log_best_metrics()


def evaluate(trainer):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def predict(trainer, predict_dataset=None):
    if predict_dataset is None:
        logger.info("No dataset is available for testing")

    elif isinstance(predict_dataset, dict):

        for dataset_name, d in predict_dataset.items():
            logger.info("*** Predict: %s ***" % dataset_name)
            predictions, labels, metrics = trainer.predict(d, metric_key_prefix="predict")
            predictions = np.argmax(predictions, axis=2)

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

    else:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == '__main__':
    config = [
        '--model_name=bert-base-uncased',
        '--task_name=superglue',
        '--task_name=rte',
        '--output_dir=""',
        '--do_train',
        '--do_eval',
        '--max_seq_length=128',
        '--per_device_train_batch_size=8',
        '--learning_rate=1e-2',
        '--num_train_epochs=60',
        '--pre_seq_len=20',
        '--overwrite_output_dir',
        '--hidden_dropout_prob=0.1',
        '--seed=11',
        '--save_strategy=no',
        '--evaluation_strategy=epoch',
        '--prefix'

    ]
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, QuestionAnsweringArguments))
    args = parser.parse_args_into_dataclasses(config)
    model_args, data_args, training_args, qa_args = args

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    if data_args.task_name.lower() == "superglue":
        assert data_args.task_name.lower() in SUPERGLUE_DATASETS
        from get_trainer import get_superglue_trainer as get_trainer

    elif data_args.task_name.lower() == "glue":
        assert data_args.task_name.lower() in GLUE_DATASETS
        from get_trainer import get_glue_trainer as get_trainer

    elif data_args.task_name.lower() == "ner":
        assert data_args.task_name.lower() in NER_DATASETS
        from get_trainer import get_ner_trainer as get_trainer

    elif data_args.task_name.lower() == "srl":
        assert data_args.task_name.lower() in SRL_DATASETS
        from get_trainer import get_srl_trainer as get_trainer

    elif data_args.task_name.lower() == "qa":
        assert data_args.task_name.lower() in QA_DATASETS
        from get_trainer import get_qa_trainer as get_trainer

    else:
        raise NotImplementedError(
            'Task {} is not implemented. Please choose a task from: {}'.format(data_args.task_name, ", ".join(TASKS)))

    set_seed(training_args.seed)

    trainer, predict_dataset = get_trainer(args)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if training_args.do_train:
        train(trainer, training_args.resume_from_checkpoint, last_checkpoint)

    if training_args.do_eval:
        evaluate(trainer)

    if training_args.do_predict:
        predict(trainer, predict_dataset)
