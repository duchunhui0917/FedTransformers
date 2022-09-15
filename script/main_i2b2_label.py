import os
import sys
import datetime
import warnings
import logging
import json

import wandb
from transformers import HfArgumentParser

sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from src.arguments import (
    DataArguments,
    ModelArguments,
    FederatedLearningArguments,
    WandbArguments,
    SYSTEMS)
from src.processor import process_dataset_model
from src import plot_class_samples, status_mtx, init_log, set_seed
from src.utils.plot_utils import plot_hist

base_dir = os.path.expanduser('~/FedTransformers')
warnings.filterwarnings('ignore')
logger = logging.getLogger(os.path.basename(__file__))

task_name = 'i2b2'
dataset_path = os.path.join(base_dir, f"data/i2b2_data.h5")
model_type = 'distilbert'
model_name = 'distilbert-base-uncased'
model_path = os.path.join(base_dir,
                          f"ckpt/centralized/i2b2_s223/{model_name}_tgwg")

config = [
    f'--task_name={task_name}',
    f'--dataset_path={dataset_path}',
    f'--model_name={model_name}',
    f'--model_type={model_type}',
    # f'--model_path={model_path}',
    '--lr=5e-5',
    '--algorithm=FedAvg',
    '--split_type=label_split',
    '--num_clients=10',
    '--dirichlet_alpha=0.05',
    '--train_batch_size=8',
    '--eval_batch_size=8',
    # '--max_train_samples=800',
    # '--max_eval_samples=800',
    # '--max_test_samples=800',
    '--seed=223',
    '--num_iterations=100',
    '--do_train=True',
    '--do_test=True',
    '--augment=prototype_aug',
    f'--enable_wandb=True',
    f'--project_name=FedTransformers_i2b2'
]

parser = HfArgumentParser((DataArguments, ModelArguments, FederatedLearningArguments, WandbArguments))
all_args = parser.parse_args(config)
data_args, model_args, fl_args, wandb_args = parser.parse_args_into_dataclasses(config)
if wandb_args.enable_wandb:
    wandb.init(
        config=all_args,
        project=wandb_args.project_name,
        entity=wandb_args.team_name,
    )

set_seed(fl_args.seed)

model_name = model_name.replace('/', '_')
if model_args.augment:
    model_name += f'_{model_args.augment}'
if model_args.tunning_method:
    model_name += f'_{model_args.tunning_method}'

if fl_args.split_type == 'label_split':
    ckpt = f'{model_name}_{fl_args.dirichlet_alpha}_{fl_args.num_clients}_{fl_args.num_epochs}'
elif fl_args.split_type == 'feature_split':
    ckpt = f'{model_name}_{fl_args.num_clients}_{fl_args.num_epochs}'
elif fl_args.split_type == 'doc_split':
    ckpt = f'{model_name}_{fl_args.num_epochs}'
else:
    ckpt = f'{model_name}'

system = SYSTEMS[fl_args.algorithm]

task_name += f'_s{fl_args.seed}'

if data_args.max_train_samples:
    task_name += f'_{data_args.max_train_samples}'
if data_args.max_eval_samples:
    task_name += f'_{data_args.max_eval_samples}'
if data_args.max_test_samples:
    task_name += f'_{data_args.max_test_samples}'

ckpt_dir = f'ckpt/{fl_args.algorithm}/{task_name}'
if not os.path.exists(os.path.join(base_dir, ckpt_dir)):
    os.makedirs(os.path.join(base_dir, ckpt_dir), exist_ok=True)
fl_args.ckpt = os.path.join(base_dir, ckpt_dir, ckpt)

log_dir = f'log/{fl_args.algorithm}/{task_name}'
if not os.path.exists(os.path.join(base_dir, log_dir)):
    os.makedirs(os.path.join(base_dir, log_dir), exist_ok=True)
log_file = os.path.join(base_dir, log_dir, f'{ckpt}_{datetime.datetime.now():%y-%m-%d %H:%M}.log')
init_log(log_file)

logger.info(data_args)
logger.info(model_args)
logger.info(fl_args)

dataset, model = process_dataset_model(data_args, model_args, fl_args)
num_labels = dataset.num_labels

train_dataset = dataset.train_dataset
eval_dataset = dataset.eval_dataset
test_dataset = dataset.test_dataset

if num_labels != 1:
    if fl_args.algorithm != 'centralized':
        train_datasets = dataset.train_datasets
        eval_datasets = dataset.eval_datasets
        test_datasets = dataset.test_datasets

        mtx, mtx_ = status_mtx(train_datasets, num_labels)
        plot_class_samples(mtx)
        mtx = json.dumps(mtx.tolist())
        logger.info(f'clients train class samples\n{mtx}')

        mtx, mtx_ = status_mtx(eval_datasets, num_labels)
        plot_class_samples(mtx)
        mtx = json.dumps(mtx.tolist())
        logger.info(f'clients eval class samples\n{mtx}')

        mtx, mtx_ = status_mtx(test_datasets, num_labels)
        plot_class_samples(mtx)
        mtx = json.dumps(mtx.tolist())
        logger.info(f'clients test class samples\n{mtx}')

    mtx, mtx_ = status_mtx([train_dataset, eval_dataset, test_dataset], num_labels)
    plot_class_samples(mtx)
    mtx = json.dumps(mtx.tolist())
    logger.info(f'train test class samples\n{mtx}')

s = system(dataset, model, fl_args)

if model_args.model_path:
    logger.info(model_args.model_path)
    s.load(model_args.model_path)

if fl_args.do_train:
    s.run()

if fl_args.do_test:
    s.eval_model(data_loader=s.central_client.test_loader)
    s.get_re_model_gradient_norm()
