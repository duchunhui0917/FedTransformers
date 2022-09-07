import os
import sys
import datetime
import warnings
import logging
import json

import numpy as np
from transformers import HfArgumentParser
from src.utils.status_utils import cmp_CKA_sim

sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from src.arguments import DataArguments, ModelArguments, FederatedLearningArguments
from src.processor import process_dataset_model
from src import plot_class_samples, status_mtx, init_log, set_seed
from src.utils.plot_utils import plot_hist

base_dir = os.path.expanduser('~/FedTransformers')
warnings.filterwarnings('ignore')
logger = logging.getLogger(os.path.basename(__file__))

task_name = '20news'
dataset_path = os.path.join(base_dir, f"data/20news_data.h5")
model_type = 'bert'
model_name = 'bert-base-uncased'
tunning_method = 'frozen'
prompt_method = 'manual'
seed = 222

model_path = os.path.join(base_dir,
                          f"ckpt/FedAvg/20news_s223_100_100_100/{model_name}_{prompt_method}_0.01_10_5")
config = [
    f'--task_name={task_name}',
    f'--dataset_path={dataset_path}',
    f'--model_type={model_type}',
    f'--model_name={model_name}',
    # f'--tunning_method={tunning_method}',
    f'--prompt_method={prompt_method}',
    # f'--model_path={model_path}',
    '--lr=1e-5',
    '--algorithm=centralized',
    '--split_type=centralized',
    '--train_batch_size=8',
    '--eval_batch_size=8',
    '--max_train_samples=100',
    '--max_eval_samples=100',
    '--max_test_samples=100',
    f'--seed={seed}',
    '--num_iterations=100',
    '--do_train=False',
    '--do_test=True'
]

parser = HfArgumentParser((DataArguments, ModelArguments, FederatedLearningArguments))
data_args, model_args, fl_args = parser.parse_args_into_dataclasses(config)
set_seed(fl_args.seed)

model_name = model_name.replace('/', '_')
if model_args.tunning_method:
    model_name += f'_{model_args.tunning_method}'
if model_args.prompt_method:
    model_name += f'_{model_args.prompt_method}'

if fl_args.split_type == 'label_split':
    ckpt = f'{model_name}_{fl_args.dirichlet_alpha}_{fl_args.num_clients}_{fl_args.num_epochs}'
elif fl_args.split_type == 'feature_split':
    ckpt = f'{model_name}_{fl_args.num_clients}_{fl_args.num_epochs}'
elif fl_args.split_type == 'doc_split':
    ckpt = f'{model_name}_{fl_args.num_epochs}'
else:
    ckpt = f'{model_name}'

if fl_args.algorithm == 'centralized':
    from src.systems.base import Base as system

elif fl_args.algorithm == 'FedAvg':
    from src.systems.fedavg import FedAvg as system

elif fl_args.algorithm == 'FedProx':
    from src.systems.fedprox import FedProx as system

    fl_args.mu = 1
elif fl_args.algorithm == 'MOON':
    from src.systems.moon import MOON as system

    fl_args.mu = 0
    fl_args.temperature = 0.05
else:
    raise NotImplementedError

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
features = []
n = 2
for i in range(n):
    p = model_path + f'_client{i}_tgwp'
    logger.info(p)
    s.load(p)
    _, feature = s.eval_model(data_loader=s.central_client.test_loader)
    features.append(feature)

matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim = cmp_CKA_sim(features[i], features[j])
        matrix[i][j] = sim[0]
print(matrix)
