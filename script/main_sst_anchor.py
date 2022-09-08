import os
import sys
import datetime
import warnings
import logging
import json
from transformers import HfArgumentParser

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

task_name = 'sst'
dataset_name = 'default'
# dataset_path = os.path.join(base_dir, f"data/20news_data.h5")
model_type = 'bert'
model_name = 'bert-base-uncased'
model_path = os.path.join(base_dir,
                          f"ckpt/centralized/sst_s223_800_800_800/{model_name}_anchor_tgwg")
label_words = [
    ['negative'], ['somewhat negative'], ['neutral'], ['somewhat positive'], ['positive']
]
anchor_texts = [
    f"This is a sentence describing a {label_word[0]} sentiment." for label_word in label_words
]

template_text = 'The sentiment of this sentence: {"mask"}. {"placeholder":"text_a"}'
verbalizer = 'anchor'
tunning_name = 'anchor'

config = [
    f'--task_name={task_name}',
    f'--dataset_nam={dataset_name}',
    # f'--dataset_path={dataset_path}',
    f'--model_type={model_type}',
    f'--model_name={model_name}',
    f'--tunning_name={tunning_name}',
    # f'--model_path={model_path}',
    '--lr=5e-5',
    '--algorithm=centralized',
    '--split_type=centralized',
    '--train_batch_size=16',
    '--eval_batch_size=16',
    f'--template_text={template_text}',
    f'--label_words={json.dumps(label_words)}',
    f'--anchor_texts={json.dumps(anchor_texts)}',
    f'--verbalizer={verbalizer}',
    # '--max_train_samples=800',
    # '--max_eval_samples=800',
    # '--max_test_samples=800',
    '--seed=223',
    '--num_iterations=100',
    '--do_train=True',
    '--do_test=True'
]

parser = HfArgumentParser((DataArguments, ModelArguments, FederatedLearningArguments))
data_args, model_args, fl_args = parser.parse_args_into_dataclasses(config)
set_seed(fl_args.seed)

model_name = model_name.replace('/', '_')
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

if model_args.model_path:
    logger.info(model_args.model_path)
    s.load(model_args.model_path)

if fl_args.do_train:
    s.run()

if fl_args.do_test:
    s.eval_model(data_loader=s.central_client.test_loader)
