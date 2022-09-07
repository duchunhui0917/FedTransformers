import json
import os
import random

import h5py

from src.datasets.utils import (
    get_embedding_Kmeans,
    generate_idxes_dirichlet,
    generate_idxes_group,
    sentence_embedding_tsne
)

import logging
from src.arguments import (
    SS_DATASETS,
    SC_DATASETS,
    QA_DATASETS,
    RE_DATASETS,
    TC_DATASETS,
    LM_DATASETS
)

logger = logging.getLogger(os.path.basename(__file__))
base_dir = os.path.expanduser('~/FedTransformers')


def select(x, idx):
    if isinstance(x, dict):
        d = {}
        for key, val in x.items():
            d[key] = [xx for i, xx in enumerate(x[key]) if i in idx]
        return d
    else:
        return x.select(idx)


def num(dataset):
    return len(dataset['input_example']) if isinstance(dataset, dict) else len(dataset)


def map_fn(dataset):
    input_ids = dataset['input_ids']
    if not isinstance(input_ids[0], list):
        return input_ids
    else:
        res = []
        for input_id in input_ids:
            res += input_id
        return res


def label_split(dataset, fl_args):
    logger.info('start splitting data according to label shift')

    num_labels = dataset.num_labels
    train_dataset = dataset.train_dataset
    eval_dataset = dataset.eval_dataset
    test_dataset = dataset.test_dataset

    num_clusters = fl_args.num_clusters
    num_clients = fl_args.num_clients
    dirichlet_alpha = fl_args.dirichlet_alpha

    if num_clusters == 0:
        train_idxes = generate_idxes_dirichlet(train_dataset['label'], num_clients, num_labels, dirichlet_alpha,
                                               seed=fl_args.seed)
        eval_idxes = generate_idxes_dirichlet(eval_dataset['label'], num_clients, num_labels, dirichlet_alpha,
                                              seed=fl_args.seed)
        test_idxes = generate_idxes_dirichlet(test_dataset['label'], num_clients, num_labels, dirichlet_alpha,
                                              seed=fl_args.seed)
    else:
        train_idxes = generate_idxes_group(train_dataset['label'], num_clients, num_labels, dirichlet_alpha,
                                           num_clients // fl_args.num_clusters, fl_args.seed)
        eval_idxes = generate_idxes_dirichlet(eval_dataset['label'], num_clients, num_labels, dirichlet_alpha,
                                              seed=fl_args.seed)

        test_idxes = generate_idxes_group(test_dataset['label'], num_clients, num_labels, dirichlet_alpha,
                                          num_clients // fl_args.num_clusters, fl_args.seed)
    train_datasets = []
    eval_datasets = []
    test_datasets = []

    for i in range(num_clients):
        train_idx = train_idxes[i]
        num_train = len(train_idx)
        train_datasets.append(select(train_dataset, train_idx))

        eval_idx = eval_idxes[i]
        num_eval = len(eval_idx)
        eval_datasets.append(select(eval_dataset, eval_idx))

        test_idx = test_idxes[i]
        num_test = len(test_idx)
        test_datasets.append(select(test_dataset, test_idx))
        logger.info(f'client {i}, number of samples of train/eval/test dataset: {num_train}/{num_eval}/{num_test}')
    dataset.train_datasets = train_datasets
    dataset.eval_datasets = eval_datasets
    dataset.test_datasets = test_datasets


def feature_split(dataset, fl_args):
    logger.info('start splitting data according to feature shift')

    num_clients = fl_args.num_clients
    seed = fl_args.seed
    name = dataset.task_name
    if dataset.dataset_name:
        name += f'_{dataset.dataset_name}'
    tokenizer = dataset.tokenizer

    train_dataset = dataset.train_dataset
    eval_dataset = dataset.eval_dataset
    test_dataset = dataset.test_dataset

    num_train = len(train_dataset)
    num_eval = len(eval_dataset)
    num_test = len(test_dataset)

    if 'text' in train_dataset.column_names:
        train_text = train_dataset['text']
        eval_text = eval_dataset['text']
        test_text = test_dataset['text']
    else:
        train_text = tokenizer.batch_decode(map(map_fn, train_dataset), skip_special_tokens=True)
        eval_text = tokenizer.batch_decode(map(map_fn, eval_dataset), skip_special_tokens=True)
        test_text = tokenizer.batch_decode(map(map_fn, test_dataset), skip_special_tokens=True)

    train_path = os.path.join(base_dir,
                              f'embedding/{name}_{num_train}_{num_clients}_s{seed}_train.pkl')
    eval_path = os.path.join(base_dir,
                             f'embedding/{name}_{num_eval}_{num_clients}_s{seed}_eval.pkl')
    test_path = os.path.join(base_dir,
                             f'embedding/{name}_{num_test}_{num_clients}_s{seed}_test.pkl')
    train_classes = get_embedding_Kmeans(train_text, num_clients, train_path)
    eval_classes = get_embedding_Kmeans(eval_text, num_clients, eval_path)
    test_classes = get_embedding_Kmeans(test_text, num_clients, test_path)

    train_idxes = [[] for _ in range(num_clients)]
    for i, c in enumerate(train_classes):
        train_idxes[c].append(i)
    eval_idxes = [[] for _ in range(num_clients)]
    for i, c in enumerate(eval_classes):
        eval_idxes[c].append(i)
    test_idxes = [[] for _ in range(num_clients)]
    for i, c in enumerate(test_classes):
        test_idxes[c].append(i)

    train_datasets = []
    eval_datasets = []
    test_datasets = []

    for i in range(num_clients):
        train_idx = train_idxes[i]
        num_train = len(train_idx)
        train_datasets.append(train_dataset.select(train_idx))

        eval_idx = eval_idxes[i]
        num_eval = len(eval_idx)
        eval_datasets.append(eval_dataset.select(eval_idx))

        test_idx = test_idxes[i]
        num_test = len(test_idx)
        test_datasets.append(test_dataset.select(test_idx))
        logger.info(f'client {i}, number of samples of train/eval/test dataset: {num_train}/{num_eval}/{num_test}')

    dataset.train_datasets = train_datasets
    dataset.eval_datasets = eval_datasets
    dataset.test_datasets = test_datasets


def idx_split(dataset, fl_args):
    logger.info('start splitting data according to idxes')

    num_clients = fl_args.num_clients
    partition_path = fl_args.partition_path
    partition_group = fl_args.partition_group

    rf = h5py.File(partition_path, 'r')
    train_idxes = [rf[partition_group]["partition_data"][str(idx)]["train"][()] for idx in range(num_clients)]
    eval_idxes = [rf[partition_group]["partition_data"][str(idx)]["test"][()] for idx in range(num_clients)]
    test_idxes = [rf[partition_group]["partition_data"][str(idx)]["test"][()] for idx in range(num_clients)]

    rf.close()

    train_datasets = []
    eval_datasets = []
    test_datasets = []

    train_dataset = dataset.train_dataset
    eval_dataset = dataset.eval_dataset
    test_dataset = dataset.test_dataset

    for i in range(num_clients):
        train_idx = train_idxes[i]
        num_train = len(train_idx)
        train_datasets.append(train_dataset.select(train_idx))

        eval_idx = eval_idxes[i]
        num_eval = len(eval_idx)
        eval_datasets.append(eval_dataset.select(eval_idx))

        test_idx = test_idxes[i]
        num_test = len(test_idx)
        test_datasets.append(test_dataset.select(test_idx))
        logger.info(f'client {i}, number of samples of train/eval/test dataset: {num_train}/{num_eval}/{num_test}')

    dataset.train_datasets = train_datasets
    dataset.eval_datasets = eval_datasets
    dataset.test_datasets = test_datasets


def uniform_split(dataset, fl_args):
    logger.info('start splitting data uniformly')

    num_clients = fl_args.num_clients

    train_dataset = dataset.train_dataset
    eval_dataset = dataset.eval_dataset
    test_dataset = dataset.test_dataset

    train_idxes = list(range(len(train_dataset)))
    random.shuffle(train_idxes)
    num_train_per_client = len(train_idxes) // num_clients

    eval_idxes = list(range(len(eval_dataset)))
    random.shuffle(eval_idxes)
    num_eval_per_client = len(eval_idxes) // num_clients

    test_idxes = list(range(len(test_dataset)))
    random.shuffle(test_idxes)
    num_test_per_client = len(test_idxes) // num_clients

    train_datasets = []
    eval_datasets = []
    test_datasets = []

    for i in range(num_clients):
        train_idx = train_idxes[i * num_train_per_client: (i + 1) * num_train_per_client]
        cur_train_dataset = train_dataset.select(train_idx)
        num_train = len(cur_train_dataset)
        train_datasets.append(cur_train_dataset)

        eval_idx = train_idxes[i * num_eval_per_client: (i + 1) * num_eval_per_client]
        cur_eval_dataset = eval_dataset.select(eval_idx)
        num_eval = len(cur_eval_dataset)
        eval_datasets.append(cur_eval_dataset)

        test_idx = test_idxes[i * num_test_per_client: (i + 1) * num_test_per_client]
        cur_test_dataset = test_dataset.select(test_idx)
        num_test = len(cur_test_dataset)
        test_datasets.append(cur_test_dataset)

        logger.info(f'client {i}, number of samples of train/eval/test dataset: {num_train}/{num_eval}/{num_test}')
    dataset.train_datasets = train_datasets
    dataset.eval_datasets = eval_datasets
    dataset.test_datasets = test_datasets


def doc_split(dataset, fl_args):
    logger.info('start splitting data according to doc shift')

    train_dataset = dataset.train_dataset
    eval_dataset = dataset.eval_dataset
    test_dataset = dataset.test_dataset

    unique_docs = dataset.unique_docs
    fl_args.num_clients = len(unique_docs)

    train_docs = train_dataset['doc']
    eval_docs = eval_dataset['doc']
    test_docs = test_dataset['doc']

    train_datasets = []
    eval_datasets = []
    test_datasets = []

    texts = []
    for i, cur_doc in enumerate(unique_docs):
        train_idx = [idx for idx, doc in enumerate(train_docs) if doc == cur_doc]
        num_train = len(train_idx)
        cur_train_dataset = train_dataset.select(train_idx)
        train_datasets.append(cur_train_dataset)
        texts.append(cur_train_dataset['text'])

        eval_idx = [idx for idx, doc in enumerate(eval_docs) if doc == cur_doc]
        num_eval = len(eval_idx)
        cur_eval_dataset = eval_dataset.select(eval_idx)
        eval_datasets.append(cur_eval_dataset)

        test_idx = [idx for idx, doc in enumerate(test_docs) if doc == cur_doc]
        num_test = len(test_idx)
        cur_test_dataset = test_dataset.select(test_idx)
        test_datasets.append(cur_test_dataset)

        logger.info(f'client {i}, number of samples of train/eval/test dataset: {num_train}/{num_eval}/{num_test}')
    sentence_embedding_tsne(texts)
    dataset.train_datasets = train_datasets
    dataset.eval_datasets = eval_datasets
    dataset.test_datasets = test_datasets


def class_split(dataset, fl_args):
    logger.info('start splitting data according to class shift')

    num_clients = fl_args.num_clients
    classes = eval(fl_args.classes)
    assert num_clients == len(classes)

    train_dataset = dataset.train_dataset
    eval_dataset = dataset.eval_dataset
    test_dataset = dataset.test_dataset

    train_label = train_dataset['label']
    eval_label = eval_dataset['label']
    test_label = test_dataset['label']

    train_datasets = []
    eval_datasets = []
    test_datasets = []

    for i in range(num_clients):
        client_class = classes[i]
        train_idx = [idx for idx, label in enumerate(train_label) if label in client_class]
        num_train = len(train_idx)
        train_datasets.append(train_dataset.select(train_idx))

        eval_idx = [idx for idx, label in enumerate(eval_label) if label in client_class]
        num_eval = len(eval_idx)
        eval_datasets.append(eval_dataset.select(eval_idx))

        test_idx = [idx for idx, label in enumerate(test_label) if label in client_class]
        num_test = len(test_idx)
        test_datasets.append(test_dataset.select(test_idx))

        logger.info(f'client {i}, number of samples of train/eval/test dataset: {num_train}/{num_eval}/{num_test}')

    dataset.train_datasets = train_datasets
    dataset.eval_datasets = eval_datasets
    dataset.test_datasets = test_datasets


def process_dataset_model(data_args, model_args, fl_args):
    split_type = fl_args.split_type
    task_name = data_args.task_name
    dataset_name = data_args.dataset_name

    if task_name == "superglue":
        # from .datasets.superglue import SuperGlueDataset as ds
        from .datasets.sequence_classification import SequenceClassificationDataset as ds

        if dataset_name == 'copa':
            from src.models.multiple_choice import MultiChoiceModel as md
        else:
            from src.models.sequence_classification import SequenceClassificationModel as md

    elif task_name == "glue":
        # from .datasets.glue import GlueDataset as ds
        from .datasets.sequence_classification import SequenceClassificationDataset as ds
        from src.models.sequence_classification import SequenceClassificationModel as md

    elif task_name == 'xglue':
        # from .datasets.xglue import XGlueDataset as ds
        from .datasets.sequence_classification import SequenceClassificationDataset as ds

        if dataset_name in ['ner', 'pos']:
            from src.models.token_classification import TokenClassificationModel as md
        elif dataset_name in ['mlqa']:
            from src.models.question_answering import QuestionAnsweringModel as md
        elif dataset_name in ['qg', 'ntg']:
            from src.models.seq2seq import Sequence2SequenceModel as md
        else:
            from src.models.sequence_classification import SequenceClassificationModel as md

    elif task_name in SC_DATASETS:
        from .datasets.sequence_classification import SequenceClassificationDataset as ds
        from src.models.sequence_classification import SequenceClassificationModel as md

    elif task_name in QA_DATASETS:
        from .datasets.question_answering import QuestionAnsweringDataset as ds
        from src.models.question_answering import QuestionAnsweringModel as md

    elif task_name in TC_DATASETS:
        from .datasets.token_classification import TokenClassificationDataset as ds
        from src.models.token_classification import TokenClassificationModel as md

    elif task_name in SS_DATASETS:
        from .datasets.seq2seq import Sequence2SequenceDataset as ds
        from src.models.seq2seq import Sequence2SequenceModel as md
    elif task_name in LM_DATASETS:
        from .datasets.language_modeling import CasualLanguageModelingDataset as ds
        from src.models.language_modeling import LanguageModelingModel as md
    elif task_name in RE_DATASETS:
        from .datasets.relation_extraction import RelationExtractionDataset as ds
        from src.models.relation_extraction import RelationExtractionModel as md
    else:
        raise Exception('invalid task name')

    dataset = ds(data_args, model_args)

    model_args.num_labels = dataset.num_labels
    model = md(model_args, dataset)
    num_labels = dataset.num_labels
    num_train = num(dataset.train_dataset)
    num_eval = num(dataset.eval_dataset)
    num_test = num(dataset.test_dataset)

    logger.info(f'number of labels: {num_labels}')
    logger.info(f'number of samples of train/eval/test dataset: {num_train}/{num_eval}/{num_test}')

    if split_type == 'doc_split':
        doc_split(dataset, fl_args)
    elif split_type == 'label_split':
        label_split(dataset, fl_args)
    elif split_type == 'feature_split':
        feature_split(dataset, fl_args)
    elif split_type == 'class_split':
        class_split(dataset, fl_args)
    elif split_type == 'uniform_split':
        uniform_split(dataset, fl_args)
    elif split_type == 'idx_split':
        idx_split(dataset, fl_args)

    return dataset, model
