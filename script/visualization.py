import sys
import os

sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os.path
import numpy as np
from FedBioNLP.processors import process_dataset
from FedBioNLP.systems import *
import argparse
import pickle
import logging
import warnings
import datetime
from FedBioNLP import process_dataset, plot_class_samples, status_mtx, init_log, set_seed, visualize_features

set_seed(223)
base_dir = os.path.expanduser('~/src')

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

# control hyperparameters
parser.add_argument('--load_model', default=True)
parser.add_argument('--vis_labels', default=True)
parser.add_argument('--vis_docs', default=False)
parser.add_argument('--case_study', default=False)
parser.add_argument('--load_rep', default=False)
parser.add_argument('--mode', default='average')

# common FL hyperparameters
parser.add_argument('--task_name', type=str,
                    default='i2b2',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100',
                             '20news', 'agnews', 'sst_2', 'sentiment140',
                             'GAD', 'EU-ADR', 'PGR_Q1', 'PGR_Q2', 'CoMAGC', 'PolySearch',
                             'i2b2', 'i2b2_BIDMC', 'i2b2_Partners',
                             'semeval_2010_task8',
                             'wikiner', 'ploner',
                             'squad_1.1',
                             'cnn_dailymail', 'cornell_movie_dialogue',
                             'Books*Electronics*Home_&_Kitchen*Movies_&_TV'
                             ])
parser.add_argument('--alg', type=str, default='centralized')
parser.add_argument('--split_type', default='centralized',
                    choices=['centralized', 'doc_split', 'label_split', 'feature_split'])
parser.add_argument('--beta', type=int, default=0.5)
parser.add_argument('--n_clients', type=int, default=10)
parser.add_argument('--n_clusters', type=int, default=0)
parser.add_argument('--select_ratio', type=float, default=1)
parser.add_argument('--sm', default=0)
parser.add_argument('--cm', default=0)
parser.add_argument('--save_central', default=True)
parser.add_argument('--save_global', default=False)
parser.add_argument('--save_personal', default=False)
parser.add_argument('--aggregate_method', default='sample',
                    choices=['equal', 'sample', 'attention'])
parser.add_argument('--layers',
                    default='.*embedding*layer.0*layer.1*layer.2*layer.3*layer.4*layer.5*classifier')
# parser.add_argument('--layers', default='.')
# parser.add_argument('--vocab_file', default=os.path.join(base_dir, 'data/amazon_review/vocab.json'))
parser.add_argument('--vocab_file', default=os.path.join(base_dir, 'data/glove.6B/glove.6B.100d.json'))

# training hyperparameters
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--model_name', type=str, default='emilyalsentzer/Bio_ClinicalBERT',
                    choices=['CNN',
                             'LSTM',
                             'distilbert-base-uncased',
                             'google/mobilebert-uncased',
                             'bert-base-uncased',
                             'bert-large-uncased',
                             'google/vit-base-patch16-224-in21k',
                             'dmis-lab/biobert-v1.1',
                             'allenai/scibert_scivocab_uncased',
                             'emilyalsentzer/Bio_ClinicalBERT',
                             'cardiffnlp/twitter-roberta-base-sentiment'])
parser.add_argument('--adapter_name', default='P-Tunning_v2',
                    choices=[None, 'bottleneck_adapter', 'P-Tunning_v2', 'P-Tunning'])
parser.add_argument('--load_method', default='pre_train', choices=['pre_train', 'scratch', 'mlm'])
parser.add_argument('--load_path', default=None)
parser.add_argument('--num_iterations', type=int, default=100)
parser.add_argument('--num_epochs', default=5)
parser.add_argument('--num_batches', default=0)
parser.add_argument('--opt', default='AdamW',
                    choices=['SGD', 'AdamW', 'WPOptim'])
parser.add_argument('--batch_size', default=8)
parser.add_argument('--weight_sampler', default=False)

args = parser.parse_args()

model_name = args.model_name.replace('/', '_')
if args.adapter_name is not None:
    model_name += f'_{args.adapter_name}'
if args.load_method != 'pre_train':
    model_name += f'_{args.load_method}'
if args.split_type == 'label_split':
    ckpt = f'{model_name}_{args.beta}_{args.n_clients}_{args.n_epochs}'
elif args.split_type == 'feature_split':
    ckpt = f'{model_name}_{args.n_clients}_{args.n_epochs}'
else:
    ckpt = f'{model_name}'

args.mlm_method = 'None'
args.mlm_prob = 0.8
args.K_LCA = 1
args.lambda_ = None

if args.alg == 'centralized':
    system = Centralized
    args.num_gcn_layers = 0

    if args.mlm_method != 'None':
        ckpt = f'{ckpt}_{args.mlm_method}_{args.mlm_prob}_{args.K_LCA}'
    if args.lambda_ is not None:
        ckpt += f'_{args.lambda_}'

elif args.alg == 'SCL':
    system = SCL
elif args.alg == 'FedAvg':
    system = FedAvg

    if args.mlm_method != 'None':
        ckpt = f'{ckpt}_{args.mlm_method}_{args.mlm_prob}_{args.K_LCA}'
    if args.lambda_ is not None:
        ckpt += f'_{args.lambda_}'

elif args.alg == 'DomainGRL':
    system = Centralized
    args.num_gcn_layers = 0
    args.lambda_ = 1
    if args.lambda_ is not None:
        ckpt = f'{ckpt}_{args.lambda_}'
elif args.alg == 'DomainMMD':
    system = Centralized
    if args.lambda_ is not None:
        ckpt = f'{ckpt}_{args.lambda_}'

elif args.alg == 'FedProx':
    system = FedProx
    args.mu = 1
elif args.alg == 'MOON':
    system = MOON
    args.mu = 0
    args.temperature = 0.05

    ckpt = f'{ckpt}_{args.mu}_{args.temperature}'

elif args.alg == 'FedGS':
    system = FedGS
elif args.alg == 'FedGP':
    system = FedGP
elif args.alg == 'HarmoFL':
    system = HarmoFL
elif args.alg == 'PartialFL':
    system = PartialFL
    args.pk = 're_classifier'
elif args.alg == 'GSN':
    system = GSN
    args.mu = 0
    args.temperature = 0.05

elif args.alg == 'FedMatch':
    system = PartialFL
elif args.alg == 'pFedMe':
    system = pFedMe
    args.n_inner_loops = 4
    args.mu = 1
elif args.alg == 'ICFA':
    args.n_clusters = 0
    system = ICFA
else:
    raise NotImplementedError

ckpt_dir = f'ckpt/{args.alg}/{args.dataset_name}'
if not os.path.exists(os.path.join(base_dir, ckpt_dir)):
    os.makedirs(os.path.join(base_dir, ckpt_dir), exist_ok=True)
args.ckpt = os.path.join(base_dir, ckpt_dir, ckpt + '_central')

log_dir = f'log/{args.alg}/{args.dataset_name}_{ckpt}_{datetime.datetime.now():%y-%m-%d %H:%M}.log'
log_file = os.path.join(base_dir, log_dir)
init_log(log_file)

_ = process_dataset(args.dataset_name, args.model_name, args.split_type, args.n_clients, args)
clients, train_datasets, test_datasets, train_dataset, test_dataset, model, res = _
doc_index = test_dataset.doc_index

cs = Centralized(train_datasets, test_datasets, train_dataset, test_dataset, model, args)
if args.load_model:
    print(args.ckpt)

    cs.load(args.ckpt)
    logging.info('model has been loaded')

p = f'rep/{args.alg}/{args.dataset_name}_{ckpt}'
rep = os.path.join(base_dir, p)
if os.path.exists(rep) and args.load_rep:
    print(rep)
    with open(rep, 'rb') as f:
        ls = pickle.load(f)
    logging.info('pickle has been loaded')
else:
    ls = []
    metric, inputs, labels, features, logits = cs.eval_model()
    ls.append({'metric': metric, 'labels': labels, 'features': features})

    # for i, test_loader in enumerate(cs.test_loaders):
    #     metric, inputs, labels, features, logits = cs.test_model(data_loader=test_loader)
    #     ls.append({'metric': metric, 'labels': labels, 'features': features})
    with open(rep, 'wb') as f:
        pickle.dump(ls, f)
    print(rep)
    logging.info('pickle has been dumped')
for x in ls:
    print(x['metric'])


def vis(nodes, all_layers=True):
    if all_layers:
        features_n = [x['features'] for x in ls]
        labels_n = [x['labels'][0] for x in ls]
    else:
        features_n = [[x['features'][-1]] for x in ls]
        labels_n = [x['labels'][0] for x in ls]

    vis_features = [np.concatenate([features_n[n][l] for n in nodes]) for l in range(len(features_n[0]))]
    # vis_labels = np.concatenate(
    #     [labels_n[n] for n in nodes]
    # )
    vis_docs = np.concatenate(
        [np.ones(len(labels_n[n]), dtype=np.int64) * n for n in nodes])

    vis_labels = np.concatenate([
        labels_n[n] + np.ones(len(labels_n[n]), dtype=np.int64) * 2 * i for i, n in enumerate(nodes)
    ])
    if args.vis_docs:
        visualize_features(vis_features, vis_docs, mode=args.mode)
    if args.vis_labels:
        visualize_features(vis_features, vis_labels, mode=args.mode)


vis([0])
