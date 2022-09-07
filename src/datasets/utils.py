import pickle
import logging
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import string
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

logger = logging.getLogger(os.path.basename(__file__))

import re
import string
from collections import defaultdict, Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_embedding_Kmeans(text, n_clients, path, bsz=16):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info('embedding has been loaded')

    else:
        embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda:0')
        embeddings = embedder.encode(text, show_progress_bar=True, batch_size=bsz)

        with open(path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info('embedding has been dumped')

    logger.info("start kmeans embedding")
    clustering_model = KMeans(n_clusters=n_clients)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    return cluster_assignment


def sentence_embedding_tsne(texts, bsz=16):
    embeddings = []
    labels = []
    embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda:0')
    for i, text in enumerate(texts):
        embedding = embedder.encode(text, show_progress_bar=True, batch_size=bsz)
        embeddings.append(embedding)
        labels.append(np.ones(len(text)) * i)
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    d = {0: 'SST-2', 1: 'IMDB', 2: 'YELP', 3: 'MR'}
    labels = [d[label] for label in labels]
    # d = {0: 'SST-2', 1: 'IMDB', 2: 'YELP', 3: 'MR'}
    # labels = [d[label] for label in labels]


    X_features = TSNE(n_components=2, random_state=33).fit_transform(embeddings)
    palette = sns.color_palette("bright", n_colors=np.unique(labels).shape[0])
    sns.scatterplot(X_features[:, 0], X_features[:, 1], hue=labels, palette=palette, s=8)
    plt.show()


def generate_idxes_dirichlet(targets, num_clients, num_labels, alpha=0.5, seed=None):
    idxes = [[] for _ in range(num_clients)]
    targets = np.array(targets)
    for k in range(num_labels):
        idx_k = np.where(targets == k)[0]
        if seed is not None:
            np.random.seed(seed + k)
        distribution = np.random.dirichlet(np.repeat(alpha, num_clients))
        distribution = (np.cumsum(distribution) * idx_k.size).astype(int)[:-1]
        idx_k_split = np.split(idx_k, distribution)
        for i in range(num_clients):
            idxes[i].extend(idx_k_split[i].tolist())
    return idxes


def generate_idxes_group(targets, n_clients, n_classes, beta, n_groups, seed):
    idxes = []
    targets_per_cluster = len(targets) // n_groups
    for i in range(n_groups):
        cur_targets = targets[i * targets_per_cluster:(i + 1) * targets_per_cluster]
        cur_idxes = generate_idxes_dirichlet(cur_targets, n_clients // n_groups, n_classes, beta, seed)
        cur_idxes = [[idx + i * targets_per_cluster for idx in cur_idx] for cur_idx in cur_idxes]
        idxes.extend(cur_idxes)
    cur_targets = targets[(n_groups - 1) * targets_per_cluster:len(targets)]
    cur_idxes = generate_idxes_dirichlet(cur_targets, n_clients % n_groups, n_classes, beta)
    cur_idxes = [[idx + (n_groups - 1) * targets_per_cluster for idx in cur_idx] for cur_idx in cur_idxes]
    idxes.extend(cur_idxes)
    return idxes


def add_space_for_punctuation(s):
    punctuations = string.punctuation
    ns = s[:]
    j = 0
    for i, x in enumerate(s):
        if x in punctuations:
            ns = ns[:j] + ' ' + ns[j] + ' ' + ns[j + 1:]
            j += 2
        j += 1
    return ns


def sst_label_fn(ls):
    res = np.zeros(len(ls), dtype=np.int64)
    for i, x in enumerate(ls):
        if 0 <= x < 0.2:
            res[i] = 0
        elif 0.2 <= x < 0.4:
            res[i] = 1
        elif 0.4 <= x < 0.6:
            res[i] = 2
        elif 0.6 <= x < 0.8:
            res[i] = 3
        else:
            res[i] = 4
    return res
