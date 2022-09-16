import wandb

from FedBioNLP.utils.status_utils import tensor_cos_sim
from .fedavg import FedAvg
from src.clients.fedproto import FedProtoClient
import random
import os
import numpy as np
import logging
import torch
import copy
import json
import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.status_utils import cmp_CKA_sim

logger = logging.getLogger(os.path.basename(__file__))


class FedProto(FedAvg):
    def __init__(self, dataset, model, args):
        super(FedProto, self).__init__(dataset, model, args)

        self.clients = []
        t = tqdm.tqdm(range(self.num_clients))
        for i in t:
            self.clients.append(FedProtoClient(i, dataset, model, args))
