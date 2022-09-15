import wandb

from FedBioNLP.utils.status_utils import tensor_cos_sim
from .fedavg import FedAvg
from src.clients.moon import MOONClient
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


class MOON(FedAvg):
    def __init__(self, dataset, model, args):
        super(MOON, self).__init__(dataset, model, args)

        self.clients = []
        t = tqdm.tqdm(range(self.num_clients))
        for i in t:
            self.clients.append(MOONClient(i, dataset, model, args))

    def run(self):
        for self.ite in range(self.num_iterations):
            logger.info(f'********iteration: {self.ite}********')

            # 1. select clients
            select_clients = random.sample(list(range(self.num_clients)), int(self.num_clients * self.select_ratio))
            select_clients.sort()
            weights = self.compute_weights(select_clients)

            # 2. distribute server model to all clients
            global_model_dict = self.model.state_dict()
            for i in select_clients:
                self.clients[i].receive_global_model(global_model_dict)

            # 3. train client models
            model_dicts = []
            for i in select_clients:
                model_state_dict, scalars = self.clients[i].train_model(self.ite)
                model_dicts.append(model_state_dict)
                # print(i, int(torch.cuda.memory_allocated() / (1024 * 1024)))
                # print(i, int(torch.cuda.max_memory_allocated() / (1024 * 1024)))

            # update gradient and compute gradient cosine similarity
            # grad_dicts = self.get_grad_dicts(model_dicts)
            # logger.info('compute model cosine similarity')
            # self.compute_model_cos_sims(model_dicts)

            # 4. aggregate into server model
            model_dict = self.get_global_model_dict(model_dicts, weights)
            self.model.load_state_dict(model_dict)

            # test and save models
            if self.ite % self.test_frequency == 0:
                features = self.test_save_models(select_clients)
                np.set_printoptions(precision=3)
                n = len(features)
                if n != 0:
                    l = len(features[0])
                    for ll in range(l):
                        matrix = np.zeros((n, n))
                        for i in range(n):
                            for j in range(n):
                                sim = cmp_CKA_sim(features[i][ll], features[j][ll])
                                matrix[i][j] = sim
                        avg = (matrix - np.eye(n, n)).sum() / (n * n - n)
                        logger.info('\n' + '\n'.join([str(_) for _ in matrix]))
                        logger.info(f'average CKA: {avg}')
