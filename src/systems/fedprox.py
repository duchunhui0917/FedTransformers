from .base import Base
from src.clients.base import BaseClient
import random
import os
import logging

logger = logging.getLogger(os.path.basename(__file__))


class FedProx(Base):
    def __init__(self, dataset, model, args):
        super(FedProx, self).__init__(dataset, model, args)
        self.aggregate_method = args.aggregate_method

        self.central_client = BaseClient(None, dataset, model, args)

        self.clients = [
            BaseClient(i, dataset, model, args) for i in range(self.num_clients)
        ]
        self.momentum = args.server_momentum
        self.layers = args.layers.split('*')
        self.select_ratio = args.select_ratio
        self.model_cos_sims = {}
        self.grad_cos_sims = {}

    def run(self, m='f1'):
        for self.ite in range(self.num_iterations):
            logger.info(f'********iteration: {self.ite}********')

            # 1. distribute server model to all clients
            global_model_dict = self.model.state_dict()
            for client in self.clients:
                client.receive_global_model(global_model_dict)

            # 2. select clients
            select_clients = random.sample(list(range(self.num_clients)), int(self.num_clients * self.select_ratio))
            select_clients.sort()
            weights = self.compute_weights(select_clients)

            # 2. train client models
            model_dicts = []
            for i in select_clients:
                model_state_dict = self.clients[i].train_model(self.ite)
                model_dicts.append(model_state_dict)
                # print(i, int(torch.cuda.memory_allocated() / (1024 * 1024)))
                # print(i, int(torch.cuda.max_memory_allocated() / (1024 * 1024)))

            # update gradient and compute gradient cosine similarity
            # grad_dicts = self.get_grad_dicts(model_dicts)
            # logger.info('compute model cosine similarity')
            # self.compute_model_cos_sims(model_dicts)

            # 3. aggregate into server model
            model_dict = self.get_global_model_dict(model_dicts, weights)
            self.model.load_state_dict(model_dict)

            # test and save models
            if self.ite % self.test_frequency == 0:
                self.test_save_models(m, select_clients)
