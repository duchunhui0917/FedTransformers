import wandb

from FedBioNLP.utils.status_utils import tensor_cos_sim
from .base import Base
from src.clients.base import BaseClient
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


class FedAvg(Base):
    def __init__(self, dataset, model, args):
        super(FedAvg, self).__init__(dataset, model, args)
        self.aggregate_method = args.aggregate_method

        self.num_samples = [len(dataset['input_example']) if isinstance(dataset, dict) else len(dataset)
                            for dataset in self.dataset.train_datasets]

        logger.info('initializing clients')
        self.clients = []
        t = tqdm.tqdm(range(self.num_clients))
        for i in t:
            self.clients.append(BaseClient(i, dataset, model, args))
        self.momentum = args.server_momentum
        self.layers = args.layers.split('*')
        self.select_ratio = args.select_ratio
        self.model_cos_sims = {}
        self.grad_cos_sims = {}

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
            losses = []
            for i in select_clients:
                model_state_dict, scalars = self.clients[i].train_model(self.ite)
                losses.append(scalars['loss0'])
                model_dicts.append(model_state_dict)
                # print(i, int(torch.cuda.memory_allocated() / (1024 * 1024)))
                # print(i, int(torch.cuda.max_memory_allocated() / (1024 * 1024)))
            avg_loss = sum(losses) / len(losses)
            logger.info(f'training loss: {avg_loss}')
            wandb.log({'training loss': avg_loss}, step=self.ite)

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
                # np.set_printoptions(precision=3)
                # n = len(features)
                # if n != 0:
                #     l = len(features[0])
                #     for ll in range(l):
                #         matrix = np.zeros((n, n))
                #         for i in range(n):
                #             for j in range(n):
                #                 sim = cmp_CKA_sim(features[i][ll], features[j][ll])
                #                 matrix[i][j] = sim
                #         avg = (matrix - np.eye(n, n)).sum() / (n * n - n)
                #         logger.info('\n' + '\n'.join([str(_) for _ in matrix]))
                #         logger.info(f'average CKA: {avg}')

    def get_grad_dicts(self, model_dicts):
        grad_dicts = [{} for _ in range(self.num_clients)]
        for i in range(self.num_clients):
            for key, val in self.model.state_dict().items():
                grad = val - model_dicts[i][key]
                grad_dicts[i].update({key: grad})
        return grad_dicts

    def get_global_model_dict(self, model_dicts, weights):
        logger.info(f'aggregation weights: {weights}')
        model_dict = self.model.state_dict()
        for l, key in enumerate(model_dict.keys()):
            if not isinstance(model_dict[key], torch.FloatTensor):
                continue
            model_dict[key] = self.momentum * model_dict[key]
            for i, sd in enumerate(model_dicts):
                val = sd[key] * weights[i]
                model_dict[key] += (1 - self.momentum) * val
        return model_dict

    def compute_model_cos_sims(self, model_dicts):
        for key in self.model.state_dict().keys():
            self.model_cos_sims.update({key: np.zeros((self.num_clients, self.num_clients))})

            for i in range(self.num_clients):
                for j in range(self.num_clients):
                    model_sim = tensor_cos_sim(model_dicts[i][key], model_dicts[j][key])
                    self.model_cos_sims[key][i][j] = model_sim

        for layer in self.layers:
            cos_sim = np.zeros((self.num_clients, self.num_clients))
            count = 0
            for key, val in self.model_cos_sims.items():
                if layer in key:
                    cos_sim = (cos_sim * count + val) / (count + 1)
                    count += 1
            cos_sim = json.dumps(np.around(cos_sim, decimals=4).tolist())
            logger.info(f'layer {layer} model cosine similarity matrix\n{cos_sim}')

    def compute_grad_cos_sims(self, grad_dicts):
        for key in self.model.state_dict().keys():
            self.grad_cos_sims.update({key: np.zeros((self.num_clients, self.num_clients))})

            for i in range(self.num_clients):
                for j in range(self.num_clients):
                    grad_sim = tensor_cos_sim(grad_dicts[i][key], grad_dicts[j][key])
                    self.grad_cos_sims[key][i][j] = grad_sim

        for layer in self.layers:
            cos_sim = np.zeros((self.num_clients, self.num_clients))
            count = 0
            for key, val in self.grad_cos_sims.items():
                if layer in key:
                    cos_sim = (cos_sim * count + val) / (count + 1)
                    count += 1
            cos_sim = json.dumps(np.around(cos_sim, decimals=4).tolist())
            logger.info(f'layer {layer} gradient cosine similarity matrix\n{cos_sim}')

    def compute_weights(self, select_clients):
        if self.aggregate_method == 'sample':
            weights = [self.num_samples[i] for i in select_clients]
            weights = np.array([w / sum(weights) for w in weights])
        else:
            n = len(select_clients)
            weights = np.array([1 / n for _ in range(n)])
        return weights

    def test_save_models(self, select_clients):
        tpwg_test_metrics = []
        features = []
        if self.tgwg:
            logger.info('test global test dataset with global model')
            model = copy.deepcopy(self.model)
            metrics, _ = self.eval_model(model, data_loader=self.central_client.eval_loader)
            tgwg_metric = metrics[self.m]
            wandb.log({'tgwg': tgwg_metric}, step=self.ite)

            if tgwg_metric > self.tgwg_best_metric:
                self.tgwg_best_metric = tgwg_metric
                torch.save(self.model.state_dict(), self.tgwg_ckpt)
                logger.info('new model saved')
            logger.info(f'best {self.m}: {self.tgwg_best_metric:.4f}')
            # for name, params in model.named_parameters():
            #     if name == 'encoder.classifier.weight':
            #         cosine_matrix = cosine_similarity(params.detach().numpy(), params.detach().numpy())
            #         print('classifier cosine similarity')
            #         print(cosine_matrix)

        for i in select_clients:
            model = copy.deepcopy(self.clients[i].model)
            data_loader = self.clients[i].eval_loader
            if self.tgwp:
                logger.info(f'test global test dataset with client {i} personal model')
                metrics, feature = self.eval_model(model=model)
                features.append(feature)
                if metrics[self.m] > self.tgwp_best_metrics[i]:
                    self.tgwp_best_metrics[i] = metrics[self.m]
                    torch.save(model.state_dict(), self.tgwp_ckpts[i])
                    logger.info('new model saved')
                logger.info(f'best {self.m}: {self.tgwp_best_metrics[i]:.4f}')

                for name, params in model.named_parameters():
                    if name == 'encoder.classifier.weight':
                        cosine_matrix = cosine_similarity(params.detach().numpy(), params.detach().numpy())
                        print('classifier cosine similarity')
                        print(cosine_matrix)

            if self.tpwg:
                logger.info(f'test personal {i} test dataset with global model')
                metrics, _ = self.eval_model(data_loader=data_loader)
                tpwg_test_metrics.append(metrics[self.m])

            if self.tpwp:
                logger.info(f'test personal {i} test dataset with client {i} personal model')
                metrics, _ = self.eval_model(model=model)
                if metrics[self.m] > self.tpwp_best_metrics[i]:
                    self.tpwp_best_metrics[i] = metrics[self.m]
                    torch.save(model.state_dict(), self.tpwp_ckpts[i])
                    logger.info('new model saved')
                logger.info(f'best {self.m}: {self.tpwp_best_metrics[i]:.4f}')

        if self.tgwp:
            tgwp_test_metrics = np.array(self.tgwp_best_metrics)
            best = np.max(tgwp_test_metrics)
            avg = np.mean(tgwp_test_metrics)
            worst = np.min(tgwp_test_metrics)
            std = np.std(tgwp_test_metrics)
            logger.info(f'best/avg/worst/std metrics for global test dataset: '
                        f'{best:.4f}/{avg:.4f}/{worst:.4f}/{std:.4f}')

        if self.tpwg:
            tpwg_test_metric = sum(tpwg_test_metrics) / len(tpwg_test_metrics)
            if tpwg_test_metric > self.tpwg_best_metric:
                self.tpwg_best_metric = tpwg_test_metric
                torch.save(self.model.state_dict(), self.tpwg_ckpt)
                logger.info('new global model for personal test datasets saved')
            logger.info(f'best {self.m} of global model for personal test datasets: {self.tpwg_best_metric:.4f}')

        if self.tpwp:
            tpwp_test_metrics = np.array(self.tpwp_best_metrics)
            best = np.max(tpwp_test_metrics)
            avg = np.mean(tpwp_test_metrics)
            worst = np.min(tpwp_test_metrics)
            std = np.std(tpwp_test_metrics)
            logger.info(f'best/avg/worst/std metrics for personal test datasets: '
                        f'{best:.4f}/{avg:.4f}/{worst:.4f}/{std:.4f}')
        return features
