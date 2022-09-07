import copy
import numpy as np
from torch.utils.data import DataLoader
import logging
import os
from tqdm import tqdm
import torch
from ..clients.base import BaseClient

logger = logging.getLogger(os.path.basename(__file__))


class Base(object):
    def __init__(self, dataset, model, args):
        self.dataset = dataset
        self.model = copy.deepcopy(model)

        self.num_clients = args.num_clients
        self.num_iterations = args.num_iterations
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.n_batches = args.num_batches
        self.ckpt = args.ckpt
        self.test_frequency = args.test_frequency
        self.tgwg = args.tgwg
        self.tgwp = args.tgwp
        self.tpwg = args.tpwg
        self.tpwp = args.tpwp

        self.tgwg_ckpt = f'{self.ckpt}_tgwg'
        self.tgwp_ckpts = [f'{self.ckpt}_client{i}_tgwp' for i in range(self.num_clients)]
        self.tpwg_ckpt = f'{self.ckpt}_tpwg'
        self.tpwp_ckpts = [f'{self.ckpt}_client{i}_tpwp' for i in range(self.num_clients)]

        self.tgwg_best_metric = 0
        self.tgwp_best_metrics = [0 for _ in range(self.num_clients)]
        self.tpwg_best_metric = 0
        self.tpwp_best_metrics = [0 for _ in range(self.num_clients)]

        self.ite = 0
        self.central_client = BaseClient('None', dataset, model, args)

        self.m = self.dataset.eval_metric

    def run(self):
        for self.ite in range(self.num_iterations):
            logger.info(f'********iteration: {self.ite}********')
            # train
            model_state_dict = self.central_client.train_model()
            self.model.load_state_dict(model_state_dict)
            # test
            self.test_save_models()

    def test_save_models(self):
        logger.info('test global test dataset with global model')
        model = copy.deepcopy(self.model)
        metrics = self.eval_model(model, data_loader=self.central_client.eval_loader)
        if metrics[self.m] > self.tgwg_best_metric:
            self.tgwg_best_metric = metrics[self.m]
            torch.save(self.model.state_dict(), self.tgwg_ckpt)
            logger.info('new model saved')
        logger.info(f'best {self.m}: {self.tgwg_best_metric:.4f}')

    def eval_model(self, model=None, data_loader=None):
        if model is None:
            model = self.model
        if data_loader is None:
            data_loader = self.central_client.eval_loader

        model = model.cuda()
        model.eval()

        res = {'loss': 0}

        t = tqdm(data_loader)

        for i, data in enumerate(t):
            for key, val in data.items():
                data[key] = val.cuda()

            with torch.no_grad():
                labels, features, logits, losses = model(data)

            labels = labels[0].cpu().numpy()
            logits = logits[0].cpu().numpy()
            loss = losses[0].mean().item()

            res['loss'] += loss

            tmp = self.dataset.compute_metrics(labels, logits)
            if len(res.keys()) == 1:
                for key in tmp.keys():
                    res[key] = tmp[key]
            else:
                for key in tmp.keys():
                    res[key] += tmp[key]

        for key, val in res.items():
            res[key] = val / len(data_loader)
            logger.info(f'{key}: {res[key]:.4f}')
        model.cpu()
        return res

    def load(self, ckpt, exclude_key=' '):
        sd = self.model.state_dict()
        for key, val in torch.load(ckpt).items():
            if exclude_key not in key:
                # logger.info(key)
                sd.update({key: val})
        self.model.load_state_dict(sd)
