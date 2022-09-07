import numpy as np
import torch
import copy
from torch import optim, nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict
from .utils import BatchIterator, AverageMeter
import os
import logging
from transformers import DefaultDataCollator

logger = logging.getLogger(os.path.basename(__file__))
base_dir = os.path.expanduser('~/FedTransformers')


class BaseClient(object):
    def __init__(self, client_id, dataset, model, args):
        self.client_id = client_id
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.num_batches = args.num_batches
        self.momentum = args.client_momentum
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.test_batch_size = args.test_batch_size

        self.dataset = dataset
        self.task_name = dataset.task_name
        if client_id is 'None':
            train_dataset = dataset.train_dataset
            eval_dataset = dataset.eval_dataset
            test_dataset = dataset.test_dataset
        else:
            train_dataset = dataset.train_datasets[client_id]
            eval_dataset = dataset.eval_datasets[client_id]
            test_dataset = dataset.test_datasets[client_id]

        self.train_loader = dataset.get_loader(train_dataset, self.train_batch_size, shuffle=True)
        self.eval_loader = dataset.get_loader(eval_dataset, self.eval_batch_size)
        self.test_loader = dataset.get_loader(test_dataset, self.test_batch_size)

        self.model = copy.deepcopy(model)

        self.optimizer_sd = None
        self.optimizer_sd_path = os.path.join(base_dir, f'ckpt/state_dict/{self.task_name}_{client_id}')
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

    def train_model(self, ite=None):
        if not self.train_loader:
            return copy.deepcopy(self.model.state_dict()), None

        model = self.model.cuda()
        if self.optimizer_sd:
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
            if self.task_name == 'shakespeare':
                self.optimizer_sd = torch.load(self.optimizer_sd_path)
            state = self.optimizer_sd['state']
            for s in state.keys():
                for k, v in state[s].items():
                    if isinstance(v, torch.Tensor):
                        state[s][k] = v.cuda()
            self.optimizer.load_state_dict(self.optimizer_sd)
        model.train()

        for epoch in range(self.num_epochs):
            avg_losses = None
            avg_metrics = None

            if self.num_batches == 0:
                t = tqdm(self.train_loader, ncols=0)
            else:
                ite_train_loader = BatchIterator(self.num_batches, self.train_loader)
                t = tqdm(ite_train_loader, ncols=0)

            for i, data in enumerate(t):
                d = OrderedDict(id=self.client_id)
                for key, val in data.items():
                    data[key] = val.cuda()
                if ite is not None:
                    labels, features, logits, losses = model(data, ite)
                else:
                    labels, features, logits, losses = model(data)
                losses[0].backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                labels = [label.cpu().detach().numpy() for label in labels]
                logits = [logit.cpu().detach().numpy() for logit in logits]
                metrics = self.dataset.compute_metrics(labels, logits)

                if avg_metrics is None:
                    avg_metrics = {key: AverageMeter() for key in metrics.keys()}
                for key, val in metrics.items():
                    avg_metric = avg_metrics[key]
                    avg_metric.update(val, 1)
                    d.update({key: avg_metric.avg})

                if avg_losses is None:
                    avg_losses = [AverageMeter() for _ in range(len(losses))]

                for idx, loss in enumerate(losses):
                    avg_loss = avg_losses[idx]
                    avg_loss.update(loss.mean().item(), 1)
                    d.update({f'loss{idx}': avg_loss.avg})

                t.set_postfix(d)

            for key, val in d.items():
                if key == 'id':
                    logger.info(f'{key}: {val}')
                else:
                    logger.info(f'{key}: {val:.4f}')

        model.cpu()
        optimizer_sd = self.optimizer.state_dict()
        state = optimizer_sd['state']
        for s in state.keys():
            for k, v in state[s].items():
                if isinstance(v, torch.Tensor):
                    state[s][k] = v.cpu()
        if self.task_name == 'shakespeare':
            torch.save(optimizer_sd, self.optimizer_sd_path)
            self.optimizer_sd = True
        else:
            self.optimizer_sd = optimizer_sd

        del self.optimizer

        torch.cuda.empty_cache()
        sd = copy.deepcopy(model.state_dict())
        d.pop('id')
        return sd, d

    def receive_global_model(self, global_model_dict):
        local_model_dict = self.model.state_dict()
        for key, val in global_model_dict.items():
            local_model_dict[key] = self.momentum * local_model_dict[key] + (1 - self.momentum) * val
        self.model.load_state_dict(local_model_dict)
