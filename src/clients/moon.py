import numpy as np
import torch
import copy
from torch import optim, nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from .utils import BatchIterator, AverageMeter
from .base import BaseClient
import os
import logging
from transformers import DefaultDataCollator

logger = logging.getLogger(os.path.basename(__file__))
base_dir = os.path.expanduser('~/FedTransformers')


class MOONClient(BaseClient):
    def __init__(self, client_id, dataset, model, args):
        super(MOONClient, self).__init__(client_id, dataset, model, args)
        self.pre_model = copy.deepcopy(self.model)
        self.mu = args.mu
        self.temperature = args.temperature

    def train_model(self, ite=None):
        global_model = copy.deepcopy(self.model)
        global_model.cuda()

        pre_model = copy.deepcopy(self.pre_model)
        pre_model.cuda()

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
                nl = 0
                count = 0

                for (name1, param1), (name2, param2) in zip(model.named_parameters(),
                                                            global_model.named_parameters()):
                    nl = (nl * count + torch.norm(param1 - param2, 2)) / (count + 1)
                    count += 1

                d = OrderedDict(id=self.client_id)
                tmp_data = copy.deepcopy(data)
                for key, val in data.items():
                    tmp_data[key] = val.cuda()
                if ite is not None:
                    labels, features, logits, losses = model(tmp_data, ite)
                else:
                    labels, features, logits, losses = model(tmp_data)

                with torch.no_grad():
                    tmp_data = copy.deepcopy(data)
                    for key, val in data.items():
                        tmp_data[key] = val.cuda()
                    global_labels, global_features, global_logits, global_losses = global_model(tmp_data)
                    tmp_data = copy.deepcopy(data)
                    for key, val in data.items():
                        tmp_data[key] = val.cuda()
                    pre_labels, pre_features, pre_logits, pre_losses = pre_model(tmp_data)

                    pos_sim = F.cosine_similarity(features[-1], global_features[-1], dim=-1)
                    pos_sim = torch.exp(pos_sim / self.temperature)
                    neg_sim = F.cosine_similarity(features[-1], pre_features[-1], dim=-1)
                    neg_sim = torch.exp(neg_sim / self.temperature)

                    scl = -1.0 * torch.log(pos_sim / (pos_sim + neg_sim)).mean()

                cel = losses[0]
                losses.append(scl)
                loss = cel + self.mu * scl

                loss.backward()
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

        self.pre_model = copy.deepcopy(self.model)
        return sd, d
