from .base import BaseClient
from torch import nn
from collections import OrderedDict
from .utils import AverageMeter
import logging
import os

logger = logging.getLogger(os.path.basename(__file__))


class MTLClient(BaseClient):
    def __init__(self, client_id, dataset, model, args):
        super(MTLClient, self).__init__(client_id, dataset, model, args)

    def train_model(self, ite=None):
        model = nn.parallel.DataParallel(self.model)
        model.cuda()
        model.train()

        for epoch in range(self.num_epochs):
            avg = [[None, None] for _ in range(len(self.train_loaders))]
            ds = [None, None]
            train_loaders = [[data for data in loader] for loader in self.train_loaders]
            ls = list(zip(*train_loaders))
            for loaders in ls:
                for i, data in enumerate(loaders):
                    avg_metrics, avg_losses = avg[i]
                    ds[i] = OrderedDict(id=i)
                    d = ds[i]

                    self.optimizer.zero_grad()

                    inputs, labels = data
                    inputs, labels = [x.cuda() for x in inputs], [x.cuda() for x in labels]
                    inputs += [i]
                    if ite is not None:
                        features, logits, losses = model(inputs, labels, ite)
                    else:
                        features, logits, losses = model(inputs, labels)
                    losses[0].backward()

                    self.optimizer.step()

                    metrics = self.train_dataset.metric(inputs, labels, logits, test_mode=False)

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

            for d in ds:
                logger.info(d)
        self.model.to('cpu')
        return self.model.state_dict()
