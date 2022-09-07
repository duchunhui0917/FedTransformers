import numpy as np
from torch import optim, nn
from .base import BaseClient
from .utils import AverageMeter
from tqdm import tqdm
from collections import OrderedDict


class ICFAClient(BaseClient):
    def __init__(self, client_id, train_loader, test_loader, model, args):
        super(ICFAClient, self).__init__(client_id, train_loader, test_loader, model, args, )
        self.n_clusters = args.n_clusters
        self.cluster_models = args.cluster_models
        self.cluster_losses = np.zeros((self.n_clusters,))
        self.optimizers = [optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)
                           for model in self.cluster_models]

    def train_model(self):
        for c in range(self.n_clusters):
            model = nn.parallel.DataParallel(self.cluster_models[c])
            model.cuda()
            model.train()

            for epoch in range(self.n_epochs):
                avg_losses = None
                avg_metrics = None

                if self.n_batches == 0:
                    t = tqdm(self.train_loader, ncols=0)
                else:
                    ite_train_loader = BatchIterator(self.n_batches, self.train_loader)
                    t = tqdm(ite_train_loader, ncols=0)

                for i, data in enumerate(t):
                    d = OrderedDict(id=self.client_id)

                    self.optimizers[c].zero_grad()

                    inputs, labels = data
                    inputs, labels = [x.cuda() for x in inputs], [x.cuda() for x in labels]

                    features, logits, losses = model(inputs, labels)
                    losses[0].mean().backward()
                    self.optimizers[c].step()

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

                    t.set_postfix(d)
            self.cluster_models[c].to('cpu')
            self.cluster_losses[c] = avg_losses[0].avg
        best_c = np.argmin(self.cluster_losses)
        return self.cluster_models[best_c].state_dict(), best_c

    def receive_cluster_models(self, cluster_model_dicts):
        for c in range(self.n_clusters):
            cluster_model_dict = self.cluster_models[c].state_dict()
            for key, val in cluster_model_dicts[c].items():
                cluster_model_dict[key] = self.momentum * cluster_model_dict[key] + (1 - self.momentum) * val
            self.cluster_models[c].load_state_dict(cluster_model_dict)
