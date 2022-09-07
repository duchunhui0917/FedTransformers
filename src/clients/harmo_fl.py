from .base import BaseClient
from torch import nn
from .utils import AverageMeter
from torch import optim


class HarmoFLClient(BaseClient):
    def __init__(self, client_id, train_loader, test_loader, model, args):
        super(HarmoFLClient, self).__init__(client_id, train_loader, test_loader, model, args, )
        self.perturbation = args.perturbation
        self.optimizer = WPOptim(params=self.model.parameters(), base_optimizer=optim.Adam, lr=self.lr,
                                 alpha=self.perturbation, weight_decay=1e-4)

    def train_model(self, ite=None):
        model = nn.parallel.DataParallel(self.model)
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

                self.optimizer.zero_grad()

                inputs, labels = data
                inputs, labels = [x.cuda() for x in inputs], [x.cuda() for x in labels]

                features, logits, losses = model(inputs, labels)
                # compute the gradient
                losses[0].mean().backward()
                # normalize the gradient and add it to the parameters
                self.optimizer.generate_delta(zero_grad=True)

                features, logits, losses = model(inputs, labels)
                # compute the gradient of the parameters with perturbation
                losses[0].mean().backward()
                # gradient descent
                self.optimizer.step(zero_grad=True)

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
        self.model.to('cpu')
        return self.model.state_dict()
