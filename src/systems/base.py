import copy
import json

import numpy as np
import wandb
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import logging
import os
from tqdm import tqdm
import torch
from ..clients.base import BaseClient
from torch.utils.tensorboard import SummaryWriter
from src.modules.common_modules import LSTM, max_pooling

base_dir = os.path.expanduser('~/FedTransformers')
logger = logging.getLogger(os.path.basename(__file__))
path = os.path.join(base_dir, 'log/tensorboard')
print(path)
writer = SummaryWriter(path)


class Base(object):
    def __init__(self, dataset, model, args):
        self.dataset = dataset
        self.model = copy.deepcopy(model)

        self.num_clients = args.num_clients
        self.num_iterations = args.num_iterations
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.num_batches = args.num_batches
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
            model_state_dict, scalars = self.central_client.train_model()
            wandb.log({'training loss': scalars['loss0']}, step=self.ite)

            self.model.load_state_dict(model_state_dict)

            # test
            self.test_save_models()
            model = copy.deepcopy(self.model)
            # metrics, features = self.eval_model(model, data_loader=self.central_client.train_loader)

    def test_save_models(self):
        logger.info('test global test dataset with global model')
        model = copy.deepcopy(self.model)
        metrics, features = self.eval_model(model, data_loader=self.central_client.eval_loader)
        tgwg_metric = metrics[self.m]
        wandb.log({'tgwg': tgwg_metric}, step=self.ite)

        if tgwg_metric > self.tgwg_best_metric:
            self.tgwg_best_metric = tgwg_metric
            torch.save(self.model.state_dict(), self.tgwg_ckpt)
            logger.info('new model saved')
        logger.info(f'best {self.m}: {self.tgwg_best_metric:.4f}')
        for name, params in model.named_parameters():
            if name == 'encoder.classifier.weight':
                cosine_matrix = cosine_similarity(params.detach().numpy(), params.detach().numpy())
                print('classifier cosine similarity')
                print(cosine_matrix)

        return features

    def eval_model(self, model=None, data_loader=None):
        if model is None:
            model = self.model
        if data_loader is None:
            data_loader = self.central_client.eval_loader

        model = model.cuda()
        model.eval()

        t = tqdm(data_loader)
        res_losses = []
        res_labels, res_logits, res_features = None, None, None
        for i, data in enumerate(t):
            for key, val in data.items():
                data[key] = val.cuda()

            with torch.no_grad():
                labels, features, logits, losses = model(data)

            if res_labels is None and res_logits is None:
                res_labels = [[] for _ in range(len(labels))]
                res_logits = [[] for _ in range(len(logits))]
            if res_features is None:
                res_features = [[] for _ in range(len(features))]

            for label, ls in zip(labels, res_labels):
                ls.append(label.cpu().numpy())
            for logit, ls in zip(logits, res_logits):
                ls.append(logit.cpu().numpy())
            for feature, ls in zip(features, res_features):
                ls.append(feature.cpu().numpy())

            loss = losses[0].mean().item()
            res_losses.append(loss)

        try:
            res_labels = [np.concatenate(labels) for labels in res_labels]
            res_logits = [np.concatenate(logits) for logits in res_logits]
            res_features = [np.concatenate(features) for features in res_features]
        except:
            pass
        res = self.dataset.compute_metrics(res_labels, res_logits)
        res.update({'loss': sum(res_losses) / len(res_losses)})
        for key, val in res.items():
            logger.info(f'{key}: {res[key]:.4f}')
        model.cpu()
        # self.compute_feature_sim(res_labels, res_logits, res_features)
        return res, res_features

    def get_tc_model_gradient_norm(self, model=None, data_loader=None):
        if model is None:
            model = self.model
        if data_loader is None:
            data_loader = self.central_client.train_loader

        model = model.cuda()
        model.eval()

        all_special_ids = self.dataset.tokenizer.all_special_ids
        word2id = self.dataset.tokenizer.vocab
        id2word = {v: k for k, v in word2id.items()}
        num_vocab = len(word2id)
        weight = {i: 0 for i in word2id.values()}
        frequency = {i: 0 for i in word2id.values()}

        t = tqdm(data_loader)

        for i, data in enumerate(t):
            input_ids = data['input_ids']
            attention_masks = data['attention_mask']
            labels = data['labels']
            for input_id, attention_mask, label in zip(input_ids, attention_masks, labels):

                input_id = input_id.unsqueeze(dim=0).cuda()
                attention_mask = attention_mask.unsqueeze(dim=0).cuda()
                outputs = model.encoder(input_ids=input_id, attention_mask=attention_mask, output_hidden_states=True)
                embedding = outputs['hidden_states'][0]
                logits = outputs['logits']
                embedding.retain_grad()
                label_logits = logits[0][label]
                label_logits.backward(retain_graph=True)
                grad = embedding.grad
                norm_grad = torch.norm(grad, dim=2, p=2)
                sum_norm_grad = norm_grad.sum()
                for n, i in zip(norm_grad[0], input_id[0]):
                    i = i.item()
                    n = n.item()
                    if i not in all_special_ids:
                        weight[i] += n
                        frequency[i] += 1
        norm_weight = {i: weight[i] / frequency[i] if frequency[i] != 0 else 0 for i in word2id.values()}
        sorted_norm_weight = sorted(norm_weight.items(), key=lambda d: d[1], reverse=True)
        word_norm_wight = [(id2word[i], w) for (i, w) in sorted_norm_weight]
        with open(
                os.path.join(base_dir, 'data/weight/AIMed_s233_800.json'), 'w'
        ) as f:
            json.dump(norm_weight, f)

    def get_re_model_gradient_norm(self, model=None, data_loader=None):
        if model is None:
            model = self.model
        if data_loader is None:
            data_loader = self.central_client.train_loader

        model = model.cuda()
        model.eval()

        all_special_ids = self.dataset.tokenizer.all_special_ids
        word2id = self.dataset.tokenizer.vocab
        id2word = {v: k for k, v in word2id.items()}
        num_vocab = len(word2id)
        weight = {i: 0 for i in word2id.values()}
        frequency = {i: 0 for i in word2id.values()}

        t = tqdm(data_loader)

        for i, data in enumerate(t):
            input_ids = data['input_ids']
            attention_masks = data['attention_mask']
            e1_masks = data['e1_mask']
            e2_masks = data['e2_mask']
            labels = data['labels']
            z = zip(input_ids, attention_masks, e1_masks, e2_masks, labels)
            for input_id, attention_mask, e1_mask, e2_mask, label in z:
                input_id = input_id.unsqueeze(dim=0).cuda()
                e1_mask = e1_mask.unsqueeze(dim=0).cuda()
                e2_mask = e2_mask.unsqueeze(dim=0).cuda()

                attention_mask = attention_mask.unsqueeze(dim=0).cuda()
                outputs = model.encoder(input_ids=input_id, attention_mask=attention_mask, output_hidden_states=True)
                output = outputs.hidden_states[-1]
                embedding = outputs.hidden_states[0]

                e1_h = max_pooling(output, e1_mask)  # (B, H)
                e2_h = max_pooling(output, e2_mask)  # (B, H)
                ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
                ent = model.dropout(ent)
                logits = model.classifier(ent)  # (B, C)

                embedding.retain_grad()
                label_logits = logits[0][label]
                label_logits.backward(retain_graph=True)
                grad = embedding.grad
                norm_grad = torch.norm(grad, dim=2, p=2)
                sum_norm_grad = norm_grad.sum()
                for n, i in zip(norm_grad[0], input_id[0]):
                    i = i.item()
                    n = n.item()
                    if i not in all_special_ids:
                        weight[i] += n
                        frequency[i] += 1
        norm_weight = {i: weight[i] / frequency[i] if frequency[i] != 0 else 0 for i in word2id.values()}
        sorted_norm_weight = sorted(norm_weight.items(), key=lambda d: d[1], reverse=True)
        word_norm_wight = [(id2word[i], w) for (i, w) in sorted_norm_weight]
        with open(
                os.path.join(base_dir, 'data/weight/BioInfer_s233_800.json'), 'w'
        ) as f:
            json.dump(norm_weight, f)

    def load(self, ckpt, exclude_key=' '):
        sd = self.model.state_dict()
        ckpt_sd = torch.load(ckpt)
        for key, val in sd.items():
            if exclude_key not in key and key in ckpt_sd:
                sd[key] = ckpt_sd[key]
            else:
                logger.info(key)

        self.model.load_state_dict(sd)

    def compute_feature_sim(self, res_labels, res_logits, res_features):
        num_labels = self.dataset.num_labels
        labels = res_labels[0]
        logits = res_logits[0]
        features = res_features[0]
        logger.info('feature sim')
        ls = [features[labels == l].mean(axis=0) for l in range(num_labels)]
        s = '\n'
        for i in range(num_labels):
            tmp = []
            for j in range(num_labels):
                d = np.dot([ls[i]], ls[j].T)  # 向量点乘
                n = np.linalg.norm(ls[i]) * np.linalg.norm(ls[j])  # 求模长的乘积
                sim = float(d / n)
                tmp.append(sim)
            tmp.append(sum(tmp) / len(tmp))
            for x in tmp:
                s += f'{x:.4f} '
            s += '\n'
        logger.info(s)

        logger.info('logits sim')
        ls = [logits[labels == l].mean(axis=0) for l in range(num_labels)]
        s = '\n'
        for i in range(num_labels):
            tmp = []
            for j in range(num_labels):
                d = np.dot([ls[i]], ls[j].T)  # 向量点乘
                n = np.linalg.norm(ls[i]) * np.linalg.norm(ls[j])  # 求模长的乘积
                sim = float(d / n)
                tmp.append(sim)
            tmp.append(sum(tmp) / len(tmp))
            for x in tmp:
                s += f'{x:.4f} '
            s += '\n'
        logger.info(s)
