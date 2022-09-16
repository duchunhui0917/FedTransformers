import copy

import os

import math
import torch
import torch.nn
from torch import nn
from transformers import AutoConfig, AutoAdapterModel, AdapterConfig, AutoModelForSequenceClassification
import logging
from src.modules.common_modules import LSTM, max_pooling
from src.datasets.relation_extraction import parse
from .sequence_classification import (
    BertPrefixForSequenceClassification,
    RobertaPrefixForSequenceClassification,
    BertPromptForSequenceClassification,
    RobertaPromptForSequenceClassification
)
import torch.nn.functional as F
from src.modules.utils import sim, mce_loss, mcl_loss, dce_loss

logger = logging.getLogger(__name__)
base_dir = os.path.expanduser('~/FedTransformers')


class RelationExtractionModel(nn.Module):
    def __init__(self, model_args, dataset):
        super(RelationExtractionModel, self).__init__()
        self.dataset = dataset
        self.dropout = nn.Dropout()
        self.augment = model_args.augment
        self.temperature = 1

        model_name = model_args.model_name
        num_labels = model_args.num_labels
        tunning_name = model_args.tunning_method
        if tunning_name:
            logger.info(tunning_name)

        self.criterion = nn.CrossEntropyLoss()

        self.model_name = model_name

        w = torch.empty((num_labels, 768 * 2))
        b = torch.empty(num_labels)

        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(b, -bound, bound)

        self.proto = nn.Parameter(w, requires_grad=True)
        self.bias = nn.Parameter(b, requires_grad=True)

        if model_name == 'LSTM':
            self.encoder = LSTM()
            self.classifier = nn.Sequential(
                nn.Linear(670, 100),
                nn.ReLU(),
                nn.Linear(100, num_labels)
            )
        else:
            self.classifier = nn.Linear(768 * 2, num_labels)
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)

            if tunning_name == 'bottleneck_adapter':
                logger.info('tunning model with bottleneck adapter')
                self.encoder = AutoAdapterModel.from_pretrained(model_name, config=config)
                adapter_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16,
                                               non_linearity="relu")
                self.encoder.add_adapter(tunning_name, config=adapter_config)
                self.encoder.add_classification_head(tunning_name, num_labels=num_labels)
                self.encoder.train_adapter(tunning_name)
            elif tunning_name == 'P-Tunning_v2':
                config.pre_seq_len = model_args.pre_seq_len
                config.prefix_projection = model_args.prefix_projection
                config.prefix_hidden_size = model_args.prefix_hidden_size
                config.hidden_dropout_prob = model_args.hidden_dropout_prob
                if config.model_type == 'bert':
                    self.encoder = BertPrefixForSequenceClassification.from_pretrained(model_name, config=config)
                elif config.model_type == 'roberta':
                    self.encoder = RobertaPrefixForSequenceClassification.from_pretrained(model_name, config=config)
                else:
                    raise Exception
            elif tunning_name == 'P-Tunning':
                config.pre_seq_len = model_args.per_seq_len
                config.prefix_projection = model_args.prefix_projection
                config.prefix_hidden_size = model_args.prefix_hidden_size
                config.hidden_dropout_prob = model_args.hidden_dropout_prob
                if config.model_type == 'bert':
                    self.encoder = BertPromptForSequenceClassification.from_pretrained(model_name, config=config)
                elif config.model_type == 'roberta':
                    self.encoder = RobertaPromptForSequenceClassification.from_pretrained(model_name, config=config)
                else:
                    raise Exception
            elif tunning_name == 'from_scratch':
                self.encoder = AutoModelForSequenceClassification.from_config(config)
            else:
                self.encoder = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        total_param = 0
        training_param = 0
        for name, param in self.encoder.named_parameters():
            x = param.numel()
            if param.requires_grad:
                training_param += x
            total_param += x
        freeze_param = total_param - training_param
        logger.info(f'total params: {total_param}, training params: {training_param}, freeze params: {freeze_param}')

    def forward(self, data, args=None):
        if self.model_name == 'LSTM':
            embeddings = data['embeddings']
            label = data['label']
            output, (h_n, c_n) = self.encoder(embeddings)
            x = torch.mean(h_n, dim=0)
            logits = self.classifier(x)
            loss = self.criterion(logits, label)
            return [label], [output], [logits], [loss]
        else:
            e1_mask = data.pop('e1_mask')
            e2_mask = data.pop('e2_mask')
            labels = data.pop('labels')

            if self.augment == 'gradient_aug':
                positive_input_ids = data.pop('positive_mask_input_ids')
                negative_input_ids = data.pop('negative_mask_input_ids')
                with torch.no_grad():
                    positive_data = copy.deepcopy(data)
                    positive_data['input_ids'] = positive_input_ids
                    positive_outputs = self.encoder(**positive_data, output_hidden_states=True)
                    positive_output = positive_outputs.hidden_states[-1]
                    e1_h = max_pooling(positive_output, e1_mask)  # (B, H)
                    e2_h = max_pooling(positive_output, e2_mask)  # (B, H)
                    positive_ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)

                    negative_data = copy.deepcopy(data)
                    negative_data['input_ids'] = negative_input_ids
                    negative_outputs = self.encoder(**negative_data, output_hidden_states=True)
                    negative_output = negative_outputs.hidden_states[-1]
                    e1_h = max_pooling(negative_output, e1_mask)  # (B, H)
                    e2_h = max_pooling(negative_output, e2_mask)  # (B, H)
                    negative_ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
            elif self.augment == 'prototype_aug':
                path = os.path.join(base_dir, f'data/i2b2/prototypes.txt')
                with open(path, 'r') as f:
                    proto_texts = f.readlines()
                proto_data = {'label': [0, 1, 2, 3, 4, 5, 6, 7]}
                proto_data = parse(proto_texts, proto_data, self.dataset.tokenizer)
                for k, v in proto_data.items():
                    proto_data[k] = torch.LongTensor(v).cuda()

                proto_e1_mask = proto_data.pop('e1_mask')
                proto_e2_mask = proto_data.pop('e2_mask')
                proto_labels = proto_data.pop('labels')
                proto_data.pop('label')

                proto_outputs = self.encoder(**proto_data, output_hidden_states=True)
                proto_output = proto_outputs.hidden_states[-1]
                proto_e1_h = max_pooling(proto_output, proto_e1_mask)  # (B, H)
                proto_e2_h = max_pooling(proto_output, proto_e2_mask)  # (B, H)
                proto_ent = torch.cat([proto_e1_h, proto_e2_h], dim=-1)  # (B, 2H)

                proto_logits = self.classifier(proto_ent)
                proto_loss = self.criterion(proto_logits, proto_labels)

            outputs = self.encoder(**data, output_hidden_states=True)
            encoder_output = outputs.hidden_states[-1]
            # extract entity
            e1_h = max_pooling(encoder_output, e1_mask)  # (B, H)
            e2_h = max_pooling(encoder_output, e2_mask)  # (B, H)
            ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
            ent = self.dropout(ent)

            # classifier
            logits = self.classifier(ent)  # (B, C)
            # logits = F.linear(ent, self.proto, self.bias)
            loss = self.criterion(logits, labels)

            dce = dce_loss(ent, labels, self.proto)
            # proto_loss = mce_loss(ent, labels, self.proto)

            if self.training:
                if self.augment == 'gradient_aug':
                    pos_sim = F.cosine_similarity(ent, positive_ent, dim=-1)
                    pos_sim = torch.exp(pos_sim / self.temperature)
                    neg_sim = F.cosine_similarity(ent, negative_ent, dim=-1)
                    neg_sim = torch.exp(neg_sim / self.temperature)
                    contrastive_loss = - torch.log(pos_sim / (pos_sim + neg_sim)).mean()

                    ce_loss = loss
                    loss = ce_loss + contrastive_loss
                    return [labels], [ent], [logits], [loss, ce_loss, contrastive_loss, pos_sim.mean(), neg_sim.mean()]
                # elif self.augment == 'prototype_aug':
                #     # instance-prototype loss
                #     sim_ins_proto = self.sim(ent, proto_ent)
                #     num = sim_ins_proto.shape[0]
                #     loss_ins_proto = 0.
                #     for i in range(num):
                #         pos_score = 1 - sim_ins_proto[i][labels[i]]
                #         loss_ins_proto += pos_score
                #     loss_ins_proto /= num
                #
                #     ce_loss = loss
                #     loss = ce_loss + 0.1 * proto_loss + loss_ins_proto
                #     return [labels], [ent], [logits], [loss, ce_loss, proto_loss, loss_ins_proto]
                elif self.augment == 'prototype_aug':
                    logits =
                    if args and 'proto_ent' in args:
                        proto_ent = args['proto_ent']
                    proto_dce = dce_loss(proto_ent, proto_labels, self.proto)
                    ce_loss = loss
                    loss = ce_loss + proto_dce
                    return [labels], [ent, proto_ent], [logits], [loss, ce_loss, proto_dce]

            return [labels], [ent], [logits], [loss]
