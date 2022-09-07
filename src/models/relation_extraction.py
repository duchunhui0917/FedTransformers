import copy

import torch
import torch.nn
from torch import nn
from transformers import AutoConfig, AutoAdapterModel, AdapterConfig, AutoModelForSequenceClassification
import logging
from src.modules.common_modules import LSTM, max_pooling
from .sequence_classification import (
    BertPrefixForSequenceClassification,
    RobertaPrefixForSequenceClassification,
    BertPromptForSequenceClassification,
    RobertaPromptForSequenceClassification
)
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RelationExtractionModel(nn.Module):
    def __init__(self, model_args, dataset):
        super(RelationExtractionModel, self).__init__()
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

    def forward(self, data, ite=None):
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

            if self.augment:
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

            outputs = self.encoder(**data, output_hidden_states=True)
            encoder_output = outputs.hidden_states[-1]
            # extract entity
            e1_h = max_pooling(encoder_output, e1_mask)  # (B, H)
            e2_h = max_pooling(encoder_output, e2_mask)  # (B, H)
            ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
            ent = self.dropout(ent)

            # classifier
            logits = self.classifier(ent)  # (B, C)
            loss = self.criterion(logits, labels)

            if self.augment and self.training:
                pos_sim = F.cosine_similarity(ent, positive_ent, dim=-1)
                pos_sim = torch.exp(pos_sim / self.temperature)
                neg_sim = F.cosine_similarity(ent, negative_ent, dim=-1)
                neg_sim = torch.exp(neg_sim / self.temperature)
                contrastive_loss = - torch.log(pos_sim / (pos_sim + neg_sim)).mean()

                ce_loss = loss
                loss = ce_loss + contrastive_loss
                return [labels], [logits], [logits], [loss, ce_loss, contrastive_loss, pos_sim.mean(), neg_sim.mean()]

            return [labels], [logits], [logits], [loss]
