import copy

import torch
import torch.nn
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import (
    AutoAdapterModel,
    AdapterConfig,
    PrefixTuningConfig,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM
)
from transformers import BertModel, BertPreTrainedModel
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import logging
from src.modules.common_modules import LSTM, PrefixEncoder
from openprompt import PromptForClassification
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SequenceClassificationModel(nn.Module):
    def __init__(self, model_args, dataset):
        super(SequenceClassificationModel, self).__init__()
        self.dataset = dataset
        self.augment = model_args.augment
        self.temperature = 1
        model_name = model_args.model_name
        num_labels = model_args.num_labels
        tunning_method = model_args.tunning_method
        prompt_method = model_args.prompt_method
        if tunning_method:
            logger.info(tunning_method)
        if prompt_method:
            logger.info(prompt_method)
        self.tokenized_anchor_texts = None
        self.verbalizer = None
        self.prompt = dataset.prompt

        self.verbalizer = dataset.verbalizer
        self.tokenized_anchor_texts = dataset.tokenized_anchor_texts

        self.model_name = model_name
        self.criterion = nn.CrossEntropyLoss()

        if model_name == 'LSTM':
            self.encoder = LSTM()
            self.classifier = nn.Sequential(
                nn.Linear(670, 100),
                nn.ReLU(),
                nn.Linear(100, num_labels)
            )
        else:
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)

            if tunning_method == 'bottleneck_adapter':
                self.encoder = AutoAdapterModel.from_pretrained(model_name, config=config)
                adapter_config = AdapterConfig(mh_adapter=True,
                                               output_adapter=True,
                                               reduction_factor=16,
                                               non_linearity="relu")
                self.encoder.add_adapter(tunning_method, config=adapter_config)
                if prompt_method:
                    self.encoder.add_masked_lm_head(tunning_method)
                    for name, param in self.encoder.named_parameters():
                        if 'head' in name:
                            param.requires_grad = False
                else:
                    self.encoder.add_classification_head(tunning_method, num_labels=num_labels)

                self.encoder.train_adapter(tunning_method)
            elif tunning_method == 'prefix_tunning':
                self.encoder = AutoAdapterModel.from_pretrained(model_name)
                adapter_config = PrefixTuningConfig(flat=True,
                                                    prefix_length=32)
                self.encoder.add_adapter(tunning_method, config=adapter_config)
                # self.encoder.eject_prefix_tuning(tunning_method)
                if prompt_method:
                    self.encoder.add_masked_lm_head(tunning_method)
                    for name, param in self.encoder.named_parameters():
                        if 'head' in name:
                            param.requires_grad = False
                else:
                    self.encoder.add_classification_head(tunning_method, num_labels=num_labels)

                self.encoder.train_adapter(tunning_method)
            elif tunning_method == 'from_scratch':
                if prompt_method:
                    self.encoder = AutoModelForMaskedLM.from_config(config)
                else:
                    self.encoder = AutoModelForSequenceClassification.from_config(config)
            elif tunning_method == 'frozen':
                if prompt_method:
                    self.encoder = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                else:
                    self.encoder = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
                    for param in self.encoder.bert.parameters():
                        param.requires_grad = False
            else:
                if prompt_method:
                    self.encoder = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
                    # for name, param in self.encoder.named_parameters():
                    #     if 'cls' in name:
                    #         param.requires_grad = False
                else:
                    self.encoder = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
                    # for name, param in self.encoder.named_parameters():
                    #     if 'classifier' in name:
                    #         param.requires_grad = False

        total_param = 0
        training_param = 0
        for name, param in self.encoder.named_parameters():
            # print(name)
            # print(param.size())

            x = param.numel()
            if param.requires_grad:
                training_param += x
            total_param += x
        freeze_param = total_param - training_param
        logger.info(
            f'encoder total params: {total_param}, training params: {training_param}, freeze params: {freeze_param}')

        if self.verbalizer:
            total_param = 0
            training_param = 0
            for name, param in self.verbalizer.named_parameters():
                x = param.numel()
                if param.requires_grad:
                    training_param += x
                total_param += x
            freeze_param = total_param - training_param
            logger.info(
                f'verbalizer total params: {total_param}, training params: {training_param}, freeze params: {freeze_param}')

        self.anchor_encoder = copy.deepcopy(self.encoder)
        for param in self.anchor_encoder.parameters():
            param.requires_grad = False

    def forward(self, data, ite=None):
        self.anchor_outputs = None
        if self.tokenized_anchor_texts:
            for key, val in self.tokenized_anchor_texts.items():
                self.tokenized_anchor_texts[key] = val.cuda()
            static_anchor_outputs = self.anchor_encoder(**self.tokenized_anchor_texts, output_hidden_states=True)

        if self.prompt:
            inverse_labels = data.pop("labels")
            outputs = self.encoder(**data, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            if self.tokenized_anchor_texts:
                dynamic_anchor_outputs = self.encoder(**self.tokenized_anchor_texts, output_hidden_states=True)

                logits, loss, feature = self.verbalizer(outputs, data, inverse_labels, self.training,
                                                        static_anchor_outputs=static_anchor_outputs,
                                                        dynamic_anchor_outputs=dynamic_anchor_outputs,
                                                        anchor_batch=self.tokenized_anchor_texts)
            else:
                logits, loss, feature = self.verbalizer(outputs, data, inverse_labels, self.training)

            return [inverse_labels], [feature, logits], [logits], loss
        if self.model_name == 'LSTM':
            embeddings = data['embeddings']
            label = data['label']
            output, (h_n, c_n) = self.encoder(embeddings)
            x = torch.mean(h_n, dim=0)
            logits = self.classifier(x)
            loss = self.criterion(logits, label)
            return [label], [output], [logits], [loss]
        else:
            if self.augment:
                # positive_input_ids = data.pop('positive_input_ids')
                positive_mask_input_ids = data.pop('positive_mask_input_ids')
                # negative_input_ids = data.pop('negative_input_ids')
                negative_mask_input_ids = data.pop('negative_mask_input_ids')

                with torch.no_grad():
                    labels = data['labels']
                    b = labels.shape[0]
                    inverse_labels = torch.LongTensor([1 - l for l in labels]).cuda()

                    positive_mask_outputs = self.encoder(positive_mask_input_ids, output_hidden_states=True)
                    positive_mask_hidden = positive_mask_outputs.hidden_states[-1]
                    positive_mask_feature = positive_mask_hidden[:, 0]

                    negative_mask_outputs = self.encoder(negative_mask_input_ids, output_hidden_states=True)
                    negative_mask_hidden = negative_mask_outputs.hidden_states[-1]
                    negative_mask_feature = negative_mask_hidden[:, 0]

                    # mask_outputs1 = self.encoder(mask_input_ids1, output_hidden_states=True)
                    # mask_hidden1 = mask_outputs1.hidden_states[-1]
                    # mask_feature1 = mask_hidden1[:, 0]
                    # mask_logits1 = mask_outputs1.logits
                    #
                    # mask_logits1 = F.log_softmax(mask_logits1, dim=-1)
                    # uniform_logits = F.softmax(torch.ones((b, 2)) * 0.5, dim=-1).cuda()
                    # criterion = nn.KLDivLoss()
                    # kl_loss = criterion(mask_logits1, uniform_logits)
                    #
                    # mask_outputs2 = self.encoder(mask_input_ids2, labels=labels, output_hidden_states=True)
                    # mask_loss = mask_outputs2.loss

            outputs = self.encoder(**data, output_hidden_states=True)
            logits = outputs.logits
            loss = outputs.loss
            hidden = outputs.hidden_states[-1]
            feature = hidden[:, 0]
            if self.augment and self.training:
                pos_sim = F.cosine_similarity(feature, positive_mask_feature, dim=-1)
                pos_sim = torch.exp(pos_sim / self.temperature)
                neg_sim = F.cosine_similarity(feature, negative_mask_feature, dim=-1)
                neg_sim = torch.exp(neg_sim / self.temperature)
                contrastive_loss = - torch.log(pos_sim / (pos_sim + neg_sim)).mean()

                ce_loss = loss

                # loss = ce_loss + contrastive_loss
                # return [data['labels']], [feature, logits], [logits], [loss, ce_loss, pos_sim.mean(), neg_sim.mean()]

                # loss = ce_loss + negative_loss
                # return [data['labels']], [feature, logits], [logits], [loss, ce_loss, negative_loss]

                loss = ce_loss + contrastive_loss
                return [data['labels']], [feature, logits], [logits], [loss, ce_loss, contrastive_loss,
                                                                       pos_sim.mean(), neg_sim.mean()]

            return [data['labels']], [feature, logits], [logits], [loss]


class BertPrefixForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertPromptForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.embeddings = self.bert.embeddings
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        prompts = self.get_prompt(batch_size=batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            # input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # past_key_values=past_key_values,
        )

        # pooled_output = outputs[1]
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        first_token_tensor = sequence_output[:, 0]
        pooled_output = self.bert.pooler.dense(first_token_tensor)
        pooled_output = self.bert.pooler.activation(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaPrefixForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaPromptForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.embeddings = self.roberta.embeddings
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        prompts = self.get_prompt(batch_size=batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        # print(input_embeddings.shape)
        # exit()
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            # input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # past_key_values=past_key_values,
        )

        # pooled_output = outputs[1]
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        first_token_tensor = sequence_output[:, 0]
        pooled_output = self.roberta.pooler.dense(first_token_tensor)
        pooled_output = self.roberta.pooler.activation(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
