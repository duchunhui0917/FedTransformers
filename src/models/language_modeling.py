import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import AutoAdapterModel, PrefixTuningConfig
import logging

logger = logging.getLogger(__name__)


class LanguageModelingModel(nn.Module):
    def __init__(self, model_args, dataset):
        super(LanguageModelingModel, self).__init__()
        model_name = model_args.model_name
        tunning_name = model_args.tunning_method
        if tunning_name:
            logger.info(tunning_name)

        self.model_name = model_name
        if model_name == 'LSTM':
            pass
        else:
            if tunning_name == 'prefix_tunning':
                self.encoder = AutoAdapterModel.from_pretrained(model_name)
                adapter_config = PrefixTuningConfig(flat=False, prefix_length=30)
                self.encoder.add_adapter(tunning_name, config=adapter_config)
                self.encoder.eject_prefix_tuning(tunning_name)
                self.encoder.add_causal_lm_head(tunning_name)
                self.encoder.train_adapter(tunning_name)
            else:
                self.encoder = AutoModelForCausalLM.from_pretrained(model_name)

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
            outputs = self.encoder(**data)
            logits = outputs.logits
            loss = outputs.loss
            hidden_states = [logits]
            return [data['labels']], hidden_states, [logits], [loss]
