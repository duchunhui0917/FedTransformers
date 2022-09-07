import json
from abc import abstractmethod
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput, CausalLMOutputWithCrossAttentions, MaskedLMOutput
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.generic import ModelOutput
from openprompt import Verbalizer
from openprompt.data_utils import InputFeatures
import re
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os

logger = logging.getLogger(os.path.basename(__file__))


class ManualVerbalizer(Verbalizer):
    r"""
    The basic manually defined verbalizer class, this class is inherited from the :obj:`Verbalizer` class.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 model: Optional[PreTrainedModel],
                 label_vectors: Optional[torch.tensor] = None,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "mean",
                 post_log_softmax: Optional[bool] = True,
                 ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.label_words = label_words
        self.post_log_softmax = post_log_softmax
        self.criterion = nn.CrossEntropyLoss()

    def on_label_words_set(self):
        super().on_label_words_set()
        self.label_words = self.add_prefix(self.label_words, self.prefix)

        # TODO should Verbalizer base class has label_words property and setter?
        # it don't have label_words init argument or label words from_file option at all

        self.generate_parameters()

    @staticmethod
    def add_prefix(label_words, prefix):
        r"""Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.

        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]  # wrapped it to a list of list of label words.

        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)

        max_len = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        words_ids_mask = [[[1] * len(ids) + [0] * (max_len - len(ids)) for ids in ids_per_label]
                          + [[0] * max_len] * (max_num_label_words - len(ids_per_label))
                          for ids_per_label in all_ids]
        words_ids = [[ids + [0] * (max_len - len(ids)) for ids in ids_per_label]
                     + [[0] * max_len] * (max_num_label_words - len(ids_per_label))
                     for ids_per_label in all_ids]

        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False)  # A 3-d mask
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)

    def project(self,
                logits: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        r"""
        Project the labels, the return value is the normalized (sum to 1) probs of label words.

        Args:
            logits (:obj:`torch.Tensor`): The orginal logits of label words.

        Returns:
            :obj:`torch.Tensor`: The normalized logits of label words
        """

        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000 * (1 - self.label_words_mask)
        return label_words_logits

    def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        (4) Aggregate (for multiple label words)

        Args:
            logits (:obj:`torch.Tensor`): The orginal logits.

        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        # project
        label_words_logits = self.project(logits,
                                          **kwargs)  # Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)

        if self.post_log_softmax:
            # normalize
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            label_words_logits = torch.log(label_words_probs + 1e-15)

        # aggregate
        label_logits = self.aggregate(label_words_logits)
        return label_logits

    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""Use weight to aggregate the logits of label words.

        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.

        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words.
        """
        label_words_logits = (label_words_logits * self.label_words_mask).sum(-1) / self.label_words_mask.sum(-1)
        return label_words_logits

    def calibrate(self, label_words_probs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""

        Args:
            label_words_probs (:obj:`torch.Tensor`): The probability distribution of the label words with the shape of [``batch_size``, ``num_classes``, ``num_label_words_per_class``]

        Returns:
            :obj:`torch.Tensor`: The calibrated probability of label words.
        """
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() == 1, "self._calibrate_logits are not 1-d tensor"
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] \
               and calibrate_label_words_probs.shape[0] == 1, "shape not match"
        label_words_probs /= (calibrate_label_words_probs + 1e-15)
        # normalize # TODO Test the performance
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1,
                                                           keepdim=True)  # TODO Test the performance of detaching()
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs

    def extract_at_mask(self,
                        outputs: torch.Tensor,
                        batch: Union[Dict, InputFeatures]):
        r"""Get outputs at all <mask> token
        E.g., project the logits of shape
        (``batch_size``, ``max_seq_length``, ``vocab_size``)
        into logits of shape (if num_mask_token > 1)
        (``batch_size``, ``num_mask_token``, ``vocab_size``)
        or into logits of shape (if ``num_mask_token`` = 1)
        (``batch_size``, ``vocab_size``).

        Args:
            outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
                 `gather_outputs` before) etc. of the whole sequence.
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.

        """
        mask_token_index = torch.where(batch["input_ids"] == 103)
        mask_token_logits = outputs[mask_token_index]
        return mask_token_logits

    def gather_outputs(self, outputs: ModelOutput):
        logits = outputs.logits
        if isinstance(outputs, Seq2SeqLMOutput):
            ret = outputs.decoder_hidden_states[-1]
        elif isinstance(outputs, MaskedLMOutput) or isinstance(outputs, CausalLMOutputWithCrossAttentions):
            ret = outputs.hidden_states[-1]
        else:
            try:
                ret = outputs.hidden_states[-1]
            except AttributeError:
                raise NotImplementedError(f"Gather outputs method for outputs' type {type(outputs)} not implemented")

        return ret, logits

    def forward(self, outputs: ModelOutput, batch: dict, labels: torch.tensor, training: bool, *args, **kwargs):
        ret, logits = self.gather_outputs(outputs)
        if isinstance(outputs, tuple):
            outputs_at_mask = [self.extract_at_mask(logit, batch) for logit in logits]
            features_at_mask = [self.extract_at_mask(r, batch) for r in ret]
        else:
            outputs_at_mask = self.extract_at_mask(logits, batch)
            features_at_mask = self.extract_at_mask(ret, batch)
        label_words_logits = self.process_logits(outputs_at_mask, batch=batch)
        loss = self.criterion(label_words_logits, labels)

        if 'static_anchor_outputs' in kwargs and 'dynamic_anchor_outputs' in kwargs and 'anchor_batch' in kwargs:
            static_anchor_outputs = kwargs['static_anchor_outputs']
            dynamic_anchor_outputs = kwargs['dynamic_anchor_outputs']
            anchor_batch = kwargs['anchor_batch']
            dynamic_anchor_ret, dynamic_anchor_logits = self.gather_outputs(dynamic_anchor_outputs)

            if isinstance(outputs, tuple):
                dynamic_anchor_outputs_at_mask = [self.extract_at_mask(logit, anchor_batch)
                                                  for logit in dynamic_anchor_logits]
            else:
                dynamic_anchor_outputs_at_mask = self.extract_at_mask(dynamic_anchor_logits, anchor_batch)
            dynamic_anchor_label_words_logits = self.process_logits(dynamic_anchor_outputs_at_mask, batch=batch)
            labels = torch.LongTensor(list(range(20))).cuda()
            anchor_loss = self.criterion(dynamic_anchor_label_words_logits, labels)
            loss += anchor_loss
            return label_words_logits, [loss]

        return label_words_logits, [loss], features_at_mask


class SoftVerbalizer(Verbalizer):
    r"""
    The implementation of the verbalizer in `Prototypical Verbalizer for Prompt-based Few-shot Tuning`

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
        lr: (:obj:`float`, optional): The learning rate for prototypes.
        mid_dim: (:obj:`int`, optional): The dimension of prototype embeddings.
        epochs: (:obj:`int`, optional): The training epochs of prototypes.
        multi_verb (:obj:`str`, optional): `multi` to ensemble with manual verbalizers, `proto` to use only ProtoVerb.
    """

    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizer],
                 model: Optional[PreTrainedModel],
                 label_vectors: Optional[torch.tensor] = None,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "mean",
                 post_log_softmax: Optional[bool] = True,
                 mid_dim: Optional[int] = 64,
                 epochs: Optional[int] = 5,
                 multi_verb: Optional[str] = "multi",
                 ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.post_log_softmax = post_log_softmax
        self.multi_verb = multi_verb
        self.mid_dim = mid_dim
        self.epochs = epochs
        self.trained = False

        self.hidden_dims = model.config.hidden_size
        self.head = torch.nn.Linear(self.hidden_dims, self.mid_dim, bias=False)

        if label_words is not None:  # use label words as an initialization
            self.label_words = label_words
        # w = torch.empty((self.num_classes, self.mid_dim))
        # w = torch.empty((self.num_classes, self.hidden_dims))
        # nn.init.xavier_uniform_(w)
        self.proto = nn.Parameter(label_vectors, requires_grad=True)

    def on_label_words_set(self):
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        self.generate_parameters()

    @staticmethod
    def add_prefix(label_words, prefix):
        r"""Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.

        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]  # wrapped it to a list of list of label words.

        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)

        max_len = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        words_ids_mask = [[[1] * len(ids) + [0] * (max_len - len(ids)) for ids in ids_per_label]
                          + [[0] * max_len] * (max_num_label_words - len(ids_per_label))
                          for ids_per_label in all_ids]
        words_ids = [[ids + [0] * (max_len - len(ids)) for ids in ids_per_label]
                     + [[0] * max_len] * (max_num_label_words - len(ids_per_label))
                     for ids_per_label in all_ids]

        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False)  # A 3-d mask
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)

    def process_hiddens(self, hiddens: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:
        """
        proto_logits = self.sim(self.head(hiddens), self.proto)
        return proto_logits

    def project(self,
                logits: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        r"""
        Project the labels, the return value is the normalized (sum to 1) probs of label words.

        Args:
            logits (:obj:`torch.Tensor`): The orginal logits of label words.

        Returns:
            :obj:`torch.Tensor`: The normalized logits of label words
        """

        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000 * (1 - self.label_words_mask)
        return label_words_logits

    def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        (4) Aggregate (for multiple label words)

        Args:
            logits (:obj:`torch.Tensor`): The orginal logits.

        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        # project
        label_words_logits = self.project(logits, **kwargs)
        # aggregate
        label_logits = self.aggregate(label_words_logits)
        return label_logits

    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""Use weight to aggregate the logits of label words.

        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.

        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words.
        """
        label_words_logits = (label_words_logits * self.label_words_mask).sum(-1) / self.label_words_mask.sum(-1)
        return label_words_logits

    def calibrate(self, label_words_probs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""

        Args:
            label_words_probs (:obj:`torch.Tensor`): The probability distribution of the label words with the shape of [``batch_size``, ``num_classes``, ``num_label_words_per_class``]

        Returns:
            :obj:`torch.Tensor`: The calibrated probability of label words.
        """
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() == 1, "self._calibrate_logits are not 1-d tensor"
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] \
               and calibrate_label_words_probs.shape[0] == 1, "shape not match"
        label_words_probs /= (calibrate_label_words_probs + 1e-15)
        # normalize # TODO Test the performance
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1,
                                                           keepdim=True)  # TODO Test the performance of detaching()
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs

    def ensemble_logits(self, manual_logits, proto_logits):

        logits = torch.stack([manual_logits, proto_logits])
        logits = logits.permute(1, 0, 2)
        logits = self.scaler(logits)
        logits = torch.mean(logits, 1)
        return logits

    @staticmethod
    def scaler(logits):
        m = logits.mean(-1, keepdim=True)
        s = logits.std(-1, keepdim=True)
        return (logits - m) / s

    def extract_at_mask(self,
                        outputs: torch.Tensor,
                        batch: Union[Dict, InputFeatures]):
        r"""Get outputs at all <mask> token
        E.g., project the logits of shape
        (``batch_size``, ``max_seq_length``, ``vocab_size``)
        into logits of shape (if num_mask_token > 1)
        (``batch_size``, ``num_mask_token``, ``vocab_size``)
        or into logits of shape (if ``num_mask_token`` = 1)
        (``batch_size``, ``vocab_size``).

        Args:
            outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
                 `gather_outputs` before) etc. of the whole sequence.
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.

        """
        mask_token_index = torch.where(batch["input_ids"] == 103)
        mask_token_logits = outputs[mask_token_index]
        return mask_token_logits

    def gather_outputs(self, outputs: ModelOutput):
        logits = outputs.logits
        if isinstance(outputs, Seq2SeqLMOutput):
            ret = outputs.decoder_hidden_states[-1]
        elif isinstance(outputs, MaskedLMOutput) or isinstance(outputs, CausalLMOutputWithCrossAttentions):
            ret = outputs.hidden_states[-1]
        else:
            try:
                ret = outputs.hidden_states[-1]
            except AttributeError:
                raise NotImplementedError(f"Gather outputs method for outputs' type {type(outputs)} not implemented")

        return ret, logits

    @staticmethod
    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1, 0))

    def pcl_loss(self, v_ins, labels, training):
        # instance-prototype loss
        sim_mat = torch.exp(self.sim(v_ins, self.proto))
        num = sim_mat.shape[0]
        loss = 0.
        for i in range(num):
            pos_score = sim_mat[i][labels[i]]
            neg_score = sim_mat[i].sum() - pos_score
            loss += - torch.log(pos_score / (pos_score + neg_score))
        loss /= num

        return sim_mat, loss

    def forward(self, outputs: ModelOutput, batch: Dict, labels: torch.tensor, training: bool, *args, **kwargs):
        ret, logits = self.gather_outputs(outputs)
        if isinstance(ret, tuple):
            outputs_at_mask = [self.extract_at_mask(hidden, batch) for hidden in ret]
        else:
            outputs_at_mask = self.extract_at_mask(ret, batch)
        # embedding = self.head(outputs_at_mask)
        embedding = outputs_at_mask
        logits, loss = self.pcl_loss(embedding, labels, training)
        return logits, [loss]


class ProtoVerbalizer(Verbalizer):
    r"""
    The implementation of the verbalizer in `Prototypical Verbalizer for Prompt-based Few-shot Tuning`

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
        lr: (:obj:`float`, optional): The learning rate for prototypes.
        mid_dim: (:obj:`int`, optional): The dimension of prototype embeddings.
        epochs: (:obj:`int`, optional): The training epochs of prototypes.
        multi_verb (:obj:`str`, optional): `multi` to ensemble with manual verbalizers, `proto` to use only ProtoVerb.
    """

    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizer],
                 model: Optional[PreTrainedModel],
                 label_vectors: Optional[torch.tensor] = None,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "mean",
                 post_log_softmax: Optional[bool] = True,
                 mid_dim: Optional[int] = 64,
                 epochs: Optional[int] = 5,
                 multi_verb: Optional[str] = "multi",
                 ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.post_log_softmax = post_log_softmax
        self.multi_verb = multi_verb
        self.mid_dim = mid_dim
        self.epochs = epochs
        self.trained = False

        self.hidden_dims = model.config.hidden_size
        self.head = torch.nn.Linear(self.hidden_dims, self.mid_dim, bias=False)

        if label_words is not None:  # use label words as an initialization
            self.label_words = label_words
        # w = torch.empty((self.num_classes, self.mid_dim))
        # w = torch.empty((self.num_classes, self.hidden_dims))
        # nn.init.xavier_uniform_(w)
        self.proto = nn.Parameter(label_vectors, requires_grad=True)

    def on_label_words_set(self):
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        self.generate_parameters()

    @staticmethod
    def add_prefix(label_words, prefix):
        r"""Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.

        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]  # wrapped it to a list of list of label words.

        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)

        max_len = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        words_ids_mask = [[[1] * len(ids) + [0] * (max_len - len(ids)) for ids in ids_per_label]
                          + [[0] * max_len] * (max_num_label_words - len(ids_per_label))
                          for ids_per_label in all_ids]
        words_ids = [[ids + [0] * (max_len - len(ids)) for ids in ids_per_label]
                     + [[0] * max_len] * (max_num_label_words - len(ids_per_label))
                     for ids_per_label in all_ids]

        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False)  # A 3-d mask
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)

    def process_hiddens(self, hiddens: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:
        """
        proto_logits = self.sim(self.head(hiddens), self.proto)
        return proto_logits

    def project(self,
                logits: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        r"""
        Project the labels, the return value is the normalized (sum to 1) probs of label words.

        Args:
            logits (:obj:`torch.Tensor`): The orginal logits of label words.

        Returns:
            :obj:`torch.Tensor`: The normalized logits of label words
        """

        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000 * (1 - self.label_words_mask)
        return label_words_logits

    def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        (4) Aggregate (for multiple label words)

        Args:
            logits (:obj:`torch.Tensor`): The orginal logits.

        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        # project
        label_words_logits = self.project(logits, **kwargs)
        # aggregate
        label_logits = self.aggregate(label_words_logits)
        return label_logits

    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""Use weight to aggregate the logits of label words.

        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.

        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words.
        """
        label_words_logits = (label_words_logits * self.label_words_mask).sum(-1) / self.label_words_mask.sum(-1)
        return label_words_logits

    def calibrate(self, label_words_probs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""

        Args:
            label_words_probs (:obj:`torch.Tensor`): The probability distribution of the label words with the shape of [``batch_size``, ``num_classes``, ``num_label_words_per_class``]

        Returns:
            :obj:`torch.Tensor`: The calibrated probability of label words.
        """
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() == 1, "self._calibrate_logits are not 1-d tensor"
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] \
               and calibrate_label_words_probs.shape[0] == 1, "shape not match"
        label_words_probs /= (calibrate_label_words_probs + 1e-15)
        # normalize # TODO Test the performance
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1,
                                                           keepdim=True)  # TODO Test the performance of detaching()
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs

    def ensemble_logits(self, manual_logits, proto_logits):

        logits = torch.stack([manual_logits, proto_logits])
        logits = logits.permute(1, 0, 2)
        logits = self.scaler(logits)
        logits = torch.mean(logits, 1)
        return logits

    @staticmethod
    def scaler(logits):
        m = logits.mean(-1, keepdim=True)
        s = logits.std(-1, keepdim=True)
        return (logits - m) / s

    def extract_at_mask(self,
                        outputs: torch.Tensor,
                        batch: Union[Dict, InputFeatures]):
        r"""Get outputs at all <mask> token
        E.g., project the logits of shape
        (``batch_size``, ``max_seq_length``, ``vocab_size``)
        into logits of shape (if num_mask_token > 1)
        (``batch_size``, ``num_mask_token``, ``vocab_size``)
        or into logits of shape (if ``num_mask_token`` = 1)
        (``batch_size``, ``vocab_size``).

        Args:
            outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
                 `gather_outputs` before) etc. of the whole sequence.
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.

        """
        mask_token_index = torch.where(batch["input_ids"] == 103)
        mask_token_logits = outputs[mask_token_index]
        return mask_token_logits

    def gather_outputs(self, outputs: ModelOutput):
        logits = outputs.logits
        if isinstance(outputs, Seq2SeqLMOutput):
            ret = outputs.decoder_hidden_states[-1]
        elif isinstance(outputs, MaskedLMOutput) or isinstance(outputs, CausalLMOutputWithCrossAttentions):
            ret = outputs.hidden_states[-1]
        else:
            try:
                ret = outputs.hidden_states[-1]
            except AttributeError:
                raise NotImplementedError(f"Gather outputs method for outputs' type {type(outputs)} not implemented")

        return ret, logits

    @staticmethod
    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1, 0))

    def pcl_loss(self, ins, labels, training):
        # instance-prototype loss
        sim_ins_proto = torch.exp(self.sim(ins, self.proto))
        num = sim_ins_proto.shape[0]
        loss_ins_proto = 0.
        for i in range(num):
            pos_score = sim_ins_proto[i][labels[i]]
            neg_score = sim_ins_proto[i].sum()
            loss_ins_proto += - torch.log(pos_score / neg_score)
        loss_ins_proto /= num

        if self.training:
            # instance-instance loss
            sim_ins_ins = torch.exp(self.sim(ins, ins))
            num = sim_ins_ins.shape[0]
            loss_ins_ins = 0.
            for i in range(num):
                pos_idx = list(set(torch.where(labels == labels[i])[0]))
                neg_idx = list(set(range(num)))
                l = len(pos_idx)
                neg_score = sim_ins_ins[i][neg_idx].sum()
                for j in pos_idx:
                    pos_score = sim_ins_ins[i][j]
                    loss_ins_ins += - torch.log(pos_score / neg_score) / l
            loss_ins_ins /= num

            loss = loss_ins_proto + loss_ins_ins
            return sim_ins_proto, [loss, loss_ins_proto, loss_ins_ins]

        return sim_ins_proto, [loss_ins_proto]

    def forward(self, outputs: ModelOutput, batch: Dict, labels: torch.tensor, training: bool, *args, **kwargs):
        ret, logits = self.gather_outputs(outputs)
        if isinstance(ret, tuple):
            outputs_at_mask = [self.extract_at_mask(hidden, batch) for hidden in ret]
        else:
            outputs_at_mask = self.extract_at_mask(ret, batch)
        # embedding = self.head(outputs_at_mask)
        embedding = outputs_at_mask
        logits, loss = self.pcl_loss(embedding, labels, training)
        return logits, loss


class AnchorVerbalizer(Verbalizer):
    r"""
    The implementation of the verbalizer in `Prototypical Verbalizer for Prompt-based Few-shot Tuning`

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
        lr: (:obj:`float`, optional): The learning rate for prototypes.
        mid_dim: (:obj:`int`, optional): The dimension of prototype embeddings.
        epochs: (:obj:`int`, optional): The training epochs of prototypes.
        multi_verb (:obj:`str`, optional): `multi` to ensemble with manual verbalizers, `proto` to use only ProtoVerb.
    """

    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizer],
                 model: Optional[PreTrainedModel],
                 label_vectors: Optional[torch.tensor] = None,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "mean",
                 post_log_softmax: Optional[bool] = True,
                 mid_dim: Optional[int] = 64,
                 epochs: Optional[int] = 5,
                 multi_verb: Optional[str] = "multi",
                 ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.label_words = label_words
        self.post_log_softmax = post_log_softmax
        self.criterion = nn.CrossEntropyLoss()

        self.multi_verb = multi_verb
        self.mid_dim = mid_dim
        self.epochs = epochs

        self.hidden_dims = model.config.hidden_size
        self.head = torch.nn.Linear(self.hidden_dims, self.mid_dim, bias=False)

        if label_words is not None:  # use label words as an initialization
            self.label_words = label_words

        # w = torch.empty((self.num_classes, self.hidden_dims))
        # nn.init.xavier_uniform_(w)
        # self.proto = nn.Parameter(w, requires_grad=True)
        self.proto = nn.Parameter(label_vectors, requires_grad=True)

    def on_label_words_set(self):
        super().on_label_words_set()
        self.label_words = self.add_prefix(self.label_words, self.prefix)

        # TODO should Verbalizer base class has label_words property and setter?
        # it don't have label_words init argument or label words from_file option at all

        self.generate_parameters()

    @staticmethod
    def add_prefix(label_words, prefix):
        r"""Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.

        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]  # wrapped it to a list of list of label words.

        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)

        max_len = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        words_ids_mask = [[[1] * len(ids) + [0] * (max_len - len(ids)) for ids in ids_per_label]
                          + [[0] * max_len] * (max_num_label_words - len(ids_per_label))
                          for ids_per_label in all_ids]
        words_ids = [[ids + [0] * (max_len - len(ids)) for ids in ids_per_label]
                     + [[0] * max_len] * (max_num_label_words - len(ids_per_label))
                     for ids_per_label in all_ids]

        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False)  # A 3-d mask
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)

    def process_hiddens(self, hiddens: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:
        """
        proto_logits = self.sim(self.head(hiddens), self.proto)
        return proto_logits

    def project(self,
                logits: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        r"""
        Project the labels, the return value is the normalized (sum to 1) probs of label words.

        Args:
            logits (:obj:`torch.Tensor`): The orginal logits of label words.

        Returns:
            :obj:`torch.Tensor`: The normalized logits of label words
        """

        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000 * (1 - self.label_words_mask)
        return label_words_logits

    def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        (4) Aggregate (for multiple label words)

        Args:
            logits (:obj:`torch.Tensor`): The orginal logits.

        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        # project
        label_words_logits = self.project(logits,
                                          **kwargs)  # Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)

        if self.post_log_softmax:
            # normalize
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            label_words_logits = torch.log(label_words_probs + 1e-15)

        # aggregate
        label_logits = self.aggregate(label_words_logits)
        return label_logits

    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""Use weight to aggregate the logits of label words.

        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.

        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words.
        """
        label_words_logits = (label_words_logits * self.label_words_mask).sum(-1) / self.label_words_mask.sum(-1)
        return label_words_logits

    def calibrate(self, label_words_probs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""

        Args:
            label_words_probs (:obj:`torch.Tensor`): The probability distribution of the label words with the shape of [``batch_size``, ``num_classes``, ``num_label_words_per_class``]

        Returns:
            :obj:`torch.Tensor`: The calibrated probability of label words.
        """
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() == 1, "self._calibrate_logits are not 1-d tensor"
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] \
               and calibrate_label_words_probs.shape[0] == 1, "shape not match"
        label_words_probs /= (calibrate_label_words_probs + 1e-15)
        # normalize # TODO Test the performance
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1,
                                                           keepdim=True)  # TODO Test the performance of detaching()
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs

    def ensemble_logits(self, manual_logits, proto_logits):

        logits = torch.stack([manual_logits, proto_logits])
        logits = logits.permute(1, 0, 2)
        logits = self.scaler(logits)
        logits = torch.mean(logits, 1)
        return logits

    @staticmethod
    def scaler(logits):
        m = logits.mean(-1, keepdim=True)
        s = logits.std(-1, keepdim=True)
        return (logits - m) / s

    def extract_at_mask(self,
                        outputs: torch.Tensor,
                        batch: Union[Dict, InputFeatures]):
        r"""Get outputs at all <mask> token
        E.g., project the logits of shape
        (``batch_size``, ``max_seq_length``, ``vocab_size``)
        into logits of shape (if num_mask_token > 1)
        (``batch_size``, ``num_mask_token``, ``vocab_size``)
        or into logits of shape (if ``num_mask_token`` = 1)
        (``batch_size``, ``vocab_size``).

        Args:
            outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
                 `gather_outputs` before) etc. of the whole sequence.
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.

        """
        mask_token_index = torch.where(batch["input_ids"] == 103)
        mask_token_logits = outputs[mask_token_index]
        return mask_token_logits

    def gather_outputs(self, outputs: ModelOutput):
        logits = outputs.logits
        if isinstance(outputs, Seq2SeqLMOutput):
            ret = outputs.decoder_hidden_states[-1]
        elif isinstance(outputs, MaskedLMOutput) or isinstance(outputs, CausalLMOutputWithCrossAttentions):
            ret = outputs.hidden_states[-1]
        else:
            try:
                ret = outputs.hidden_states[-1]
            except AttributeError:
                raise NotImplementedError(f"Gather outputs method for outputs' type {type(outputs)} not implemented")

        return ret, logits

    @staticmethod
    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1, 0))

    def pcl_loss(self, ins, proto, labels, training):
        # instance-prototype loss
        sim_ins_proto = torch.exp(self.sim(ins, proto))
        num = sim_ins_proto.shape[0]
        loss_ins_proto = 0.
        for i in range(num):
            pos_score = sim_ins_proto[i][labels[i]]
            neg_score = sim_ins_proto[i].sum()
            loss_ins_proto += - torch.log(pos_score / neg_score)
        loss_ins_proto /= num

        # if training:
        #     # instance-instance loss
        #     # sim_ins_ins = torch.exp(self.sim(ins, ins))
        #     # num = sim_ins_ins.shape[0]
        #     # loss_ins_ins = 0.
        #     # for i in range(num):
        #     #     pos_idx = list(set(torch.where(labels == labels[i])[0]))
        #     #     neg_idx = list(set(range(num)))
        #     #     l = len(pos_idx)
        #     #     neg_score = sim_ins_ins[i][neg_idx].sum()
        #     #     for j in pos_idx:
        #     #         pos_score = sim_ins_ins[i][j]
        #     #         loss_ins_ins += - torch.log(pos_score / neg_score) / l
        #     # loss_ins_ins /= num
        #
        sim_ins_ins = torch.exp(self.sim(ins, ins))
        num = sim_ins_ins.shape[0]
        loss_ins_ins = 0.
        for i in range(num):
            pos_score = sim_ins_ins[i][labels == labels[i]].sum()
            neg_score = sim_ins_ins[i].sum()
            loss_ins_ins += - torch.log(pos_score / neg_score)
        loss_ins_ins /= num
        #
        #     # prototype-prototype loss
        #     # sim_proto_proto = torch.exp(self.sim(proto, proto))
        #     # num = sim_proto_proto.shape[0]
        #     # loss_proto_proto = 0.
        #     # for i in range(num):
        #     #     pos_score = sim_proto_proto[i][i]
        #     #     neg_score = sim_proto_proto[i].sum()
        #     #     loss_proto_proto += - torch.log(pos_score / neg_score)
        #     # loss_proto_proto /= num
        #
        #     sim_proto_proto = self.sim(proto, proto)
        #     num = sim_proto_proto.shape[0]
        #     loss_proto_proto = 0.
        #     for i in range(num):
        #         for j in range(num):
        #             if i == j:
        #                 continue
        #             neg_score = 1 - sim_proto_proto[i][j]
        #             neg_score = torch.tensor([0., 0.9 - neg_score]).max() ** 2
        #             loss_proto_proto += neg_score / (num - 1)
        #     loss_proto_proto /= num
        #
        #     loss = loss_ins_proto + 0 * loss_ins_ins + 0 * loss_proto_proto
        #
        #     return sim_ins_proto, [loss, loss_ins_proto, loss_ins_ins, loss_proto_proto]

        return sim_ins_proto, loss_ins_ins

    def forward(self, outputs: ModelOutput, batch: Dict, labels: torch.tensor, training: bool, *args, **kwargs):
        static_anchor_outputs = kwargs['static_anchor_outputs']
        dynamic_anchor_outputs = kwargs['dynamic_anchor_outputs']
        anchor_batch = kwargs['anchor_batch']

        ret, logits = self.gather_outputs(outputs)
        static_anchor_ret, static_anchor_logits = self.gather_outputs(static_anchor_outputs)
        dynamic_anchor_ret, dynamic_anchor_logits = self.gather_outputs(dynamic_anchor_outputs)

        if isinstance(ret, tuple):
            ret_at_mask = [self.extract_at_mask(hidden, batch) for hidden in ret]
            logits_at_mask = [self.extract_at_mask(logit, batch) for logit in logits]
            static_anchor_ret_at_mask = [self.extract_at_mask(hidden, anchor_batch) for hidden in static_anchor_ret]
            static_anchor_logits_at_mask = [self.extract_at_mask(logit, anchor_batch) for logit in static_anchor_logits]
            dynamic_anchor_ret_at_mask = [self.extract_at_mask(hidden, anchor_batch) for hidden in dynamic_anchor_ret]
            dynamic_anchor_logits_at_mask = [self.extract_at_mask(logit, anchor_batch) for logit in
                                             dynamic_anchor_logits]

        else:
            ret_at_mask = self.extract_at_mask(ret, batch)
            logits_at_mask = self.extract_at_mask(logits, batch)
            static_anchor_ret_at_mask = self.extract_at_mask(static_anchor_ret, anchor_batch)
            static_anchor_logits_at_mask = self.extract_at_mask(static_anchor_logits, anchor_batch)
            dynamic_anchor_ret_at_mask = self.extract_at_mask(dynamic_anchor_ret, anchor_batch)
            dynamic_anchor_logits_at_mask = self.extract_at_mask(dynamic_anchor_logits, anchor_batch)

        # anchor_ret_at_mask = anchor_ret_at_mask.detach()
        # anchor_logits_at_mask = anchor_logits_at_mask.detach()

        # embedding = self.head(ret_at_mask)
        embedding = ret_at_mask

        # embedding_anchor = self.head(dynamic_anchor_ret_at_mask)
        embedding_anchor = dynamic_anchor_ret_at_mask.detach()

        sim_ins_proto, cl_loss = self.pcl_loss(embedding, embedding_anchor, labels, training)

        label_words_logits = self.process_logits(logits_at_mask, batch=batch)
        ce_loss = self.criterion(label_words_logits, labels)
        if self.training:
            loss = [ce_loss + cl_loss, ce_loss, cl_loss]
        else:
            loss = [ce_loss]

        # label_words_logits = self.process_logits(dynamic_anchor_logits_at_mask, batch=anchor_batch)
        # labels = torch.LongTensor(range(len(label_words_logits))).cuda()
        # ce_loss = self.criterion(label_words_logits, labels)
        # loss += [ce_loss]
        # loss[0] += ce_loss

        # kl_loss = F.kl_div(dynamic_anchor_logits_at_mask.softmax(dim=-1).log(),
        #                    static_anchor_logits_at_mask.softmax(dim=-1),
        #                    reduction='sum')
        # loss += kl_loss

        return sim_ins_proto, loss
