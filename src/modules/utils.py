from enum import Enum

import torch
import torch.nn.functional as F

from src.models.token_classification import (
    BertPrefixForTokenClassification,
    RobertaPrefixForTokenClassification,
    # DebertaPrefixForTokenClassification,
    # DebertaV2PrefixForTokenClassification
)

from src.models.sequence_classification import (
    BertPrefixForSequenceClassification,
    BertPromptForSequenceClassification,
    RobertaPrefixForSequenceClassification,
    RobertaPromptForSequenceClassification,
    # DebertaPrefixForSequenceClassification
)

from src.models.question_answering import (
    BertPrefixForQuestionAnswering,
    RobertaPrefixModelForQuestionAnswering,
    # DebertaPrefixModelForQuestionAnswering
)

from src.models.multiple_choice import (
    BertPrefixForMultipleChoice,
    RobertaPrefixForMultipleChoice,
    # DebertaPrefixForMultipleChoice,
    BertPromptForMultipleChoice,
    RobertaPromptForMultipleChoice
)

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice
)


class TaskType(Enum):
    TOKEN_CLASSIFICATION = 1,
    SEQUENCE_CLASSIFICATION = 2,
    QUESTION_ANSWERING = 3,
    MULTIPLE_CHOICE = 4


PREFIX_MODELS = {
    "bert": {
        TaskType.TOKEN_CLASSIFICATION: BertPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: BertPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: BertPrefixForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: BertPrefixForMultipleChoice
    },
    "roberta": {
        TaskType.TOKEN_CLASSIFICATION: RobertaPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: RobertaPrefixModelForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: RobertaPrefixForMultipleChoice,
    },
    # "deberta": {
    #     TaskType.TOKEN_CLASSIFICATION: DebertaPrefixForTokenClassification,
    #     TaskType.SEQUENCE_CLASSIFICATION: DebertaPrefixForSequenceClassification,
    #     TaskType.QUESTION_ANSWERING: DebertaPrefixModelForQuestionAnswering,
    #     TaskType.MULTIPLE_CHOICE: DebertaPrefixForMultipleChoice,
    # },
    "deberta-v2": {
        # TaskType.TOKEN_CLASSIFICATION: DebertaV2PrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: None,
        TaskType.QUESTION_ANSWERING: None,
        TaskType.MULTIPLE_CHOICE: None,
    }
}

PROMPT_MODELS = {
    "bert": {
        TaskType.SEQUENCE_CLASSIFICATION: BertPromptForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: BertPromptForMultipleChoice
    },
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPromptForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: RobertaPromptForMultipleChoice
    }
}

AUTO_MODELS = {
    TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
    TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
    TaskType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice,
}


def get_model(model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False):
    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size

        model_class = PREFIX_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    elif model_args.prompt:
        config.pre_seq_len = model_args.pre_seq_len
        model_class = PROMPT_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    else:
        model_class = AUTO_MODELS[task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )

        bert_param = 0
        if fix_bert:
            if config.model_type == "bert":
                for param in model.bert.parameters():
                    param.requires_grad = False
                for _, param in model.bert.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "deberta":
                for param in model.deberta.parameters():
                    param.requires_grad = False
                for _, param in model.deberta.named_parameters():
                    bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('***** total param is {} *****'.format(total_param))
    return model


def get_model_deprecated(model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False):
    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size

        if task_type == TaskType.TOKEN_CLASSIFICATION:
            from model.token_classification import BertPrefixModel, RobertaPrefixModel, DebertaPrefixModel, \
                DebertaV2PrefixModel
        elif task_type == TaskType.SEQUENCE_CLASSIFICATION:
            from model.sequence_classification import BertPrefixModel, RobertaPrefixModel, DebertaPrefixModel, \
                DebertaV2PrefixModel
        elif task_type == TaskType.QUESTION_ANSWERING:
            from model.question_answering import BertPrefixModel, RobertaPrefixModel, DebertaPrefixModel, \
                DebertaV2PrefixModel
        elif task_type == TaskType.MULTIPLE_CHOICE:
            from model.multiple_choice import BertPrefixModel

        if config.model_type == "bert":
            model = BertPrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "roberta":
            model = RobertaPrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "deberta":
            model = DebertaPrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "deberta-v2":
            model = DebertaV2PrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        else:
            raise NotImplementedError


    elif model_args.prompt:
        config.pre_seq_len = model_args.pre_seq_len

        from src.models.sequence_classification import BertPromptModel, RobertaPromptModel
        if config.model_type == "bert":
            model = BertPromptModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "roberta":
            model = RobertaPromptModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        else:
            raise NotImplementedError


    else:
        if task_type == TaskType.TOKEN_CLASSIFICATION:
            model = AutoModelForTokenClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )

        elif task_type == TaskType.SEQUENCE_CLASSIFICATION:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )

        elif task_type == TaskType.QUESTION_ANSWERING:
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif task_type == TaskType.MULTIPLE_CHOICE:
            model = AutoModelForMultipleChoice.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )

        bert_param = 0
        if fix_bert:
            if config.model_type == "bert":
                for param in model.bert.parameters():
                    param.requires_grad = False
                for _, param in model.bert.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "deberta":
                for param in model.deberta.parameters():
                    param.requires_grad = False
                for _, param in model.deberta.named_parameters():
                    bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('***** total param is {} *****'.format(total_param))
    return model


def distance(features, centers):
    f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
    c_2 = torch.sum(torch.pow(centers, 2), dim=1, keepdim=True)
    dist = f_2 - 2 * torch.matmul(features, centers.transpose(1, 0)) + c_2.transpose(1, 0)
    return dist


def mce_loss(features, labels, centers, epsilon=1):
    dist = distance(features, centers)
    values, indexes = torch.topk(- dist, k=2, sorted=True)
    top2 = -values
    d_1 = top2[:, 0]
    d_2 = top2[:, 1]

    row_idx = torch.range(0, labels.size(0) - 1, dtype=torch.int64)
    d_y = dist[row_idx, labels]

    indicator = (torch.argmin(dist) == labels).int()
    d_c = indicator * d_2 + (1 - indicator) * d_1

    measure = d_y - d_c
    loss = torch.sigmoid(epsilon * measure).mean()
    return loss


# features = torch.rand((8, 16))
# labels = torch.ones(8, dtype=torch.int64)
# centers = torch.rand((10, 16))
# mce_loss(features, labels, centers)


def mcl_loss(features, labels, centers, margin=1):
    dist = distance(features, centers)
    values, indexes = torch.topk(- dist, k=2, sorted=True)
    top2 = -values
    d_1 = top2[:, 0]
    d_2 = top2[:, 1]

    row_idx = torch.range(0, labels.size(0) - 1, dtype=torch.int64)
    d_y = dist[row_idx, labels]

    indicator = (torch.argmin(dist) == labels).int()
    d_c = indicator * d_2 + (1 - indicator) * d_1

    loss = torch.relu(d_y - d_c + margin).mean()
    return loss


def dce_loss(features, labels, centers, t=1):
    dist = distance(features, centers)
    logits = -dist / t

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    return loss


def sim(x, y):
    norm_x = F.normalize(x, dim=-1)
    norm_y = F.normalize(y, dim=-1)
    return torch.matmul(norm_x, norm_y.transpose(1, 0))

# dist = torch.LongTensor([1, 2, 3, 4, 5])
# mcl_loss(dist)
