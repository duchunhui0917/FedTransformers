import json
from enum import Enum
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments
from .datasets.glue import task_to_keys as glue_task_to_keys
from .datasets.superglue import task_to_keys as superglue_task_to_keys
import os

base_dir = os.path.expanduser('~/src')

GLUE_DATASETS = list(glue_task_to_keys.keys())
SUPERGLUE_DATASETS = list(superglue_task_to_keys.keys())
TC_DATASETS = ["conll2003", "conll2004", "ontonotes", "ploner"]
SRL_DATASETS = ["conll2005", "conll2012"]
QA_DATASETS = ["squad", "squad_v2"]
RE_DATASETS = ["AIMed", "PGR", "AIMed*PGR", "BioInfer"]
SC_DATASETS = [
    "20news", "20news_class6", 'web_of_science', "agnews",
    "sst5", "sst2", "sentiment", "imdb", "rotten_tomatoes", "yelp_polarity", "amazon_polarity",
    "sst2*imdb*yelp_polarity*rotten_tomatoes",
    "imdb*yelp_polarity*rotten_tomatoes",
    "sst2*yelp_polarity*rotten_tomatoes",
    "sst2*imdb*yelp_polarity",
    "sst2*imdb*rotten_tomatoes",
]
SS_DATASETS = ["cornell_movie_dialogue", "cnn_dailymail", "opus_books"]
LM_DATASETS = ['shakespeare', 'eli5']

ADD_PREFIX_SPACE = {
    'bert': False,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': True,
}

USE_FAST = {
    'bert': True,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': False,
}


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.training_args
    """

    task_name: str = field(
        metadata={
            "help": "The name of the task"
        }
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset"
        }
    )
    split: Optional[str] = field(
        default=None,
        metadata={}
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: Optional[int] = field(
        default=256,
    )
    decoder_max_length: Optional[int] = field(
        default=256,
    )
    stride: Optional[int] = field(
        default=128,
        metadata={
            "help": "The stride for question answering."
        }
    )
    n_best: Optional[int] = field(
        default=20,
        metadata={
            "help": "The n_best for question answering."
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv aor a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."}
    )
    template_id: Optional[int] = field(
        default=0,
        metadata={
            "help": "The specific prompt string to use"
        }
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_type: str = field(
        metadata={

        }
    )
    model_name: str = field(
        metadata={
            "help": "model identifier from huggingface.co/models",
        }
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    tunning_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "The methods of tunning model.",
            "choices": [
                'bottleneck_adapter',
                'prefix_tunning',
                'frozen'
            ]
        }
    )
    prompt_method: Optional[str] = field(
        default=None,
        metadata={
            "choices": {
                'manual',
                'soft',
                'proto',
                'anchor'
            }
        }
    )
    prefix_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The length of prompt"
        }
    )
    vocab_file: Optional[str] = field(
        default=os.path.join(base_dir, 'data/glove.6B/glove.6B.100d.json'),
        metadata={
            "help": "The vocab file."
        }
    )
    label_words: Optional[str] = field(
        default=None
    )
    anchor_texts: Optional[str] = field(
        default=None
    )
    template_text: Optional[str] = field(
        default=None
    )
    template: Optional[str] = field(
        default=None
    )
    augment: Optional[str] = field(
        default=None,
        metadata={}
    )


@dataclass
class QuestionAnsweringArguments:
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
                    "the score of the null answer minus this threshold, the null answer is selected for this example. "
                    "Only useful when `version_2_with_negative=True`."
        },
    )


@dataclass
class FederatedLearningArguments:
    algorithm: str = field(
        metadata={}
    )
    split_type: str = field(
        metadata={
            "choices": [
                "centralized",
                "doc_split",
                "label_split",
                "feature_split",
                "class_split",
                "uniform_split",
                "idx_split"
            ]
        }
    )
    do_train: Optional[bool] = field(
        default=True,
        metadata={}
    )
    do_test: Optional[bool] = field(
        default=True,
        metadata={}
    )
    lr: Optional[float] = field(
        default=5e-5,
        metadata={"help": "Learning rate."}
    )
    seed: Optional[int] = field(
        default=223,
        metadata={}
    )
    dirichlet_alpha: Optional[float] = field(
        default=0.5,
        metadata={"help": "The alpha of dirichlet distribution."}
    )
    num_clients: Optional[int] = field(
        default=1,
        metadata={}
    )
    num_clusters: Optional[int] = field(
        default=0,
        metadata={}
    )
    train_batch_size: Optional[int] = field(
        default=16
    )
    eval_batch_size: Optional[int] = field(
        default=16
    )
    test_batch_size: Optional[int] = field(
        default=16
    )
    num_epochs: Optional[int] = field(
        default=1
    )
    num_batches: Optional[int] = field(
        default=0
    )
    num_iterations: Optional[int] = field(
        default=100,
        metadata={}
    )
    select_ratio: Optional[float] = field(
        default=1,
        metadata={}
    )
    aggregate_method: Optional[str] = field(
        default='sample',
        metadata={}
    )
    weight_sampler: Optional[bool] = field(
        default=False,
        metadata={}
    )
    tgwg: Optional[bool] = field(
        default=True,
        metadata={"help": "test global test dataset with global model"}
    )
    tgwp: Optional[bool] = field(
        default=False,
        metadata={"help": "test global test dataset with personal models"}
    )
    tpwg: Optional[bool] = field(
        default=False,
        metadata={"help": "test personal test datasets with global model"}
    )
    tpwp: Optional[bool] = field(
        default=False,
        metadata={"help": "test personal test datasets with personal models"}
    )
    test_frequency: Optional[int] = field(
        default=1,
        metadata={}
    )
    server_momentum: Optional[float] = field(
        default=0,
        metadata={}
    )

    client_momentum: Optional[float] = field(
        default=0,
        metadata={}
    )
    layers: Optional[str] = field(
        default='',
        metadata={}
    )
    classes: Optional[str] = field(
        default='',
        metadata={}
    )
    partition_path: Optional[str] = field(
        default='',
        metadata={}
    )
    partition_group: Optional[str] = field(
        default='',
        metadata={}
    )


@dataclass
class WandbArguments:
    enable: Optional[bool] = field(
        default=False,
        metadata={}
    )
    team_name: Optional[str] = field(
        default='',
        metadata={}
    )
    project_name: Optional[str] = field(
        default='',
        metadata={}
    )
