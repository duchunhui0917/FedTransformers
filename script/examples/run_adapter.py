from datasets import load_dataset
from transformers import RobertaTokenizer
from transformers import RobertaConfig, RobertaAdapterModel
import numpy as np
from transformers import TrainingArguments, Trainer, AdapterTrainer, EvalPrediction
from transformers import list_adapters
from transformers.adapters import AdapterConfig


def encode_batch(batch):
    """Encodes a batch of input data using the model tokenizer."""
    return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")


dataset = load_dataset("rotten_tomatoes")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Encode the input data
dataset = dataset.map(encode_batch, batched=True)
# The transformers model expects the target class column to be named "labels"
dataset = dataset.rename_column("label", "labels")
# Transform to pytorch tensors and only output the required columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=2,
)
model = RobertaAdapterModel.from_pretrained(
    "roberta-base",
    config=config,
)
named_parameters = dict(model.named_parameters())
# Add a new adapter
adapter_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
model.add_adapter("bottleneck_adapter", config=adapter_config)
named_parameters = dict(model.named_parameters())

# Add a matching classification head
model.add_classification_head(
    "rotten_tomatoes",
    num_labels=2,
    id2label={0: "👎", 1: "👍"}
)
# Activate the adapter
model.train_adapter("bottleneck_adapter")
named_parameters = dict(model.named_parameters())

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_accuracy,
)

trainer.train()
trainer.evaluate()
