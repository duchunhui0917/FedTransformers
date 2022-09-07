import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

imdb = load_dataset("gpt3mix/sst2")
remove_columns = list(set(imdb["train"].column_names) - {'label'})
tokenized_imdb = imdb.map(preprocess_function, batched=True, remove_columns=remove_columns)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

loader = DataLoader(tokenized_imdb['train'], batch_size=32, collate_fn=data_collator)
for data in loader:
    continue

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
