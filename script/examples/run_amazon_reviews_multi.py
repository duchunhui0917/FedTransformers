from datasets import load_dataset, load_metric, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from transformers import Seq2SeqTrainingArguments
import numpy as np
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


def filter_books(example):
    return (
            example["product_category"] == "book"
            or example["product_category"] == "digital_ebook_purchase"
    )


def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])


def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text) for text in dataset["review_body"]]
    return metric.compute(predictions=summaries, references=dataset["review_title"])


max_input_length = 512
max_target_length = 30


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"], max_length=max_input_length, truncation=True
    )
    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["review_title"], max_length=max_target_length, truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


spanish_dataset = load_dataset("amazon_reviews_multi", "es")
english_dataset = load_dataset("amazon_reviews_multi", "en")

spanish_books = spanish_dataset.filter(filter_books)
english_books = english_dataset.filter(filter_books)

english_dataset.set_format("pandas")
english_df = english_dataset["train"][:]

books_dataset = DatasetDict()
for split in english_books.keys():
    books_dataset[split] = concatenate_datasets(
        [english_dataset[split], spanish_dataset[split]]
    )
    books_dataset[split] = books_dataset[split].shuffle(seed=42)

books_dataset = books_dataset.filter(lambda x: len(x["review_title"].split()) > 2)
model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenized_datasets = books_dataset.map(preprocess_function, batched=True)

rouge_score = load_metric("rouge")
batch_size = 8
num_train_epochs = 8
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-amazon-en-es",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
tokenized_datasets = tokenized_datasets.remove_columns(
    books_dataset["train"].column_names
)
features = [tokenized_datasets["train"][i] for i in range(2)]
data_collator(features)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
