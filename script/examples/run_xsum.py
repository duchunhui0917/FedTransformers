import os.path

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

prefix = "summarize: "


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]
    pred_labels = np.argmax(preds, axis=-1)
    decoded_pred_labels = tokenizer.batch_decode(pred_labels, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_pred_labels = [pred.strip() for pred in decoded_pred_labels]

    decoded_labels = [label.strip() for label in decoded_labels]
    rouge_results = rouge.compute(predictions=decoded_pred_labels, references=decoded_labels)

    decoded_labels = [[label.strip()] for label in decoded_labels]
    bleu_results = sacrebleu.compute(predictions=decoded_pred_labels, references=decoded_labels)

    # perplexity_results = self.perplexity.compute(model_id='t5', input_texts=decoded_pred_labels)

    return {"rouge1": rouge_results["rouge1"].mid.fmeasure,
            "rouge2": rouge_results["rouge2"].mid.fmeasure,
            "rougeL": rouge_results["rougeL"].mid.fmeasure,
            "bleu": bleu_results["score"]}


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print('loading dataset')
raw_datasets = load_dataset("xsum", "default")
print(len(raw_datasets['train']), len(raw_datasets['validation']), len(raw_datasets['test']))
raw_datasets['train'] = raw_datasets['train'].shuffle().select(range(800))
raw_datasets['validation'] = raw_datasets['validation'].shuffle().select(range(100))

model = AutoModelForSeq2SeqLM.from_pretrained('t5-large')
tokenizer = AutoTokenizer.from_pretrained('t5-large')
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True,
                                      remove_columns=raw_datasets["train"].column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

print('loading metric')
rouge = load_metric("rouge")
sacrebleu = load_metric("sacrebleu")

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# trainer.train()
res = trainer.evaluate()
print(res)
