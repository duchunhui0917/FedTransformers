from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import os

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

block_size = 128


def preprocess_function(examples):
    results = tokenizer([" ".join(x) for x in examples["answers.text"]],
                        truncation=True, padding=True, max_length=256)
    results["labels"] = results["input_ids"][:]
    return results


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    results = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    results["labels"] = results["input_ids"][:]
    # for key, val in results.items():
    #     results[key] = val[:-1]
    return results


eli5 = load_dataset("eli5", split="train_asks[:5000]")

eli5 = eli5.train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
gpt2 = AutoModelForCausalLM.from_pretrained("distilgpt2")

t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

eli5 = eli5.flatten()
lm_dataset = eli5.map(
    preprocess_function,
    batched=True,
    # num_proc=4,
    remove_columns=eli5["train"].column_names,
)
# lm_dataset = lm_dataset.map(
#     group_texts,
#     batched=True,
#     # num_proc=4
# )

loader = DataLoader(lm_dataset['train'], collate_fn=data_collator, batch_size=16)
for data in loader:
    continue

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()