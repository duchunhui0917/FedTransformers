from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import DataLoader

block_size = 128


def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]], truncation=True)


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    for key, val in result.items():
        result[key] = val[:-1]
    result["labels"] = result["input_ids"].copy()
    return result


tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
eli5 = load_dataset("eli5", split="train_asks[:5000]")
eli5 = eli5.train_test_split(test_size=0.2)
eli5 = eli5.flatten()
tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5["train"].column_names,
)
lm_dataset = tokenized_eli5.map(
    group_texts,
    batched=True,
    num_proc=4
)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
model = AutoModelForCausalLM.from_pretrained("distilroberta-base")

loader = DataLoader(lm_dataset['train'], collate_fn=data_collator, batch_size=16)
for data in loader:
    continue

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()
