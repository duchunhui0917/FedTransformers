# 对模型进行MLM预训练
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
import os
from torch.utils.data import DataLoader
from torch import optim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

base_dir = os.path.expanduser('~/src')
train_file = "data/bio_RE/PGR_Q1/PGR_Q1-train.tsv"
train_file = os.path.join(base_dir, train_file)
test_file = "data/bio_RE/PGR_Q1/PGR_Q1-test.tsv"
test_file = os.path.join(base_dir, test_file)
batch_size = 16

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
# model = AutoModelForMaskedLM.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_config(config)
model.cuda()

# model = AutoModel.from_pretrained(model_name)
# classifier = MaskedLMClassifier(config)
# model.cuda()
# classifier.cuda()

optimizer = optim.AdamW(model.parameters(), lr=5e-5)
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128
)
test_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=test_file,
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)


def train():
    best_true_acc = 0
    for epoch in range(100):
        avg_loss = 0
        avg_acc = 0
        avg_true_acc = 0
        for count, inputs in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = inputs['input_ids'], inputs["attention_mask"], inputs["labels"]
            input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
            logits, loss = outputs.logits, outputs.loss

            # outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # last_hidden_state = outputs.last_hidden_state
            # logits, loss = classifier(labels=labels, hidden_state=last_hidden_state)

            pred_labels = logits.argmax(-1)
            acc = (pred_labels == labels).sum() / (labels.size(0) * labels.size(1))
            true_acc = ((pred_labels == labels) & (labels != -100)).sum() / (labels != -100).sum()
            loss.backward()
            optimizer.step()
            labels = labels.cpu().numpy()
            pred_labels = pred_labels.cpu().numpy()

            avg_loss = (avg_loss * count + loss.item()) / (count + 1)
            avg_acc = (avg_acc * count + acc) / (count + 1)
            avg_true_acc = (avg_true_acc * count + true_acc) / (count + 1)
        print(f'epoch {epoch}, avg loss {avg_loss:.4f}, acc: {avg_acc:.4f}, true acc: {avg_true_acc:.4f}')

        true_acc = test()
        if true_acc > best_true_acc:
            best_true_acc = true_acc
            sd = model.state_dict()
            path = os.path.join(base_dir, f'ckpt/MaskedLM/PGR_Q1/{model_name}_scratch')
            torch.save(sd, path)
            print('new model has been saved')


def test():
    avg_loss = 0
    avg_acc = 0
    avg_true_acc = 0
    print('******testing dataset******')
    for count, inputs in enumerate(test_dataloader):
        input_ids, attention_mask, labels = inputs['input_ids'], inputs["attention_mask"], inputs["labels"]
        input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        logits, loss = outputs.logits, outputs.loss

        # outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # last_hidden_state = outputs.last_hidden_state
        # logits, loss = classifier(labels=labels, last_hidden_state=last_hidden_state)

        pred_labels = logits.argmax(-1)
        acc = (pred_labels == labels).sum() / (labels.size(0) * labels.size(1))
        true_acc = ((pred_labels == labels) & (labels != -100)).sum() / (labels != -100).sum()

        avg_loss = (avg_loss * count + loss.item()) / (count + 1)
        avg_acc = (avg_acc * count + acc) / (count + 1)
        avg_true_acc = (avg_true_acc * count + true_acc) / (count + 1)
    print(f'avg loss {avg_loss:.4f}, acc: {avg_acc:.4f}, true acc: {avg_true_acc:.4f}')
    return avg_true_acc


train()
