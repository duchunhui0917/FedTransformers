import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel

# model_name = "distilbert-base-uncased"
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
pad_token_id = tokenizer.pad_token_id
cls_token_id = tokenizer.cls_token_id
sep_token_id = tokenizer.sep_token_id

config = AutoConfig.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
text = ['i love you [MASK] i love you',
        'i love you [MASK] i love you']
inputs = tokenizer(text, padding=True, truncation=True, max_length=48)
input_ids = inputs['input_ids']
print(input_ids)
