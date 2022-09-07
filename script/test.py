import json
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

base_dir = os.path.expanduser('~/FedTransformers')
path = os.path.join(base_dir, 'data/weight/PGR_s233.json')
with open(path, 'r') as f:
    d = json.load(f)
sorted_weight = sorted(d.items(), key=lambda d: d[1], reverse=True)
word2id = tokenizer.vocab
id2word = {v: k for k, v in word2id.items()}

word_sorted_weight = [(id2word[int(i)], w) for (i, w) in sorted_weight]
print('hello')
