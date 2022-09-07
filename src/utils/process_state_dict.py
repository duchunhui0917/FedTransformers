import os.path
import torch
from collections import OrderedDict

base_dir = os.path.expanduser('~/src')
ckpt = os.path.join(base_dir, 'ckpt/centralized/PGR_Q1/google_mobilebert-uncased_central')
d1 = torch.load(ckpt)
d2 = OrderedDict()

for key, val in d1.items():
    if 're' in key:
        ls = key.split('.')
        ls[0] = 'encoder'
        key = '.'.join(ls)
        d2.update({key: val})
    elif 'unmask' in key:
        continue
    else:
        d2.update({key: val})
torch.save(d2, ckpt)
