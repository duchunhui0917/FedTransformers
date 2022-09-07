from matplotlib import pyplot as plt
import os
import numpy as np

base_dir = os.path.expanduser('~/FedTransformers')

FT = os.path.join(base_dir,
                  'log/FedAvg/20news_s223_100_100_100/bert-base-uncased_0.01_10_5_22-08-04 19:12.log')

PET = os.path.join(base_dir,
                   'log/FedAvg/20news_s223_800_800_800/bert-base-uncased_PET_0.01_10_5_22-07-27 16:59.log')

ours = os.path.join(base_dir,
                    'log/FedAvg/20news_s223_100_100_100/bert-base-uncased_anchor_0.01_10_5_22-08-04 16:59.log')

FedAvg = {
    # 'FT': FT,
    'PET': PET,
    # 'ours': ours
}


def process(lines, key):
    flag = False
    if key == 'PET':
        ls = [22.56]
    elif key == 'ours':
        ls = [15.39]
    else:
        ls = [0.0108]

    for line in lines:
        x = line.strip().split()
        if x[-2] == 'f1:' and len(x) == 6 and flag:
            ls.append(float(x[-1]) * 100)
            flag = False
        if ' '.join(x[-9:]) == 'test global test dataset with client 3 personal model':
            flag = True
        # if ' '.join(x[-7:]) == 'test global test dataset with global model':
        #     flag = True
    plt.plot(ls, label=key)

    return ls


for key, val in FedAvg.items():
    with open(val, 'r') as f:
        lines = f.readlines()
        lines = process(lines, key)

plt.legend()
plt.xlabel('Communication rounds')
plt.ylabel('F1 score (%) on validation dataset')
plt.title(r'Client3')
plt.show()
