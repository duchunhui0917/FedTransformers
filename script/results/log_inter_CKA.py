import json
import os.path

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

base_dir = os.path.expanduser('~/FedTransformers')

FT_20news_100 = os.path.join(
    base_dir,
    'log/FedAvg/20news_s223_100_100_100/bert-base-uncased_0.01_10_5_22-08-10 21:40.log'
)
PET_20news_100 = os.path.join(
    base_dir,
    'log/FedAvg/20news_s223_100_100_100/bert-base-uncased_manual_0.01_10_5_22-08-10 22:01.log'
)
d = {
    'FT': FT_20news_100,
    'PET': PET_20news_100,
}


def process(lines):
    ls = []
    matrix0, matrix99 = None, None
    flag0, flag99 = False, False
    for line in lines:
        x = line.strip().split()
        if len(x) == 6 and x[-1] == '0********':
            flag0 = True
            print('flag0')
        if len(x) == 6 and x[-1] == '99********':
            flag99 = True
            print('flag99')

        if x[-2] == 'CKA:' and len(x) == 7:
            ls.append(float(x[-1]))
        if flag0 and len(x) == 104:
            matrix0 = np.array(eval(''.join(x[4:])))
            flag0 = False
        if flag99 and len(x) == 104:
            matrix99 = np.array(eval(''.join(x[4:])))
            flag99 = False
    return ls, matrix0, matrix99


matrix0_d = {}
matrix99_d = {}
for k, v in d.items():
    with open(v, 'r') as f:
        lines = f.readlines()
        ls, matrix0, matrix99 = process(lines)
        matrix0_d.update({k: matrix0})
        matrix99_d.update({k: matrix99})
    plt.plot(ls, label=k)
plt.legend()
plt.xlabel('Communication round')
plt.ylabel('Average CKA Similarity')
plt.title('20News/K=100')
plt.show()

for k, v in matrix0_d.items():
    sns.heatmap(v, vmin=0.3, vmax=1, linewidths=0.01, cmap="YlGnBu")
    plt.xlabel('Client ID')
    plt.ylabel('Client ID')
    plt.title(f'{k}(Communication round 1)')
    plt.show()
for k, v in matrix99_d.items():
    sns.heatmap(v, vmin=0.95, vmax=1, linewidths=0.01, cmap="YlGnBu")
    plt.xlabel('Client ID')
    plt.ylabel('Client ID')
    plt.title(f'{k}(Communication round 100)')
    plt.show()
