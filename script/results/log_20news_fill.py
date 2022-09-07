from matplotlib import pyplot as plt
import os
import numpy as np

base_dir = os.path.expanduser('~/FedTransformers')

FT = os.path.join(base_dir,
                  'log/FedAvg/20news_s223_100_100_100/bert-base-uncased_0.01_10_5_22-08-03 23:00.log')

PET = os.path.join(base_dir,
                   'log/FedAvg/20news_s223_100_100_100/bert-base-uncased_anchor_0.01_10_5_22-08-04 16:59.log')

ours = os.path.join(base_dir,
                    'log/FedAvg/20news_s223_100_100_100/bert-base-uncased_PET_0.01_10_5_22-08-04 17:00.log')

FedAvg = {
    # 'FT': FT,
    'PET': PET,
    'ours': ours
}


def process(lines, key):
    flag = False
    best = []
    avg = []
    worst = []
    for line in lines:
        x = line.strip().split()
        if x[4] == 'best/avg/worst/std':
            s = x[-1]
            ls = s.split('/')
            best.append(float(ls[0]))
            avg.append(float(ls[1]))
            worst.append(float(ls[2]))
    x = range(len(avg))
    plt.plot(avg)
    plt.fill_between(x, worst, best, alpha=0.2)


for key, val in FedAvg.items():
    with open(val, 'r') as f:
        lines = f.readlines()
        lines = process(lines, key)

plt.legend()
plt.xlabel('Communication rounds')
plt.ylabel('F1 score (%) on validation dataset for client3')
plt.title(r'20news/K=100/$\alpha$=0.01')
plt.show()
