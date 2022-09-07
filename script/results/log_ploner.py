import os
import matplotlib.pyplot as plt

base_dir = os.path.expanduser('~/src')


def process(file):
    with open(os.path.join(base_dir, file), 'r') as f:
        lines = f.readlines()
    ls_f1 = []
    flag = False
    for line in lines:
        x = line.strip().split()
        if x[-2] == 'f1:' and len(x) == 6 and flag:
            ls_f1.append(float(x[-1]))
            flag = False
        if ' '.join(x[-3:]) == 'test centralized dataset':
            flag = True
    return ls_f1


file_centralized = 'log/centralized/ploner/distilbert-base-cased_22-06-15 21:33.log'
file = 'log/FedAvg/ploner/distilbert-base-cased_22-06-16 00:11.log'
file_solo = 'log/PartialFL/ploner/distilbert-base-cased_22-06-16 09:22.log'
files = [file_centralized, file, file_solo]
ite = 101
labels = ['centralized', 'FedAvg', 'SOLO']
for file, label in zip(files, labels):
    ls_f1 = process(file)
    if label == 'centralized':
        m = max(ls_f1)
        ls_f1 = [m for _ in range(ite)]
        plt.plot(ls_f1[:ite], label=label, linestyle='--')
    else:
        plt.plot(ls_f1[:ite], label=label)
plt.legend()
plt.title('PLONER/DistilBERT')
plt.xlabel('communication rounds')
plt.ylabel('F1')
plt.show()
