import os
import matplotlib.pyplot as plt

base_dir = os.path.expanduser('~/FedTransformers')


def process(file):
    with open(os.path.join(base_dir, file), 'r') as f:
        lines = f.readlines()
    ls_f1 = []
    for line in lines:
        x = line.strip().split()
        if x[-2] == 'f1:' and len(x) == 6:
            ls_f1.append(float(x[-1]))
    return ls_f1


# file_centralized = 'log/centralized/PGR_Q1/distilbert-base-cased_22-06-15 20:43.log'
file_5_10 = 'log/FedAvg/PGR_Q1/distilbert-base-cased_5_10_22-06-15 13:28.log'
file_1_10 = 'log/FedAvg/PGR_Q1/distilbert-base-cased_1_10_22-06-15 10:56.log'
file_dot5_10 = 'log/FedAvg/PGR_Q1/distilbert-base-cased_0.5_10_22-06-15 16:01.log'
files = [file_5_10, file_1_10, file_dot5_10]
ite = 20
plt.plot([0.7667 for _ in range(ite)], label='centralized', linestyle='--')

labels = [r'$\alpha$=5', r'$\alpha$=1', r'$\alpha$=0.5']
for file, label in zip(files, labels):
    ls_f1 = process(file)
    plt.plot(ls_f1[1:ite], label=label)
plt.legend()
plt.xticks(list(range(0, 21, 5)))

plt.title('PGR/DistilBERT')
plt.xlabel('communication rounds')
plt.ylabel('F1')
plt.show()
