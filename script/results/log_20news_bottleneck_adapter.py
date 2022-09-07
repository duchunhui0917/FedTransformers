import os
import matplotlib.pyplot as plt

base_dir = os.path.expanduser('~/FedTransformers')


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
        if ' '.join(x[-7:]) == 'test global test dataset with global model':
            flag = True

    return ls_f1


file_5_10 = 'log/FedAvg/20news/distilbert-base-uncased_bottleneck_adapter_5.0_10_1_22-07-14 13:31.log'
file_1_10 = 'log/FedAvg/20news/distilbert-base-uncased_bottleneck_adapter_1.0_10_1_22-07-14 13:30.log'
file_dot5_10 = 'log/FedAvg/20news/distilbert-base-uncased_bottleneck_adapter_0.5_10_1_22-07-14 13:29.log'
files = [file_5_10, file_1_10, file_dot5_10]
labels = [r'$\alpha$=5', r'$\alpha$=1', r'$\alpha$=0.5']

plt.plot()
ite = 31

x = [0.84 for _ in range(ite)]
plt.plot(x, label='centralized', linestyle='--')

for file, label in zip(files, labels):
    ls_f1 = process(file)
    plt.plot(ls_f1[:ite], label=label)
plt.legend()
plt.title('20news/DistilBERT')
plt.xlabel('communication rounds')
plt.ylabel('F1')
plt.show()
