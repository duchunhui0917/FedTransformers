import matplotlib.pyplot as plt

centralized = [76.01, 84.17, 86.12, 87.26]
FedAvg_5 = [64.29, 83.34, 85.87, 86.94]
FedAvg_dot5 = [48.73, 83.32, 84.91, 86.29]
plt.plot(centralized, label='centralized', marker='.')
plt.plot(FedAvg_5, label=r'FedAvg($\alpha$=5)', marker='*')
plt.plot(FedAvg_dot5, label=r'FedAvg($\alpha$=0.5)', marker='s')
plt.xticks(range(4), ['BiLSTM(16M)', 'DistilBERT(67M)', 'BERT-base(110M)', 'BERT-large(335M)'])
plt.ylabel('F1')
plt.title('20news')
plt.legend()
plt.show()
