from matplotlib import pyplot as plt

centralized_20news_FT = [45.4, 75.8, 86.51]
FedAvg_20news_FT = [32.33, 69.6, 83.35]
centralized_20news_PET = [62.91, 74.98, 86.97]
FedAvg_20news_PET = [54.75, 71.22, 82.99]

centralized_sst5_FT = [37.93, 44.74, 50.31]
FedAvg_sst5_FT = [27.61, 40.03, 43.33]
centralized_sst5_PET = [42.06, 43.34, 49.96]
FedAvg_sst5_PET = [42.06, 40.99, 44.06]

d1 = {
    'FT(centralized)': {'x': centralized_20news_FT, 'c': '#1f77b4', 'l': '-'},
    'FT(FedAvg)': {'x': FedAvg_20news_FT, 'c': '#1f77b4', 'l': '--'},
    'PET(centralized)': {'x': centralized_20news_PET, 'c': '#ff7f0e', 'l': '-'},
    'PET(FedAvg)': {'x': FedAvg_20news_PET, 'c': '#ff7f0e', 'l': '--'},
}

d2 = {
    'FT(centralized)': {'x': centralized_sst5_FT, 'c': '#1f77b4', 'l': '-'},
    'FT(FedAvg)': {'x': FedAvg_sst5_FT, 'c': '#1f77b4', 'l': '-.'},
    'PET(centralized)': {'x': centralized_sst5_PET, 'c': '#ff7f0e', 'l': '-'},
    'PET(FedAvg)': {'x': FedAvg_sst5_PET, 'c': '#ff7f0e', 'l': '-.'},
}
x = ['K=100', 'K=800', 'Full']
for k, v in d1.items():
    plt.plot(v['x'], label=k, color=v['c'], linestyle=v['l'], marker='s')
    # plt.errorbar(x=range(3), y=v['x'], yerr=[1, 5, 7], color=v['c'], capsize=3)
plt.xticks(range(3), x)
plt.ylabel('F1 score (%) on validation dataset')
plt.title('20News')
plt.legend()
plt.show()

for k, v in d2.items():
    plt.plot(v['x'], label=k, color=v['c'], linestyle=v['l'], marker='s')
plt.xticks(range(3), x)
plt.ylabel('F1 score (%) on validation dataset')
plt.title('SST-5')
plt.legend()
plt.show()
