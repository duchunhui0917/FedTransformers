from matplotlib import pyplot as plt
import os
import numpy as np

base_dir = os.path.expanduser('~/FedTransformers')
centralized_FT = os.path.join(base_dir,
                              'log/centralized/20news_s223_100_100_100/bert-base-uncased_22-08-05 16:44.log')
centralized_PET = os.path.join(base_dir,
                               'log/centralized/20news_s223_100_100_100/bert-base-uncased_PET_22-08-05 15:20.log')

centralized_WARP = os.path.join(base_dir,
                                'log/centralized/20news_s223_100_100_100/bert-base-uncased_WRAP_22-08-05 15:33.log')
centralized_proto = os.path.join(base_dir,
                                 'log/centralized/20news_s223_100_100_100/bert-base-uncased_proto_22-08-05 14:40.log')

centralized_anchor = os.path.join(base_dir,
                                  'log/centralized/20news_s223_100_100_100/bert-base-uncased_anchor_22-07-29 20:49.log')

FedAvg_FT1 = os.path.join(base_dir,
                          'log/FedAvg/20news_s221_100_100_100/bert-base-uncased_0.01_10_5_22-08-03 21:35.log')
FedAvg_FT2 = os.path.join(base_dir,
                          'log/FedAvg/20news_s222_100_100_100/bert-base-uncased_0.01_10_5_22-08-03 21:32.log')
FedAvg_FT3 = os.path.join(base_dir,
                          'log/FedAvg/20news_s223_100_100_100/bert-base-uncased_0.01_10_5_22-07-30 10:57.log')

FedAvg_PET1 = os.path.join(base_dir,
                           'log/FedAvg/20news_s221_100_100_100/bert-base-uncased_PET_0.01_10_5_22-08-04 10:49.log')
FedAvg_PET2 = os.path.join(base_dir,
                           'log/FedAvg/20news_s222_100_100_100/bert-base-uncased_PET_0.01_10_5_22-08-04 10:50.log')
FedAvg_PET3 = os.path.join(base_dir,
                           'log/FedAvg/20news_s223_100_100_100/bert-base-uncased_PET_0.01_10_5_22-07-30 09:48.log')

FedAvg_WARP1 = os.path.join(base_dir,
                            'log/FedAvg/20news_s221_100_100_100/bert-base-uncased_WRAP_0.01_10_5_22-08-03 20:58.log')
FedAvg_WARP2 = os.path.join(base_dir,
                            'log/FedAvg/20news_s222_100_100_100/bert-base-uncased_WRAP_0.01_10_5_22-08-03 22:02.log')
FedAvg_WARP3 = os.path.join(base_dir,
                            'log/FedAvg/20news_s223_100_100_100/bert-base-uncased_WRAP_0.01_10_5_22-07-30 10:58.log')

FedAvg_Proto1 = os.path.join(base_dir,
                             'log/FedAvg/20news_s221_100_100_100/bert-base-uncased_proto_0.01_10_5_22-08-04 09:19.log')
FedAvg_Proto2 = os.path.join(base_dir,
                             'log/FedAvg/20news_s222_100_100_100/bert-base-uncased_proto_0.01_10_5_22-08-04 09:19.log')
FedAvg_Proto3 = os.path.join(base_dir,
                             'log/FedAvg/20news_s223_100_100_100/bert-base-uncased_proto_0.01_10_5_22-07-30 09:49.log')

FedAvg_ours1 = os.path.join(base_dir,
                            'log/FedAvg/20news_s223_100_100_100/bert-base-uncased_anchor_0.01_10_5_22-07-30 09:53.log')
FedAvg_ours2 = os.path.join(base_dir,
                            'log/FedAvg/20news_s223_100_100_100/bert-base-uncased_anchor_0.01_10_5_22-07-30 09:53.log')
FedAvg_ours3 = os.path.join(base_dir,
                            'log/FedAvg/20news_s223_100_100_100/bert-base-uncased_anchor_0.01_10_5_22-07-30 09:53.log')

centralized_d = {
    'FT': centralized_FT,
    'PET': centralized_PET,
    'SoftVerb': centralized_WARP,
    'ProtoVerb': centralized_proto,
    # 'ours': centralized_anchor
}

FedAvg_d = {
    'FT': [FedAvg_FT1, FedAvg_FT2, FedAvg_FT3],
    'PET': [FedAvg_PET1, FedAvg_PET2, FedAvg_PET3],
    'SoftVerb': [FedAvg_WARP1, FedAvg_WARP2, FedAvg_WARP3],
    'ProtoVerb': [FedAvg_Proto1, FedAvg_Proto2, FedAvg_Proto3],
    'ours': [FedAvg_ours1, FedAvg_ours2, FedAvg_ours3],
}


def process(lines):
    flag = False
    ls = []
    for line in lines:
        x = line.strip().split()
        if x[-2] == 'f1:' and len(x) == 6 and flag:
            ls.append(float(x[-1]) * 100)
            flag = False
        if ' '.join(x[-7:]) == 'test global test dataset with global model':
            flag = True
    return ls


def plot_ls(ls, key):
    ls = np.array(ls)
    x = np.arange(ls.shape[1])
    mean = ls.mean(axis=0)
    max = ls.max(axis=0)
    min = ls.min(axis=0)

    plt.plot(x, mean, label=key)
    plt.fill_between(x, min, max, alpha=0.2)


len1 = 31
for key, val in centralized_d.items():
    with open(val, 'r') as f:
        lines = f.readlines()
        lines = process(lines)[:len1]
        plt.plot(lines, label=key)

plt.legend()
plt.xticks(range(0, len1, 5))
plt.xlabel('Communication rounds')
plt.ylabel('F1 score (%) on validation dataset')
plt.title(r'20news')
plt.show()

for key, vals in FedAvg_d.items():
    ls = []
    for val in vals:
        with open(val, 'r') as f:
            len2 = 51 if key in ['PET', 'ours'] else 101
            lines = f.readlines()
            lines = process(lines)[:len2]
            ls.append(lines)
    plot_ls(ls, key)
plt.legend()
plt.xlabel('Communication rounds')
plt.ylabel('F1 score (%) on validation dataset')
plt.title(r'20news/K=100/$\alpha$=0.01')
plt.show()
