from matplotlib import pyplot as plt
import os
import numpy as np
import sys

sys.path.append('../../')
from src.utils.common_utils import smooth_curve

base_dir = os.path.expanduser('~/FedTransformers')
centralized_FT = os.path.join(base_dir,
                              'log/centralized/sst_s223_100_100_100/bert-base-uncased_22-07-28 20:58.log')
centralized_PET = os.path.join(base_dir,
                               'log/centralized/sst_s223_100_100_100/bert-base-uncased_PET_22-07-29 21:10.log')

centralized_WARP = os.path.join(base_dir,
                                'log/centralized/sst_s223_100_100_100/bert-base-uncased_WRAP_22-07-29 16:57.log')
centralized_proto = os.path.join(base_dir,
                                 'log/centralized/sst_s223_100_100_100/bert-base-uncased_proto_22-07-29 19:53.log')

centralized_anchor = os.path.join(base_dir,
                                  'log/centralized/sst_s223_100_100_100/bert-base-uncased_anchor_22-07-29 20:49.log')

FedAvg_FT = os.path.join(base_dir,
                         'log/FedAvg/sst_s223_100_100_100/bert-base-uncased_0.5_10_5_22-07-31 18:55.log')
FedAvg_PET = os.path.join(base_dir,
                          'log/FedAvg/sst_s223_100_100_100/bert-base-uncased_PET_0.5_10_5_22-07-31 18:59.log')
FedAvg_WARP = os.path.join(base_dir,
                           'log/FedAvg/sst_s223_100_100_100/bert-base-uncased_WRAP_0.5_10_5_22-07-31 18:58.log')
FedAvg_Proto = os.path.join(base_dir,
                            'log/FedAvg/sst_s223_100_100_100/bert-base-uncased_proto_0.5_10_5_22-07-31 16:24.log')
FedAvg_ours = os.path.join(base_dir,
                           'log/FedAvg/sst_s223_100_100_100/bert-base-uncased_anchor_0.5_10_5_22-07-31 16:23.log')

centralized_d = {
    # 'FT': centralized_FT,
    # 'PET': centralized_PET,
    # 'SoftVerbalizer': centralized_WARP,
    # 'ProtoVerbalizer': centralized_proto,
    # 'ours': centralized_anchor
}

FedAvg_d = {
    'FT': FedAvg_FT,
    'PET': FedAvg_PET,
    'SoftVerbalizer': FedAvg_WARP,
    'ProtoVerbalizer': FedAvg_Proto,
    'ours': FedAvg_ours
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


len1 = 21
for key, val in centralized_d.items():
    with open(val, 'r') as f:
        lines = f.readlines()
        ls = process(lines)
        plt.plot(ls[:len1], label=key)
plt.legend()
plt.xticks(range(0, len1, 5))
plt.xlabel('Communication rounds')
plt.ylabel('F1 score (%) on validation dataset')
plt.title(r'SST-5')
plt.show()

for key, val in FedAvg_d.items():
    with open(val, 'r') as f:
        len2 = 51 if key in ['ours'] else 101
        # len2 = 101
        lines = f.readlines()
        ls = process(lines)
        # ls = smooth_curve(ls, delta=3)
        plt.plot(ls[:len2], label=key)
plt.legend()
plt.xlabel('Communication rounds')
plt.ylabel('F1 score (%) on validation dataset')
plt.title(r'SST-5/K=100/$\alpha$=0.5')
plt.show()
