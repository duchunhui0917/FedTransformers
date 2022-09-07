import os
import matplotlib.pyplot as plt
from common_utils import smooth_curve

base_dir = os.path.expanduser('~/FedTransformers')


def process(file):
    with open(os.path.join(base_dir, file), 'r') as f:
        lines = f.readlines()
    ls_f1 = []
    flag = False

    for line in lines:
        x = line.strip().split()
        if len(x) == 6 and x[-2] == 'f1:' and flag:
            ls_f1.append(float(x[-1]))
            flag = False
        if ' '.join(x[-3:]) == 'test centralized dataset':
            flag = True

    return ls_f1


central_mobilebert = 'log/centralized/PGR_Q1/google_mobilebert-uncased_22-06-17 18:37.log'
central_distilbert = 'log/centralized/PGR_Q1/distilbert-base-uncased_22-06-19 17:10.log'
central_bert = 'log/centralized/PGR_Q1/bert-base-uncased_22-06-17 22:42.log'
central_large_bert = 'log/centralized/PGR_Q1/bert-large-uncased_22-06-17 21:23.log'
central_biobert = 'log/centralized/PGR_Q1/dmis-lab_biobert-v1.1_22-06-16 09:32.log'
central_scibert = 'log/centralized/PGR_Q1/allenai_scibert_scivocab_uncased_22-06-17 20:44.log'
central_twitter = 'log/centralized/PGR_Q1/cardiffnlp_twitter-roberta-base-sentiment_22-06-17 20:10.log'
central_clinical = 'log/centralized/PGR_Q1/emilyalsentzer_Bio_ClinicalBERT_22-06-17 18:34.log'
central_unbert = 'log/centralized/PGR_Q1/bert-base-uncased_unload_22-06-20 09:56.log'

FedAvg_dot5_10_5_mobilebert = 'log/FedAvg/PGR_Q1/google_mobilebert-uncased_0.5_10_5_22-06-17 17:24.log'
FedAvg_dot5_10_5_distilbert = 'log/FedAvg/PGR_Q1/distilbert-base-uncased_0.5_10_5_22-06-19 11:30.log'
FedAvg_dot5_10_5_bert = 'log/FedAvg/PGR_Q1/bert-base-uncased_0.5_10_5_22-06-18 10:32.log'
FedAvg_dot5_10_5_large_bert = 'log/FedAvg/PGR_Q1/'
FedAvg_dot5_10_5_biobert = 'log/FedAvg/PGR_Q1/dmis-lab_biobert-v1.1_0.5_10_5_22-06-19 11:35.log'
FedAvg_dot5_10_5_scibert = 'log/FedAvg/PGR_Q1/allenai_scibert_scivocab_uncased_0.5_10_5_22-06-19 11:21.log'
FedAvg_dot5_10_5_twitter = 'log/FedAvg/PGR_Q1/cardiffnlp_twitter-roberta-base-sentiment_0.5_10_5_22-06-19 11:26.log'
FedAvg_dot5_10_5_clinical = 'log/FedAvg/PGR_Q1/emilyalsentzer_Bio_ClinicalBERT_0.5_10_5_22-06-19 13:38.log'
FedAvg_dot5_10_5_unbert = 'log/FedAvg/PGR_Q1/bert-base-uncased_unload_0.5_10_5_22-06-20 11:01.log'

FedAvg_dot5_10_3_mobilebert = 'log/FedAvg/PGR_Q1/google_mobilebert-uncased_0.5_10_3_22-06-18 01:24.log'
FedAvg_dot5_10_3_distilbert = 'log/FedAvg/PGR_Q1/distilbert-base-uncased_0.5_10_3_22-06-18 01:15.log'
FedAvg_dot5_10_3_bert = 'log/FedAvg/PGR_Q1/bert-base-uncased_0.5_10_3_22-06-18 01:16.log'
FedAvg_dot5_10_3_large_bert = 'log/FedAvg/PGR_Q1/'
FedAvg_dot5_10_3_biobert = 'log/FedAvg/PGR_Q1/dmis-lab_biobert-v1.1_0.5_10_3_22-06-19 11:37.log'
FedAvg_dot5_10_3_scibert = 'log/FedAvg/PGR_Q1/allenai_scibert_scivocab_uncased_0.5_10_3_22-06-18 01:18.log'
FedAvg_dot5_10_3_twitter = 'log/FedAvg/PGR_Q1/cardiffnlp_twitter-roberta-base-sentiment_0.5_10_3_22-06-18 00:58.log'
FedAvg_dot5_10_3_clinical = 'log/FedAvg/PGR_Q1/emilyalsentzer_Bio_ClinicalBERT_0.5_10_3_22-06-18 10:27.log'
FedAvg_dot5_10_3_unbert = ''

FedAvg_dot5_10_1_mobilebert = 'log/FedAvg/PGR_Q1/google_mobilebert-uncased_0.5_10_1_22-06-18 15:12.log'
FedAvg_dot5_10_1_distilbert = 'log/FedAvg/PGR_Q1/distilbert-base-uncased_0.5_10_1_22-06-18 15:16.log'
FedAvg_dot5_10_1_bert = 'log/FedAvg/PGR_Q1/bert-base-uncased_0.5_10_1_22-06-18 15:13.log'
FedAvg_dot5_10_1_large_bert = 'log/FedAvg/PGR_Q1/bert-large-uncased_0.5_10_1_22-06-18 16:34.log'
FedAvg_dot5_10_1_biobert = 'log/FedAvg/PGR_Q1/dmis-lab_biobert-v1.1_0.5_10_1_22-06-19 14:24.log'
FedAvg_dot5_10_1_scibert = 'log/FedAvg/PGR_Q1/allenai_scibert_scivocab_uncased_0.5_10_1_22-06-18 16:15.log'
FedAvg_dot5_10_1_twitter = 'log/FedAvg/PGR_Q1/cardiffnlp_twitter-roberta-base-sentiment_0.5_10_1_22-06-18 15:14.log'
FedAvg_dot5_10_1_clinical = 'log/FedAvg/PGR_Q1/emilyalsentzer_Bio_ClinicalBERT_0.5_10_1_22-06-19 01:30.log'
FedAvg_dot5_10_1_unbert = 'log/FedAvg/PGR_Q1/bert-base-uncased_unload_0.5_10_5_22-06-20 11:01.log'

FedAvg_1_10_5_mobilebert = ''
FedAvg_1_10_5_distilbert = 'log/FedAvg/PGR_Q1/distilbert-base-uncased_1_10_5_22-06-20 01:03.log'
FedAvg_1_10_5_bert = 'log/FedAvg/PGR_Q1/bert-base-uncased_1_10_5_22-06-20 00:59.log'
FedAvg_1_10_5_large_bert = ''
FedAvg_1_10_5_biobert = ''
FedAvg_1_10_5_scibert = 'log/FedAvg/PGR_Q1/allenai_scibert_scivocab_uncased_1_10_5_22-06-20 00:57.log'
FedAvg_1_10_5_twitter = 'log/FedAvg/PGR_Q1/cardiffnlp_twitter-roberta-base-sentiment_1_10_5_22-06-20 01:01.log'
FedAvg_1_10_5_clinical = ''
FedAvg_1_10_5_unbert = 'log/FedAvg/PGR_Q1/bert-base-uncased_unload_1_10_5_22-06-20 09:53.log'

central_files = [central_mobilebert, central_distilbert, central_bert, central_large_bert,
                 central_biobert, central_scibert, central_twitter, central_clinical, central_unbert]
FedAvg_dot5_10_5_files = [FedAvg_dot5_10_5_mobilebert, FedAvg_dot5_10_5_distilbert, FedAvg_dot5_10_5_bert,
                          FedAvg_dot5_10_5_large_bert, FedAvg_dot5_10_5_biobert, FedAvg_dot5_10_5_scibert,
                          FedAvg_dot5_10_5_twitter, FedAvg_dot5_10_5_clinical, FedAvg_dot5_10_5_unbert]
FedAvg_dot5_10_3_files = [FedAvg_dot5_10_3_mobilebert, FedAvg_dot5_10_3_distilbert, FedAvg_dot5_10_3_bert,
                          FedAvg_dot5_10_3_large_bert, FedAvg_dot5_10_3_biobert, FedAvg_dot5_10_3_scibert,
                          FedAvg_dot5_10_3_twitter, FedAvg_dot5_10_3_clinical, FedAvg_dot5_10_3_unbert]
FedAvg_dot5_10_1_files = [FedAvg_dot5_10_1_mobilebert, FedAvg_dot5_10_1_distilbert, FedAvg_dot5_10_1_bert,
                          FedAvg_dot5_10_1_large_bert, FedAvg_dot5_10_1_biobert, FedAvg_dot5_10_1_scibert,
                          FedAvg_dot5_10_1_twitter, FedAvg_dot5_10_1_clinical, FedAvg_dot5_10_1_unbert]
FedAvg_1_10_5_files = [FedAvg_1_10_5_mobilebert, FedAvg_1_10_5_distilbert, FedAvg_1_10_5_bert,
                       FedAvg_1_10_5_distilbert, FedAvg_1_10_5_biobert, FedAvg_1_10_5_scibert,
                       FedAvg_1_10_5_twitter, FedAvg_1_10_5_clinical, FedAvg_1_10_5_unbert]
labels = ['MobileBERT', 'DistilBERT', 'BERT', 'BERT-large', 'BioBERT', 'SciBERT', 'RoB-TW', 'BioClinical',
          'BERT w/o PT']
markers = ['.', 'o', '*', 'D', '|', 's', 'v', '+', '1']


def plot_epoch():
    nodes = [5, 2, 6]
    x1 = [central_files[n] for n in nodes]
    x2 = [FedAvg_dot5_10_1_files[n] for n in nodes]
    x3 = [FedAvg_dot5_10_3_files[n] for n in nodes]
    x4 = [FedAvg_dot5_10_5_files[n] for n in nodes]
    x5 = [labels[n] for n in nodes]
    x6 = [markers[n] for n in nodes]

    tmp = zip(x1, x2, x3, x4, x5, x6)

    for central_file, FedAvg_1_file, FedAvg_3_file, FedAvg_5_file, label, marker in tmp:
        central_f1 = process(central_file)
        max_central_f1 = max(central_f1)
        FedAvg_1_f1 = process(FedAvg_1_file)
        max_FedAvg_1_f1 = max(FedAvg_1_f1)
        FedAvg_3_f1 = process(FedAvg_3_file)
        max_FedAvg_3_f1 = max(FedAvg_3_f1)
        FedAvg_5_f1 = process(FedAvg_5_file)
        max_FedAvg_5_f1 = max(FedAvg_5_f1)
        ls = [max_central_f1, max_FedAvg_1_f1, max_FedAvg_3_f1, max_FedAvg_5_f1]
        plt.plot(ls, label=label, marker=marker)

        print(f'{label}: {max_central_f1}, {max_FedAvg_1_f1}, {max_FedAvg_3_f1}, {max_FedAvg_5_f1}')

    plt.legend()
    plt.xticks(range(4), ['centralized', 'E=1', 'E=3', 'E=5'])
    plt.title('PGR')
    plt.ylabel('F1')
    plt.show()


def plot_alpha():
    nodes = [5, 2, 6]
    x1 = [central_files[n] for n in nodes]
    x2 = [FedAvg_1_10_5_files[n] for n in nodes]
    x3 = [FedAvg_dot5_10_5_files[n] for n in nodes]
    x4 = [labels[n] for n in nodes]
    x5 = [markers[n] for n in nodes]

    tmp = zip(x1, x2, x3, x4, x5)

    for central_file, FedAvg_1_10_5_file, FedAvg_dot_10_5_file, label, marker in tmp:
        central_f1 = process(central_file)
        max_central_f1 = max(central_f1)
        FedAvg_1_10_5_f1 = process(FedAvg_1_10_5_file)
        max_FedAvg_1_10_5_f1 = max(FedAvg_1_10_5_f1)
        FedAvg_dot_10_5_f1 = process(FedAvg_dot_10_5_file)
        max_FedAvg_dot_10_5_f1 = max(FedAvg_dot_10_5_f1)
        ls = [max_central_f1, max_FedAvg_1_10_5_f1, max_FedAvg_dot_10_5_f1]
        plt.plot(ls, label=label, marker=marker)

        print(f'{label}: {max_central_f1}, {max_FedAvg_1_10_5_f1}, {max_FedAvg_dot_10_5_f1}')
    plt.legend()
    plt.xticks(range(3), ['centralized', r'$\alpha$=1', r'$\alpha$=0.5'])
    plt.title('PGR')
    plt.ylabel('F1')
    plt.show()


def plot_f1_curve():
    nodes = [3, 2, 4, 1]
    x = [FedAvg_1_10_5_files[n] for n in nodes]
    y = [labels[n] for n in nodes]

    tmp = zip(x, y)

    for x, y in tmp:
        x = process(x)
        x = smooth_curve(x, 1)
        plt.plot(x, label=y)

    plt.legend()
    plt.title('PGR')
    plt.xlabel('Communication Rounds')
    plt.ylabel('F1')
    plt.show()


plot_epoch()
plot_alpha()
# plot_f1_curve()
