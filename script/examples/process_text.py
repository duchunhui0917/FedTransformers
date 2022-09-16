import copy
import json
import os
import random

import h5py
import pandas as pd
from FedBioNLP import aug_back_translate, aug_tfidf, aug_label_reverse, aug_label_random, aug_sent_len
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

base_dir = os.path.expanduser('~/FedTransformers')


# nlp = StanfordCoreNLP(os.path.join(base_dir, 'stanford-corenlp-4.4.0'))


# def statistic_word_freq(ls):
#     for x in ls:
#         for word in x:

def amazon_review_txt_to_json(file_name, num=2500):
    train_num = num * 4 // 5
    n = file_name.split('.')[0]
    train_tsv_path = n + f'_{num}_train.tsv'
    test_tsv_path = n + f'_{num}_test.tsv'
    with open(file_name, 'r') as f:
        lines = f.readlines()
    text1 = []
    text2 = []
    text4 = []
    text5 = []
    t = tqdm(lines)
    for line in t:
        line = line.strip()
        pos = line.find(':')
        if pos != -1:
            if line[pos - 5:pos] == 'score':
                score = int(float(line[pos + 2:]))
            if line[pos - 4:pos] == 'text':
                text = line[pos + 2:]
        else:
            if score == 1:
                text1.append(text)
            elif score == 2:
                text2.append(text)
            elif score == 4:
                text4.append(text)
            elif score == 5:
                text5.append(text)
    text1 = random.sample(text1, num)
    text2 = random.sample(text2, num)
    text4 = random.sample(text4, num)
    text5 = random.sample(text5, num)

    train_text_false = text1[:train_num] + text2[:train_num]
    test_text_false = text1[train_num:] + text2[train_num:]
    train_label_false = [0] * len(train_text_false)
    test_label_false = [0] * len(test_text_false)

    train_text_true = text4[:train_num] + text5[:train_num]
    test_text_true = text4[train_num:] + text5[train_num:]
    train_label_true = [1] * len(train_text_true)
    test_label_true = [1] * len(test_text_true)

    train_data = {'text': train_text_false + train_text_true,
                  'label': train_label_false + train_label_true}

    test_data = {'text': test_text_false + test_text_true,
                 'label': test_label_false + test_label_true}

    train_df = pd.DataFrame(train_data)
    train_df.to_csv(train_tsv_path, sep='\t', index=False, header=False)
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(test_tsv_path, sep='\t', index=False, header=False)

    print('hello')


def amazon_txt2h5():
    h5_path = os.path.join(base_dir, f'data/imdb_data.h5')
    hf = h5py.File(h5_path, 'w')
    doc_index = {}
    index_list = []
    train_index = []
    test_index = []
    idx = 0

    train_data = os.path.join(base_dir, f'data/sentiment_classification/IMDB_train.tsv')
    with open(train_data, 'r', encoding='utf8') as f:
        lines = f.readlines()

    t = tqdm(lines)
    for line in t:
        ls = line.strip().split('\t')
        hf[f'/X/{idx}'] = ' '.join(ls[:-1])
        hf[f'/Y/{idx}'] = ls[-1]
        index_list.append(idx)
        train_index.append(idx)
        doc_index[idx] = 0
        idx += 1

    test_data = os.path.join(base_dir, f'data/sentiment_classification/IMDB_test.tsv')
    with open(test_data, 'r', encoding='utf8') as f:
        lines = f.readlines()

    t = tqdm(lines)
    for line in t:
        ls = line.strip().split('\t')
        hf[f'/X/{idx}'] = ' '.join(ls[:-1])
        hf[f'/Y/{idx}'] = ls[-1]
        index_list.append(idx)
        test_index.append(idx)
        doc_index[idx] = 0
        idx += 1
    attributes = {'doc_index': doc_index,
                  'label_vocab': {
                      'negative': 0,
                      'positive': 1
                  },
                  'num_labels': 2,
                  'index_list': index_list,
                  'train_index_list': train_index,
                  'test_index_list': test_index,
                  'task_type': 'text_classification'}
    hf['/attributes'] = json.dumps(attributes)
    hf.close()


# amazon_txt2h5()


# amazon_review_txt_to_json(os.path.join(base_dir, 'data/amazon_review/Movies_&_TV.txt'), 1500)
# amazon_review_txt2h5('Movies_&_TV_1500')
def amazon_csv2h5():
    path = os.path.join(base_dir, 'data/sentiment_classification/IMDB Dataset.csv')
    train_path = os.path.join(base_dir, 'data/sentiment_classification/IMDB_train.tsv')
    test_path = os.path.join(base_dir, 'data/sentiment_classification/IMDB_test.tsv')

    df = pd.read_csv(path)
    df.columns = ['text', 'label']
    df_true = df[df['label'] == 'positive']
    df_false = df[df['label'] == 'negative']
    df_true_train = df_true.sample(frac=0.8)
    df_true_test = df_true[~df_true.index.isin(df_true_train.index)]

    df_false_train = df_false.sample(frac=0.8)
    df_false_test = df_false[~df_false.index.isin(df_false_train.index)]

    df_train = pd.merge(df_true_train, df_false_train, how='outer')
    df_test = pd.merge(df_true_test, df_false_test, how='outer')

    df_train.to_csv(train_path, sep='\t', index=False, header=False)
    df_test.to_csv(test_path, sep='\t', index=False, header=False)


# amazon_csv2h5()

def process_i2b2_lines(lines, rel):
    e1, label, e2 = rel.split('||')
    tmp = e1.split()
    pos00, pos01 = tmp[-2], tmp[-1]
    line_num, pos00 = pos00.split(':')
    _, pos01 = pos01.split(':')

    tmp = e2.split()
    pos10, pos11 = tmp[-2], tmp[-1]
    _, pos10 = pos10.split(':')
    _, pos11 = pos11.split(':')

    pos00, pos01, pos10, pos11 = int(pos00), int(pos01), int(pos10), int(pos11)

    line = lines[int(line_num) - 1]
    tokens = line.strip().split()
    tokens[pos00] = ' <e1> ' + tokens[pos00]
    tokens[pos01] = tokens[pos01] + ' </e1> '
    tokens[pos10] = ' <e2> ' + tokens[pos10]
    tokens[pos11] = tokens[pos11] + ' </e2> '
    x = ' '.join(tokens)
    y = label.split('\"')[1]
    return x, y


def process_re_lines(line, name):
    if name in ['GAD', 'EU-ADR']:
        x, y = line.strip().split('\t')
        pos00, pos10 = x.index('@GENE$'), x.index('@DISEASE$')
        pos01, pos11 = pos00 + 6, pos10 + 9
    elif name in ['PGR_Q1', 'PGR_Q2']:
        _, x, _, _, _, _, pos00, pos01, pos10, pos11, y = line.strip().split('\t')
        pos00, pos01, pos10, pos11 = int(pos00), int(pos01), int(pos10), int(pos11)
        y = '0' if y == 'False' else '1'
    elif name in ['CoMAGC']:
        x, _, pos00, pos01, _, pos10, pos11, y = line.strip().split('\t')
        pos00, pos01, pos10, pos11 = int(pos00), int(pos01) + 1, int(pos10), int(pos11) + 1
        y = '0' if y == 'Negative_regulation' else '1'
    elif name in ['AIMed', 'BioInfer', 'HPRD50', 'IEPA', 'LLL', 'merged']:
        _, _, y, x, _ = line.strip().split('\t')
        pos00, pos10 = x.index('PROTEIN1'), x.index('PROTEIN2')
        pos01, pos11 = pos00 + 8, pos10 + 8
        y = '0' if y == 'False' else '1'
    else:
        raise NotImplementedError

    text = x
    label = y
    ent1 = ' <e1> ' + x[pos00:pos01] + ' </e1> '
    ent2 = ' <e2> ' + x[pos10:pos11] + ' </e2> '

    # ent1 = ' <e1> entity </e1> '
    # ent2 = ' <e2> entity </e2> '

    if pos00 < pos10:
        e_text = x[:pos00] + ent1 + x[pos01:pos10] + ent2 + x[pos11:]
    else:
        e_text = x[:pos10] + ent2 + x[pos11:pos00] + ent1 + x[pos01:]
    return text, e_text, label


def assemble(e_ls):
    dep_ls = copy.deepcopy(e_ls)
    dep_ls.remove('<e1>')
    dep_ls.remove('</e1>')
    dep_ls.remove('<e2>')
    dep_ls.remove('</e2>')
    dep_text = ' '.join(dep_ls)
    return dep_text


def h52txt(file_name):
    h5_path = os.path.join(base_dir, f'data/{file_name}_data.h5')
    train_path = os.path.join(base_dir, f'data/{file_name}-train.tsv')
    test_path = os.path.join(base_dir, f'data/{file_name}-test.tsv')
    hf = h5py.File(h5_path, 'r')
    attributes = json.loads(hf["attributes"][()])

    train_index_list = attributes['train_index_list']
    test_index_list = attributes['test_index_list']
    for path, index_list in zip([train_path, test_path], [train_index_list, test_index_list]):
        texts = [hf['text'][str(idx)][()].decode('UTF-8') for idx in index_list]
        e_texts = [hf['e_text'][str(idx)][()].decode('UTF-8') for idx in index_list]
        labels = [hf['label'][str(idx)][()].decode('UTF-8') for idx in index_list]
        lines = []
        for text, e_text, label in zip(texts, e_texts, labels):
            line = text + '\t' + e_text + '\t' + label + '\n'
            lines.append(line)
        with open(path, 'w') as f:
            f.writelines(lines)

    hf.close()


def bio_RE_txt2h5(file_name, aug_method=None, process=True):
    dir_name = file_name.split('-')[0]

    if aug_method:
        train_path = os.path.join(base_dir, f'data/bio_RE/{dir_name}/{file_name}_{aug_method}-train.tsv')
        test_path = os.path.join(base_dir, f'data/bio_RE/{dir_name}/{file_name}_{aug_method}-test.tsv')
        h5_path = os.path.join(base_dir, f'data/{file_name}_{aug_method}_data.h5')
    else:
        train_path = os.path.join(base_dir, f'data/bio_RE/{dir_name}/{file_name}-train.tsv')
        test_path = os.path.join(base_dir, f'data/bio_RE/{dir_name}/{file_name}-test.tsv')
        h5_path = os.path.join(base_dir, f'data/{file_name}_data.h5')

    print(h5_path)

    hf = h5py.File(h5_path, 'w')
    doc_index = {}
    index_list = []
    train_index = []
    test_index = []
    idx = 0

    for cur_index, path in zip([train_index, test_index], [train_path, test_path]):
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            if not process:
                text, e_text, label = line.strip().split('\t')
            else:
                text, e_text, label = process_re_lines(line, file_name)
            dep_e_ls = nlp.word_tokenize(e_text)
            dep_e_text = ' '.join(dep_e_ls)
            dep_text = assemble(dep_e_ls)
            dep_ls = nlp.word_tokenize(dep_text)
            dependency = nlp.dependency_parse(dep_text)

            l1 = len(dep_e_ls) - 4
            l2 = max([max(x[1], x[2]) for x in dependency])
            if l1 != l2:
                print(l1, l2)
                print(dep_e_ls)
                print(dep_ls)
                print(dependency)

            hf[f'/text/{idx}'] = text
            hf[f'/e_text/{idx}'] = e_text
            hf[f'/dep_text/{idx}'] = dep_text
            hf[f'/dep_e_text/{idx}'] = dep_e_text
            hf[f'/dependency/{idx}'] = json.dumps(dependency)
            hf[f'/label/{idx}'] = label

            doc_index[idx] = 0
            index_list.append(idx)
            cur_index.append(idx)
            idx += 1
    atts = ['text', 'e_text', 'dep_text', 'dep_e_text', 'dependency', 'label']
    attributes = {
        'doc_index': doc_index,
        'label_vocab': {'0': 0, '1': 1},
        'num_labels': 2,
        'index_list': index_list,
        'train_index_list': train_index,
        'test_index_list': test_index,
        'task_type': 'relation_extraction',
        'atts': json.dumps(atts)
    }
    hf['/attributes'] = json.dumps(attributes)
    hf.close()


def aug_text(file_name, aug_method='aug', mode='train', process=True):
    dir_name = file_name.split('_')[0]
    path = os.path.join(base_dir, f'data/bio_RE/{dir_name}/{file_name}-{mode}.tsv')
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    e_texts = []
    labels = []
    for line in lines:
        if not process:
            text, e_text, label = line.strip().split('\t')
        else:
            text, e_text, label = process_re_lines(line, file_name)
        e_texts.append(e_text)
        labels.append(label)

    aug_texts = e_texts
    aug_labels = labels
    if aug_method == 'back_translate':
        aug_texts = aug_back_translate(dir_name, file_name, mode)
    elif aug_method == 'tfidf':
        aug_texts = aug_tfidf(e_texts, dir_name)
    elif aug_method == 'label_reverse':
        aug_labels = aug_label_reverse(labels)
    elif aug_method == 'label_random':
        aug_labels = aug_label_random(labels)
    elif aug_method == 'sent_len':
        aug_labels = aug_sent_len(e_texts, 30)

    liens = []
    for aug_text, aug_label in zip(aug_texts, aug_labels):
        text = aug_text
        for x in ['<e1> ', ' </e1>', '<e2> ', ' </e2>']:
            if x not in text:
                print(text)
            text = text.replace(x, '')
        liens.append(text + '\t' + aug_text + '\t' + aug_label + '\n')
    aug_path = os.path.join(base_dir, f'data/bio_RE/{dir_name}/{file_name}_{aug_method}-{mode}.tsv')
    print(aug_path)
    with open(aug_path, 'w', encoding='utf8') as f:
        f.writelines(liens)


def i2b2_txt2h5(name):
    h5_path = os.path.join(base_dir, f'data/{name}_data.h5')
    hf = h5py.File(h5_path, 'w')
    doc_index = {}
    index_list = []
    train_index = []
    test_index = []
    idx = 0

    dir_path = os.path.join(base_dir, f'data/i2b2/{name}/train_data')
    file_names = os.listdir(dir_path)
    for file_name in file_names:
        train_data = os.path.join(base_dir, f'data/i2b2/{name}/train_data/{file_name}')
        train_target = os.path.join(base_dir, f'data/i2b2/{name}/train_target/{file_name}')
        with open(train_data, 'r', encoding='utf8') as f:
            lines = f.readlines()
        with open(train_target, 'r', encoding='utf8') as f:
            rels = f.readlines()

        for rel in rels:
            x, y = process_i2b2_lines(lines, rel)
            hf[f'/X/{idx}'] = x
            hf[f'/Y/{idx}'] = y
            index_list.append(idx)
            train_index.append(idx)
            doc_index[idx] = 0
            idx += 1

    dir_path = os.path.join(base_dir, f'data/i2b2/{name}/test_data')
    file_names = os.listdir(dir_path)
    for file_name in file_names:
        test_data = os.path.join(base_dir, f'data/i2b2/{name}/test_data/{file_name}')
        test_target = os.path.join(base_dir, f'data/i2b2/{name}/test_target/{file_name}')
        with open(test_data, 'r', encoding='utf8') as f:
            lines = f.readlines()
        with open(test_target, 'r', encoding='utf8') as f:
            rels = f.readlines()

        for rel in rels:
            x, y = process_i2b2_lines(lines, rel)
            hf[f'/X/{idx}'] = x
            hf[f'/Y/{idx}'] = y
            index_list.append(idx)
            test_index.append(idx)
            doc_index[idx] = 0
            idx += 1
    attributes = {'doc_index': doc_index,
                  'label_vocab': {
                      'TrIP': 0,
                      'TrWP': 1,
                      'TrCP': 2,
                      'TrAP': 3,
                      'TrNAP': 4,
                      'TeRP': 5,
                      'TeCP': 6,
                      'PIP': 7
                  },
                  'num_labels': 8,
                  'index_list': index_list,
                  'train_index_list': train_index,
                  'test_index_list': test_index,
                  'task_type': 'relation_extraction'}
    hf['/attributes'] = json.dumps(attributes)
    hf.close()


# i2b2_txt2h5(['Partners'])


def txt2h5():
    name = 'sentiment'
    h5_path = os.path.join(base_dir, f'data/{name}_data.h5')
    hf = h5py.File(h5_path, 'w')
    doc_index = {}
    index_list = []
    train_index = []
    eval_index = []
    test_index = []
    idx = 0

    train_data = os.path.join(base_dir, f'data/{name}/train_org.tsv')
    with open(train_data, 'r', encoding='utf8') as f:
        lines = f.readlines()

    t = tqdm(lines)
    for line in t:
        ls = line.strip().split('\t')
        hf[f'/X/{idx}'] = ls[1]
        hf[f'/Y/{idx}'] = ls[0]
        index_list.append(idx)
        train_index.append(idx)
        doc_index[idx] = 0
        idx += 1

    eval_data = os.path.join(base_dir, f'data/{name}/dev_org.tsv')
    with open(eval_data, 'r', encoding='utf8') as f:
        lines = f.readlines()
    t = tqdm(lines)
    for line in t:
        ls = line.strip().split('\t')
        hf[f'/X/{idx}'] = ls[1]
        hf[f'/Y/{idx}'] = ls[0]
        index_list.append(idx)
        eval_index.append(idx)
        doc_index[idx] = 0
        idx += 1

    test_data = os.path.join(base_dir, f'data/{name}/test_org.tsv')
    with open(test_data, 'r', encoding='utf8') as f:
        lines = f.readlines()
    t = tqdm(lines)
    for line in t:
        ls = line.strip().split('\t')
        hf[f'/X/{idx}'] = ls[1]
        hf[f'/Y/{idx}'] = ls[0]
        index_list.append(idx)
        test_index.append(idx)
        doc_index[idx] = 0
        idx += 1

    attributes = {
        'doc_index': doc_index,
        'label_vocab': {
            'Negative': 0,
            'Positive': 1
        },
        'num_labels': 2,
        'index_list': index_list,
        'train_index_list': train_index,
        'eval_index_list': eval_index,
        'test_index_list': test_index
    }
    hf['/attributes'] = json.dumps(attributes)
    hf.close()


# txt2h5()


def csv2tsv(name_list):
    for name in name_list:
        for i in ['train', 'test']:
            file_path = os.path.join(base_dir, f'data/bio_RE/{name}/{name}-{i}.csv')

            df = pd.read_csv(file_path, encoding='utf8')
            df = df[(df['passage'].str.contains('PROTEIN1')) & (df['passage'].str.contains('PROTEIN2'))]
            file_path = file_path.split('.')[0] + '.tsv'
            df.to_csv(file_path, index=False, sep='\t', encoding='utf-8')


def split_pair():
    path = os.path.join(base_dir, 'data/combined/paired/test_paired.tsv')
    with open(path, 'r') as f:
        lines = f.readlines()
    new_lines = [lines[i] for i in range(len(lines)) if i % 2 == 1]
    df = pd.DataFrame()
    Sentiment = []
    Text = []
    batch_id = []
    for line in new_lines:
        s, t, b = line.strip().split('\t')
        Sentiment.append(s)
        Text.append(t)
        batch_id.append(b)
    df['Sentiment'] = Sentiment
    df['Text'] = Text
    df['batch_id'] = batch_id
    file_path = os.path.join(base_dir, 'data/combined/paired/test_new.tsv')
    df.to_csv(file_path, index=False, sep='\t', encoding='utf-8')


def replace_text():
    path = os.path.join(base_dir, 'data/sentiment/train_org.tsv')
    with open(path, 'r') as f:
        lines = f.readlines()
    df = pd.DataFrame()
    Sentiment = []
    Text = []

    for line in lines:
        s, t = line.strip().split('\t')
        ls = t.split()
        idx = random.sample(range(len(ls)), 1)[0]
        ls[idx] = ' '
        t = ' '.join(ls)
        Sentiment.append(s)
        Text.append(t)
    df['Sentiment'] = Sentiment
    df['Text'] = Text
    file_path = os.path.join(base_dir, 'data/sentiment/train_positive.tsv')
    df.to_csv(file_path, index=False, sep='\t', encoding='utf-8')

    path = os.path.join(base_dir, 'data/sentiment/dev_org.tsv')
    with open(path, 'r') as f:
        lines = f.readlines()
    df = pd.DataFrame()
    Sentiment = []
    Text = []

    for line in lines:
        s, t = line.strip().split('\t')
        ls = t.split()
        idx = random.sample(range(len(ls)), 1)[0]
        ls[idx] = ' '
        t = ' '.join(ls)
        Sentiment.append(s)
        Text.append(t)
    df['Sentiment'] = Sentiment
    df['Text'] = Text
    file_path = os.path.join(base_dir, 'data/sentiment/dev_positive.tsv')
    df.to_csv(file_path, index=False, sep='\t', encoding='utf-8')

    path = os.path.join(base_dir, 'data/sentiment/test_org.tsv')
    with open(path, 'r') as f:
        lines = f.readlines()
    df = pd.DataFrame()
    Sentiment = []
    Text = []

    for line in lines:
        s, t = line.strip().split('\t')
        ls = t.split()
        idx = random.sample(range(len(ls)), 1)[0]
        ls[idx] = ' '
        t = ' '.join(ls)
        Sentiment.append(s)
        Text.append(t)
    df['Sentiment'] = Sentiment
    df['Text'] = Text
    file_path = os.path.join(base_dir, 'data/sentiment/test_positive.tsv')
    df.to_csv(file_path, index=False, sep='\t', encoding='utf-8')


def mask_text():
    new_path = os.path.join(base_dir, 'data/sentiment/train_new.tsv')
    org_path = os.path.join(base_dir, 'data/sentiment/train_org.tsv')

    mask_path1 = os.path.join(base_dir, 'data/sentiment/train_mask1')
    mask_path2 = os.path.join(base_dir, 'data/sentiment/train_mask2')

    with open(org_path, 'r') as f:
        org_lines = f.readlines()
    with open(new_path, 'r') as f:
        new_lines = f.readlines()
    text1 = []
    text2 = []
    t = tqdm(zip(org_lines, new_lines))
    for o, n in t:
        ls1 = []
        ls2 = []
        _, ot = o.strip().split('\t')
        _, nt, _ = n.strip().split('\t')
        ot = ot.split(' ')
        nt = nt.split(' ')
        ol = len(ot)
        nl = len(nt)
        i = 0
        j = 0
        while i < ol and j < nl:
            if ot[i] == nt[j]:
                ls1.append(ot[i])
                ls2.append('[MASK]')
                i += 1
                j += 1
            else:
                ls1.append('[MASK]')
                ls2.append(ot[i])
                i += 1
                jj = j
                while i < nl and jj < ol:
                    if nt[i] == ot[jj]:
                        j = jj
                        break
                    jj += 1
        text1.append(' '.join(ls1) + '\n')
        text2.append(' '.join(ls2) + '\n')

    with open(mask_path1, 'w') as f:
        f.writelines(text1)
    with open(mask_path2, 'w') as f:
        f.writelines(text2)


mask_text()
# split_pair()
# bio_RE_txt2h5('GAD', process=True)
# bio_RE_txt2h5('EU-ADR', process=True)
# bio_RE_txt2h5('CoMAGC', process=True)
# bio_RE_txt2h5('PGR_Q1', process=True)
# bio_RE_txt2h5('PGR_Q2', process=True)

# bio_RE_txt2h5('AIMed', process=True)
# bio_RE_txt2h5('BioInfer', process=True)
# bio_RE_txt2h5('HPRD50', process=True)
# bio_RE_txt2h5('IEPA', process=True)
# bio_RE_txt2h5('LLL', process=True)

# bio_RE_txt2h5('AIMed_2|2', aug_method='back_translate', process=False)
# bio_RE_txt2h5('AIMed_2|2', aug_method='label_reverse', process=False)
# bio_RE_txt2h5('AIMed_2|2', aug_method='tfidf', process=False)
# bio_RE_txt2h5('AIMed_2|2', aug_method='label_random', process=False)
# bio_RE_txt2h5('AIMed_2|2', aug_method='sent_len', process=False)

# aug_text('AIMed_2|2', aug_method='back_translate', mode='test', process=False)
# aug_text('AIMed_2|2', aug_method='label_random', mode='train', process=False)
# aug_text('AIMed_1|2_balance', aug_method='label_reverse', mode='train', process=False)
# aug_text('AIMed_2|2', aug_method='sent_len', mode='test', process=False)
# h52txt('PGR_Q1')
# h52txt('AIMed_1|2_balance')

i2b2_txt2h5('Partners')
