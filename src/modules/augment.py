import os

base_dir = os.path.expanduser('~/FedTransformers')


def sst2_augment():
    positive_path = os.path.join(base_dir, 'data/mask+')
    negative_path = os.path.join(base_dir, 'data/mask-')

    with open(positive_path, 'r') as f:
        positive_text = f.readlines()
    with open(negative_path, 'r') as f:
        negative_text = f.readlines()
    return positive_text, negative_text


def sentiment_augment(idxes=None):
    positive_path = os.path.join(base_dir, 'data/sentiment/train_positive.tsv')
    positive_mask_path = os.path.join(base_dir, 'data/sentiment/train_positive_mask.tsv')
    negative_path = os.path.join(base_dir, 'data/sentiment/train_negative.tsv')
    negative_mask_path = os.path.join(base_dir, 'data/sentiment/train_negative_mask')

    with open(positive_path, 'r') as f:
        text = f.readlines()
        positive_text = [x.strip().split('\t')[1] for x in text]
    with open(positive_mask_path, 'r') as f:
        text = f.readlines()
        positive_mask_text = [x.strip().split('\t')[1] for x in text]
    with open(negative_path, 'r') as f:
        text = f.readlines()
        negative_text = [x.strip() for x in text]
    with open(negative_mask_path, 'r') as f:
        text = f.readlines()
        negative_mask_text = [x.strip() for x in text]
    if idxes:
        positive_text = [positive_text[i] for i in range(len(positive_text)) if i in idxes]
        positive_mask_text = [positive_mask_text[i] for i in range(len(positive_mask_text)) if i in idxes]
        negative_text = [negative_text[i] for i in range(len(negative_text)) if i in idxes]
        negative_mask_text = [negative_mask_text[i] for i in range(len(negative_mask_text)) if i in idxes]

    return positive_text, positive_mask_text, negative_text, negative_mask_text
