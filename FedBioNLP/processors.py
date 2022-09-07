from torchvision import datasets as tv_datasets
from torchvision import transforms
import h5py
import json
import numpy as np
from .tokenizers import *
from .datasets import *
from .models import *
from .utils.fl_utils import generate_idxes_dirichlet, generate_idxes_kmeans, generate_idxes_group

logger = logging.getLogger(os.path.basename(__file__))
base_dir = os.path.expanduser('~/FedTransformers')


def process_dataset(dataset_name, model_name, split_type, n_clients, parser_args):
    res = {}
    clients = list(range(n_clients))
    train_datasets = {}
    test_datasets = {}
    n_classes = None
    transform_aug = None
    transform_normal = None
    task_type = None
    doc_index = None
    train_doc_index = None
    test_doc_index = None

    # load dataset
    if dataset_name == 'MNIST':
        n_classes = 10
        my_train_dataset = my_test_dataset = ImageDataset

        transform_aug = transform_normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = tv_datasets.FashionMNIST(root=os.path.join(base_dir, 'data'), train=True, download=True)
        test_dataset = tv_datasets.FashionMNIST(root=os.path.join(base_dir, 'data'), train=False, download=True)
        train_args = [train_dataset.data, np.array(train_dataset.targets)]
        test_args = [test_dataset.data, np.array(test_dataset.targets)]
    elif dataset_name == 'CIFAR10':
        n_classes = 10
        my_train_dataset = my_test_dataset = ImageDataset

        transform_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = tv_datasets.CIFAR10(root=os.path.join(base_dir, 'data'), train=True, download=True)
        test_dataset = tv_datasets.CIFAR10(root=os.path.join(base_dir, 'data'), train=False, download=True)
        train_args = [train_dataset.data, np.array(train_dataset.targets)]
        test_args = [test_dataset.data, np.array(test_dataset.targets)]
        model = ImageConvNet()
    elif dataset_name == 'MNIST*USPS':
        n_classes = 10

        my_train_dataset = ImageCCSADataset
        my_test_dataset = ImageDataset
        model = ImageConvCCSANet()

        transform_aug = transform_normal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_MNIST_dataset = tv_datasets.MNIST(root=os.path.join(base_dir, 'data'), train=True, download=True)
        test_MNIST_dataset = tv_datasets.MNIST(root=os.path.join(base_dir, 'data'), train=False, download=True)
        train_MNIST_data = train_MNIST_dataset.data
        train_MNIST_data = np.expand_dims(train_MNIST_data, -1).repeat(3, -1)
        test_MNIST_data = test_MNIST_dataset.data
        test_MNIST_data = np.expand_dims(test_MNIST_data, -1).repeat(3, -1)

        train_USPS_dataset = tv_datasets.USPS(root=os.path.join(base_dir, 'data'), train=True, download=True)
        test_USPS_dataset = tv_datasets.USPS(root=os.path.join(base_dir, 'data'), train=False, download=True)
        train_USPS_data = train_USPS_dataset.data
        train_USPS_data = np.expand_dims(train_USPS_data, -1).repeat(3, -1)
        test_USPS_data = test_USPS_dataset.data
        test_USPS_data = np.expand_dims(test_USPS_data, -1).repeat(3, -1)

        train_args = [train_USPS_data, train_MNIST_data, train_USPS_dataset.targets, train_MNIST_dataset.targets]
        test_args = [test_MNIST_data, test_MNIST_dataset.targets]


    elif dataset_name == 'CIFAR100':
        n_classes = 100
        my_train_dataset = my_test_dataset = ImageDataset

        transform_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = tv_datasets.CIFAR100(root=os.path.join(base_dir, 'data'),
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
        test_dataset = tv_datasets.CIFAR100(root=os.path.join(base_dir, 'data'),
                                            train=False,
                                            transform=transforms.ToTensor())
        train_args = [train_dataset.data, np.array(train_dataset.targets)]
        test_args = [test_dataset.data, np.array(test_dataset.targets)]
    else:
        my_dataset = NLPDataset
        data_file = os.path.join(base_dir, f'data/{dataset_name}_data.h5')
        partition_file = os.path.join(base_dir, f'fednlp_data/feature_skew_partition_files/{dataset_name}_partition.h5')
        embedding_file = os.path.join(base_dir, f'fednlp_data/embedding_files/{dataset_name}_partition.h5')

        with h5py.File(data_file, 'r+') as df:
            attributes = json.loads(df["attributes"][()])

            label_vocab = attributes['label_vocab']
            index_list = attributes['index_list']
            train_idx = attributes['train_index_list']
            test_idx = attributes['test_index_list']
            task_type = attributes['task_type']
            if 'doc_index' in attributes:
                doc_index = attributes['doc_index']
            else:
                doc_index = {str(i): 0 for i in index_list}

            # ls["attributes"][()] = json.dumps(attributes)

            data = []
            targets = []
            unique_docs = set()
            # specifically for wikiner, ploner
            if task_type == 'name_entity_recognition':
                tokenizer = NLP_tokenizer
                for idx in df['X'].keys():
                    sentence = df['X'][idx][()]
                    sentence = [i.decode('UTF-8') for i in sentence]
                    data.append(''.join(sentence))

                    label = df['Y'][idx][()]
                    label = [i.decode('UTF-8') for i in label]
                    targets.append(label)
                data = tokenizer(data)
                seq_length = len(data[0])
                targets = [target + ['O' for _ in range(seq_length - len(target))] for target in targets]
                targets = np.array([[label_vocab[w] for w in target] for target in targets])
            # specifically for squad_1.1
            elif task_type == 'reading_comprehension':
                tokenizer = NLP_tokenizer
                for i in df['context_X'].keys():
                    question_components = []
                    question = df['question_X'][i][()].decode('UTF-8')
                    answer_start = df['Y'][i][()][0]
                    answer_end = df['Y'][i][()][1]
                    answer = df['context_X'][i][()].decode('UTF-8')[answer_start: answer_end]

                    question_components.append(question)
                    question_components.append(answer)
                    data.append(" ".join(question_components))
            # specifically for cnn_dailymail, cornell_movie_dialogue
            elif task_type == 'sequence_to_sequence':
                tokenizer = sc_tokenizer
                for i in df['Y'].keys():
                    sentence = df['Y'][i][()].decode('UTF-8')
                    data.append(sentence)
            # specifically for semeval_2010_task8...

            elif task_type == 'relation_extraction':
                n_classes = attributes['num_labels']
                model = REModel(model_name, n_classes, parser_args.num_gcn_layers, parser_args.gradient_reverse)

                # model = RELatentGCNModel(model_name, n_classes)
                # if parser_args.n_clusters:
                #     cluster_models = [REModel(model_name, n_classes, parser_args.num_gcn_layers) for _ in
                #                       range(parser_args.n_clusters)]
                #     res.update({'cluster_models': cluster_models})
                # if parser_args.horizon:
                #     model = REHorizonModel(model_name, n_classes)
                # if parser_args.GRL:
                #     model = REGRLModel(model_name, n_classes)
                # if parser_args.MMD:
                #     model = REMMDModel(model_name, n_classes)
                # if parser_args.CCSA:
                #     model = RECCSAModel(model_name, n_classes)
                if parser_args.alg == 'GSN':
                    model = REGSNModel(model_name, n_classes, parser_args.num_gcn_layers, parser_args.gradient_reverse)
                if parser_args.alg == 'SCL':
                    model = RESCLModel(model_name, n_classes, parser_args.num_gcn_layers)
                # model = RelationExtractionHorizonBERT(model_name, n_classes)

                my_train_dataset = my_test_dataset = model.dataset
                n_classes = attributes['num_labels']

                try:
                    tokenizer = re_dep_tokenizer
                    args = {'text': [],
                            'e_text': [],
                            'dep_text': [],
                            'dep_e_text': [],
                            'dependency': [],
                            'doc': [],
                            'label': []}
                    for idx in index_list:
                        text = df['text'][str(idx)][()].decode('UTF-8')
                        e_text = df['e_text'][str(idx)][()].decode('UTF-8')
                        dep_text = df['dep_text'][str(idx)][()].decode('UTF-8')
                        dep_e_text = df['dep_e_text'][str(idx)][()].decode('UTF-8')
                        dependency = df['dependency'][str(idx)][()].decode('UTF-8')
                        label = df['label'][str(idx)][()].decode('UTF-8')
                        doc = doc_index[str(idx)]

                        unique_docs.add(doc)

                        args['text'].append(text)
                        args['e_text'].append(e_text)
                        args['dep_text'].append(dep_text)
                        args['dep_e_text'].append(dep_e_text)
                        args['dependency'].append(dependency)
                        args['label'].append(label_vocab[label])
                        args['doc'].append(doc)
                    args = tokenizer(args, model_name, parser_args.mlm_method, parser_args.mlm_prob, parser_args.K_LCA)
                except:
                    tokenizer = re_tokenizer
                    args = {'text': [],
                            'label': [],
                            'doc': []}

                    for idx in index_list:
                        sentence = df['X'][str(idx)][()].decode('UTF-8')
                        label = df['Y'][str(idx)][()].decode('UTF-8')
                        doc = doc_index[str(idx)]

                        args['text'].append(sentence)
                        args['label'].append(label_vocab[label])
                        args['doc'].append(doc)
                        unique_docs.add(doc)
                    args = tokenizer(args, model_name)


            # specifically for 20news, agnews, sst_2, sentiment140, semeval_2010_task8
            elif task_type == 'text_classification':
                args = {'text': [],
                        'label': [],
                        'doc': []}

                n_classes = attributes['num_labels']
                model = SCModel(model_name, n_classes)
                if parser_args.SCL:
                    model = SCSCLModel(model_name, n_classes)
                tokenizer = sc_tokenizer
                my_train_dataset = my_test_dataset = model.dataset

                for idx in index_list:
                    sentence = df['X'][str(idx)][()].decode('UTF-8')
                    args['text'].append(sentence)
                    label = df['Y'][str(idx)][()].decode('UTF-8')
                    args['label'].append(label_vocab[label])
                    doc = doc_index[str(idx)]
                    args['doc'].append(doc)
                    unique_docs.add(doc)

                args = tokenizer(args, model_name, max_seq_length)
            else:
                raise NotImplementedError

    train_args = {k: [v[i] for i in train_idx] for k, v in args.items()}
    test_args = {k: [v[i] for i in test_idx] for k, v in args.items()}
    num_train, num_test = len(train_idx), len(test_idx)

    centralized_train_dataset = my_train_dataset(train_args, num_train, n_classes, transform_aug, train_doc_index)
    centralized_test_dataset = my_test_dataset(test_args, num_test, n_classes, transform_normal, test_doc_index)
    logger.info(f'number of classes: {n_classes}')
    logger.info(f'number of samples of train/test dataset: {num_train}/{num_test}')

    if split_type == 'centralized':
        clients = [0]
        train_datasets = {0: centralized_train_dataset}
        test_datasets = {0: centralized_test_dataset}
        return clients, train_datasets, test_datasets, centralized_train_dataset, centralized_test_dataset, model, res
    elif split_type == 'idx_split':
        n_docs = len(unique_docs)
        clients = list(range(n_docs))
        logger.info(f'number of classes: {n_classes}')
        logger.info(f'unique docs: {unique_docs}')

        for d in unique_docs:
            cur_train_idx = [idx for idx in train_idx if doc_index[str(idx)] == d]
            cur_test_idx = [idx for idx in test_idx if doc_index[str(idx)] == d]
            train_args = {k: [v[idx] for idx in cur_train_idx] for k, v in args.items()}
            test_args = {k: [v[idx] for idx in cur_test_idx] for k, v in args.items()}
            num_train, num_test = len(cur_train_idx), len(cur_test_idx)

            train_dataset = my_train_dataset(train_args, num_train, n_classes, transform_aug, train_doc_index)
            train_datasets.update({d: train_dataset})
            test_dataset = my_test_dataset(test_args, num_test, n_classes, transform_normal, test_doc_index)
            test_datasets.update({d: test_dataset})

            logger.info(f'doc: {d}, number of samples of train/test dataset: {num_train}/{num_test}')
        return clients, train_datasets, test_datasets, centralized_train_dataset, centralized_test_dataset, model, res
    elif split_type == 'label_shift':
        logger.info('start splitting data according to label shift.')
        if not parser_args.n_clusters:
            train_idxes = generate_idxes_dirichlet(train_args['label'], n_clients, n_classes, parser_args.beta)
            test_idxes = generate_idxes_dirichlet(test_args['label'], n_clients, n_classes, parser_args.beta)
        else:
            train_idxes = generate_idxes_group(train_args['label'], n_clients, n_classes, parser_args.beta,
                                               n_clients // parser_args.n_clusters, parser_args.seed)
            test_idxes = generate_idxes_group(test_args['label'], n_clients, n_classes, parser_args.beta,
                                              n_clients // parser_args.n_clusters, parser_args.seed)
        for i in range(n_clients):
            train_idx = train_idxes[i]
            num_train = len(train_idx)
            client_train_args = {k: [v[idx] for idx in train_idx] for k, v in train_args.items()}
            train_dataset = my_train_dataset(client_train_args, num_train, n_classes, transform_aug, train_doc_index)
            train_datasets.update({i: train_dataset})

            test_idx = test_idxes[i]
            num_test = len(test_idx)
            client_test_args = {k: [v[idx] for idx in test_idx] for k, v in test_args.items()}
            test_dataset = my_test_dataset(client_test_args, num_test, n_classes, transform_normal, test_doc_index)
            test_datasets.update({i: test_dataset})
            logger.info(f'client: {i}, number of samples of train/test dataset: {num_train}/{num_test}')

        return clients, train_datasets, test_datasets, centralized_train_dataset, centralized_test_dataset, model, res
    elif split_type == 'feature shift':
        logger.info('Start splitting data according to feature shift.')
        train_idxes, test_idxes = generate_idxes_kmeans(data_file, partition_file, embedding_file, task_type,
                                                        n_clients)
        for i in range(n_clients):
            train_idx = train_idxes[i]
            test_idx = test_idxes[i]
            train_data, train_targets = data[train_idx], targets[train_idx]
            test_data, test_targets = data[test_idx], targets[test_idx]
            train_dataset = my_dataset(train_data, train_targets, transform_aug)
            train_datasets.update({int(idx): train_dataset})
            test_dataset = my_dataset(test_data, test_targets, transform_normal)
            test_datasets.update({int(idx): test_dataset})
        return clients, train_datasets, test_datasets, centralized_train_dataset, centralized_test_dataset
    else:
        raise Exception("Invalid split type.")
