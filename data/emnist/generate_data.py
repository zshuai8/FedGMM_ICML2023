"""
Download EMNIST dataset, and splits it among clients
"""
import argparse
import os
import pickle

from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor

from utils import split_dataset_by_labels, pathological_non_iid_split, split_and_reform_dataset

# TODO: remove this after new release of torchvision
EMNIST.url = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"

N_CLASSES = 62
RAW_DATA_PATH = "raw_data/"
PATH = "all_data/"


def save_data(l, path_):
    with open(path_, 'wb') as f:
        pickle.dump(l, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_tasks',
        help='number of tasks/clients;',
        type=int,
        required=True
    )
    parser.add_argument(
        '--pathological_split',
        help='if selected, the dataset will be split as in'
             '"Communication-Efficient Learning of Deep Networks from Decentralized Data";'
             'i.e., each client will receive `n_shards` of dataset, where each shard contains at most two classes',
        action='store_true'
    )
    parser.add_argument(
        '--n_shards',
        help='number of shards given to each clients/task; ignored if `--pathological_split` is not used;'
             'default is 2',
        type=int,
        default=2
    )
    parser.add_argument(
        '--n_components',
        help='number of components/clusters; default is -1',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--alpha',
        help='parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar; '
             'default is 0.2',
        type=float,
        default=0.2)
    parser.add_argument(
        '--s_frac',
        help='fraction of the dataset to be used; default: 0.2;',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--tr_frac',
        help='fraction in training set; default: 0.8;',
        type=float,
        default=0.8
    )
    parser.add_argument(
        '--val_frac',
        help='fraction in validation set (from train set); default: 0.0;',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--test_tasks_frac',
        help='fraction of tasks / clients not participating to the training; default is 0.0',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes; default is 12345',
        type=int,
        default=12345
    )
    parser.add_argument(
        '--distribution_shift',
        help='if selected, the dataset will be split such that each component',
        action='store_true'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    transform = Compose(
        [ToTensor(),
         ]
    )

    dataset = ConcatDataset([
        EMNIST(
            root=RAW_DATA_PATH,
            split="byclass",
            download=True,
            train=True,
            transform=transform
        ),
        EMNIST(root=RAW_DATA_PATH,
               split="byclass",
               download=True,
               train=False,
               transform=transform)
    ])

    if args.pathological_split:
        clients_indices = \
            pathological_non_iid_split(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_classes_per_client=args.n_shards,
                frac=args.s_frac,
                seed=args.seed
            )
    elif args.distribution_shift:
        clients_indices, rotation_idx, client_permute, label_projection = \
            split_and_reform_dataset(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_clusters=args.n_components,
                alpha=args.alpha,
                frac=args.s_frac,
                seed=args.seed,
            )
    else:
        clients_indices = \
            split_dataset_by_labels(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_clusters=args.n_components,
                alpha=args.alpha,
                frac=args.s_frac,
                seed=args.seed,
            )

    if args.test_tasks_frac > 0:
        train_clients_indices, test_clients_indices = \
            train_test_split(clients_indices, test_size=args.test_tasks_frac, random_state=args.seed)
    else:
        train_clients_indices, test_clients_indices = clients_indices, []

    os.makedirs(os.path.join(PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(PATH, "test"), exist_ok=True)

    if args.distribution_shift:
        save_data(rotation_idx, os.path.join(PATH, "rotations.pkl"))
        save_data(client_permute, os.path.join(PATH, "clients_permute.pkl"))
        save_data(label_projection, os.path.join(PATH, "label_projection.pkl"))
    for mode, clients_indices in [('train', train_clients_indices), ('test', test_clients_indices)]:
        for client_id, indices in enumerate(clients_indices):
            client_path = os.path.join(PATH, mode, "task_{}".format(client_id))
            os.makedirs(client_path, exist_ok=True)

            train_indices, test_indices = \
                train_test_split(
                    indices,
                    train_size=args.tr_frac,
                    random_state=args.seed
                )

            if args.val_frac > 0:
                train_indices, val_indices = \
                    train_test_split(
                        train_indices,
                        train_size=1.-args.val_frac,
                        random_state=args.seed
                    )

                save_data(val_indices, os.path.join(client_path, "val.pkl"))
            print("Train size: {} Test size: {}".format(len(train_indices), len(test_indices)))

            save_data(train_indices, os.path.join(client_path, "train.pkl"))
            save_data(test_indices, os.path.join(client_path, "test.pkl"))


if __name__ == "__main__":
    main()
