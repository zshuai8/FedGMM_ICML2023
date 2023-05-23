"""
Process Femnist dataset, and splits it among clients
"""
import os
import time
import random
import argparse
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from scipy.stats import beta
from torchvision import models


from sklearn.model_selection import train_test_split
import pickle

RAW_DATA_PATH = os.path.join("intermediate", "data_as_tensor_by_writer")
TARGET_PATH = "all_data_unseen/"
SEED = 1234

def save_data(l, path_):
    with open(path_, 'wb') as f:
        pickle.dump(l, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--s_frac',
        help='fraction of data to be used; default is 0.3',
        type=float,
        default=0.3
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
        '--train_tasks_frac',
        help='fraction of tasks / clients  participating to the training; default is 1.0',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=SEED,
        required=False
    )

    return parser.parse_args()


def save_task(dir_path, train_data, train_targets, test_data, test_targets, val_data=None, val_targets=None):
    r"""
    save (`train_data`, `train_targets`) in {dir_path}/train.pt,
    (`val_data`, `val_targets`) in {dir_path}/val.pt
    and (`test_data`, `test_targets`) in {dir_path}/test.pt

    :param dir_path:
    :param train_data:
    :param train_targets:
    :param test_data:
    :param test_targets:
    :param val_data:
    :param val_targets
    """
    torch.save((train_data, train_targets), os.path.join(dir_path, "train.pt"))
    torch.save((test_data, test_targets), os.path.join(dir_path, "test.pt"))

    if (val_data is not None) and (val_targets is not None):
        torch.save((val_data, val_targets), os.path.join(dir_path, "val.pt"))

def main():
    args = parse_args()

    rng_seed = (args.seed if (args.seed is not None and args.seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)

    n_tasks = int(len(os.listdir(RAW_DATA_PATH)) * args.s_frac)
    file_names_list = os.listdir(RAW_DATA_PATH)
    rng.shuffle(file_names_list)

    file_names_list = file_names_list[:n_tasks]
    rng.shuffle(file_names_list)

    os.makedirs(os.path.join(TARGET_PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_PATH, "test"), exist_ok=True)

    print("generating data..")

    label_projection = torch.LongTensor(np.random.permutation(62))

    alpha = 2.0
    beta = 2.0
    overall_data = []
    transform = Normalize((0.1307,), (0.3081,))
    # Generate 10 samples from the beta distribution
    samples = np.random.beta(alpha, beta, size=len(file_names_list))
    for idx, file_name in enumerate(tqdm(file_names_list)):
        if idx < int(args.train_tasks_frac * n_tasks):
            mode = "train"
        else:
            mode = "test"

        data, targets = torch.load(os.path.join(RAW_DATA_PATH, file_name))

        rotation_idx = np.random.binomial(1, samples[idx], data.shape[0])
        rotation_idx = np.where(rotation_idx)
        inverted_images = 1 - data[rotation_idx]
        flipped_images = torch.flip(inverted_images, [2])
        # sacled_images = data[rotation_idx] * 1
        rotated_images = torch.rot90(flipped_images, 1, [1, 2])
        trans_data = rotated_images
        data[rotation_idx] = trans_data
        data = transform(data)
        targets[rotation_idx] = label_projection[targets[rotation_idx]]
        train_data, test_data, train_targets, test_targets =\
            train_test_split(
                data,
                targets,
                train_size=args.tr_frac,
                random_state=args.seed
            )
        
        if args.val_frac > 0:
            train_data, val_data, train_targets, val_targets = \
                train_test_split(
                    train_data,
                    train_targets,
                    train_size=1.-args.val_frac,
                    random_state=args.seed
                )
        else:
            val_data, val_targets = None, None
        overall_data.append(train_data)
        save_path = os.path.join(TARGET_PATH, mode, f"task_{idx}")
        os.makedirs(save_path, exist_ok=True)

        save_task(
            dir_path=save_path,
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets,
            val_data=val_data,
            val_targets=val_targets
        )

    # Get the projected values of the input matrix

    # save_data(rotations, os.path.join('all_data', "rotations.pkl"))
    # save_data(label_projection, os.path.join('all_data', "label_projection.pkl"))

if __name__ == "__main__":
    main()



