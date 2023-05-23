import os
import argparse
import pickle
import numpy as np

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import ConcatDataset

from sklearn.model_selection import train_test_split

# from utils import split_dataset_by_labels, pathological_non_iid_split, pachinko_allocation_split
SEED = 12345
RAW_DATA_PATH = "raw_data/"
PATH = "all_data/"
saved_path = "all_data_unseen"


transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

dataset =\
    ConcatDataset([
        CIFAR10(root=RAW_DATA_PATH, download=True, train=True, transform=transform),
        CIFAR10(root=RAW_DATA_PATH, download=False, train=False, transform=transform)
    ])
all_indices = []
for client_id, indices in enumerate(np.arange(80)):
    client_path = os.path.join(PATH, "train", "task_{}".format(client_id))
    with open(client_path+"/test.pkl", "rb") as f:
        a = pickle.load(f)
        all_indices.append(a)
all_indices = np.concatenate(all_indices)
print(len(all_indices))
os.makedirs("all_data_unseen/train", exist_ok=True)

os.makedirs("all_data_unseen/train/task_0", exist_ok=True)
os.makedirs("all_data_unseen/test", exist_ok=True)
import numpy as np

# Assume your array is named 'data'
training_data = np.random.choice(all_indices, size=1600, replace=False)
test_data = np.setdiff1d(all_indices, training_data, assume_unique=True)
test_data = test_data[:400]
print(training_data)

with open("all_data_unseen/train/task_0/train.pkl", 'wb') as f:
    pickle.dump(training_data, f)

with open("all_data_unseen/train/task_0/test.pkl", 'wb') as f:
    pickle.dump(test_data, f)