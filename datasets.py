import os
import pickle
import string

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms.functional as FT
import math

def grayscale(imgs):
    return imgs.mean(dim=1, keepdim=True)

def random_rotation(x):
    x = torch.rot90(x, 2, [1, 2])
    # x = 1 - x
    return x

def random_invert(imgs):
    return 1-imgs

def random_horizontal_flip(imgs):
    if len(imgs.size()) < 4:
        return imgs.transpose(1, 2)
    else:
        return imgs.transpose(2, 3)


class TabularDataset(Dataset):
    """
    Constructs a torch.utils.Dataset object from a pickle file;
    expects pickle file stores tuples of the form (x, y) where x is vector and y is a scalar

    Attributes
    ----------
    data: iterable of tuples (x, y)

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path):
        """
        :param path: path to .pkl file
        """
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64), idx

class SubFEMNIST(Dataset):
    """
    Constructs a subset of FEMNIST dataset corresponding to one client;
    Initialized with the path to a `.pt` file;
    `.pt` file is expected to hold a tuple of tensors (data, targets) storing the images and there corresponding labels.

    Attributes
    ----------
    transform
    data: iterable of integers
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path):
        # self.transform = Compose([
        #     ToTensor(),
        #     Normalize((0.1307,), (0.3081,))
        # ])

        self.data, self.targets = torch.load(path)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # img = np.uint8(img.numpy() * 255)
        # img = Image.fromarray(img, mode='L')
        #
        # if self.transform is not None:
        #     img = self.transform(img)

        img = img.view(-1,1,28,28)
        return img, target, index

# class SubFEMNIST(Dataset):
#     """
#     Constructs a subset of EMNIST dataset from a pickle file;
#     expects pickle file to store list of indices
#
#     Attributes
#     ----------
#     indices: iterable of integers
#     transform
#     data
#     targets
#
#     Methods
#     -------
#     __init__
#     __len__
#     __getitem__
#     """
#
#     def __init__(self, path, emnist_data=None, emnist_targets=None, transform=None, rotation=True):
#         """
#         :param path: path to .pkl file; expected to store list of indices
#         :param emnist_data: EMNIST dataset inputs
#         :param emnist_targets: EMNIST dataset labels
#         :param transform:
#         """
#         # with open(path, "rb") as f:
#         #     self.indices = pickle.load(f)
#         self.transform = Compose([
#             ToTensor(),
#             Normalize((0.1307,), (0.3081,))
#         ])
#
#         self.data, self.targets = torch.load(path)
#
#
# #         if emnist_data is None or emnist_targets is None:
# #             self.data, self.targets = emnist_data, emnist_targets
# #         else:
#         # print(rotation)
#         # raise
# #         if rotation:
# #             transform_image = T.Compose([
# #                 T.ToPILImage(),
# #                 T.RandomRotation(degrees=90),
# #                 T.RandomInvert(1),
# #                 T.RandomHorizontalFlip(1),
# #                 T.ToTensor()
# #             ])
# #             with open('data/femnist/all_data/rotations.pkl', "rb") as f:
# #                 rotation_idx = pickle.load(f)
#
# #                 x_trans = FEMNIST_T_Dataset(self.data, transform_image)
# #                 transformed = DataLoader(x_trans, batch_size=len(rotation_idx))
# #                 # Iterate over the batches in the DataLoader
# #                 for batch in transformed:
# #                     # The `batch` tensor will contain a single sample
# #                     transformed_data = batch[0]
# #                 transformed_data = transformed_data.to(torch.float64)
# #                 self.data[rotation_idx] = transformed_data
# #                 # self.data[rotation_idx] = transform_image(self.data[rotation_idx].view(b,1,w,h))
# #                 with open('data/femnist/all_data/label_projection.pkl', "rb") as f2:
# #                     label_projection = torch.LongTensor(pickle.load(f2))
# #                     self.targets[rotation_idx] = label_projection[self.targets[rotation_idx]]
#
#     def __len__(self):
#         return self.data.size(0)
#
#     def __getitem__(self, index):
#         img, target = self.data[index], int(self.targets[index])
#         img = np.uint8(img.numpy() * 255)
#         img = Image.fromarray(img, mode='L')
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, target, index

class SubFEMNIST(Dataset):
    """
    Constructs a subset of FEMNIST dataset corresponding to one client;
    Initialized with the path to a `.pt` file;
    `.pt` file is expected to hold a tuple of tensors (data, targets) storing the images and there corresponding labels.

    Attributes
    ----------
    transform
    data: iterable of integers
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path):
        # self.transform = Compose([
        #     ToTensor(),
        #     Normalize((0.1307,), (0.3081,))
        # ])

        self.data, self.targets = torch.load(path)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # img = np.uint8(img.numpy() * 255)
        # img = Image.fromarray(img, mode='L')
        #
        # if self.transform is not None:
        #     img = self.transform(img)

        return img, target, index


# class SubFEMNIST(Dataset):
#     """
#     Constructs a subset of FEMNIST dataset corresponding to one client;
#     Initialized with the path to a `.pt` file;
#     `.pt` file is expected to hold a tuple of tensors (data, targets) storing the images and there corresponding labels.
#
#     Attributes
#     ----------
#     transform
#     data: iterable of integers
#     targets
#
#     Methods
#     -------
#     __init__
#     __len__
#     __getitem__
#     """
#     def __init__(self, path,  emnist_data=None, emnist_targets=None, transform=None, rotation=False, project=None):
#         # self.transform = Compose([
#         #     ToTensor(),
#         # ])
#         self.data, self.targets = torch.load(path)
#
#         if dist_shift:
#             with open('data/emnist_r/all_data/rotations.pkl', "rb") as f:
#                 rotation_idx = pickle.load(f)
#                 x_trans = torch.utils.data.Subset(emnist_data_transformed, rotation_idx)
#                 emnist_data[rotation_idx] = x_trans[0]
#                 with open('data/emnist_r/all_data/label_projection.pkl', "rb") as f2:
#                     label_projection = torch.LongTensor(pickle.load(f2))
#                     emnist_targets[rotation_idx] = label_projection[emnist_targets[rotation_idx]]
#
#         if dist_shift:
#             self.transform = Compose([
#                 # Resize the images to a fixed size
#                 # T.ToPILImage(),
#                 T.RandomInvert(),
#                 T.RandomRotation(degrees=(0, 180)),
#                 # # Randomly flip the images horizontally
#                 T.RandomHorizontalFlip(),
#                 # Convert the images to tensors
#                 # T.Resize((-1, 28, 28)),
#
#                 T.ToTensor()
#             ])
#
#
#
#
#
#     def __len__(self):
#         return self.data.size(0)
#
#     def __getitem__(self, index):
#         img, target = self.data[index], int(self.targets[index])
#
#         img = np.uint8(img.numpy() * 255)
#         img = Image.fromarray(img, mode='L')
#
#         # if self.distrib:
#         #     img = self.transform(img)
#
#         return img, target, index


class SubEMNIST(Dataset):
    """
    Constructs a subset of EMNIST dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path, emnist_data=None, emnist_targets=None, transform=None, rotation=False, project=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        # if transform is None:
        #     self.transform =\
        #         Compose([
        #             ToTensor(),
        #             Normalize((0.1307,), (0.3081,))
        #         ])

        if emnist_data is None or emnist_targets is None:
            print("this should not be called")
            self.data, self.targets = get_emnist(rotation=rotation)
        else:
            self.data, self.targets = emnist_data, emnist_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # img = Image.fromarray(img.numpy(), mode='L')
        #
        # if self.transform is not None:
        #     img = self.transform(img)

        return img, target, index


class SubMNIST(Dataset):
    """
    Constructs a subset of EMNIST dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path, emnist_data=None, emnist_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    # Normalize((0.1307,), (0.3081,))
                ])

        if emnist_data is None or emnist_targets is None:
            self.data, self.targets = get_mnist()
        else:
            self.data, self.targets = emnist_data, emnist_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # img = Image.fromarray(img.numpy(), mode='L')
        #
        # if self.transform is not None:
        #     img = self.transform(img)

        return img, target, index


class SubMNIST9(Dataset):
    """
    Constructs a subset of EMNIST dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path=None, emnist_data=None, emnist_targets=None, transform=None, leave_9_out=True):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        # if transform is None:
        #     self.transform = \
        #         Compose([
        #             ToTensor(),
        #             # Normalize((0.1307,), (0.3081,))
        #         ])

        if emnist_data is None or emnist_targets is None:
            self.data, self.targets = get_mnist9(leave_9_out)
        else:
            self.data, self.targets = emnist_data, emnist_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # img = Image.fromarray(img.numpy(), mode='L')
        #
        # if self.transform is not None:
        #     img = self.transform(img)

        return img, target, index


class SubMNIST_REP(Dataset):
    """
    Constructs a subset of EMNIST dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path, emnist_data=None, emnist_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if emnist_data is None or emnist_targets is None:
            self.data, self.targets = get_mnist_rep()
        else:
            self.data, self.targets = emnist_data, emnist_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        return img, target, index


class RepDataset(Dataset):
    """
    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, data, targets, transform=None):
        if data is None or targets is None:
            raise ValueError('invalid data or targets')
        self.data, self.targets = data, targets

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        return img, target, index


class SubCIFAR10(Dataset):
    """
    Constructs a subset of CIFAR10 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path, cifar10_data=None, cifar10_targets=None, transform=None, emb_size=64):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param cifar10_data: Cifar-10 dataset inputs stored as torch.tensor
        :param cifar10_targets: Cifar-10 dataset labels stored as torch.tensor
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        # if transform is None:
        #     self.transform = \
        #         Compose([
        #             ToTensor(),
        #             Normalize(
        #                 (0.4914, 0.4822, 0.4465),
        #                 (0.2023, 0.1994, 0.2010)
        #             )
        #         ])

        if cifar10_data is None or cifar10_targets is None:
            self.data, self.targets = get_cifar10()
        else:
            self.data, self.targets = cifar10_data, cifar10_targets

        # self.dimension_red = nn.Linear(512, embedding_size)
        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]
    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # print(img.size())
        # img = Image.fromarray(img.cpu().numpy())
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # target = target

        return img, target, index


class SubCIFAR100(Dataset):
    """
    Constructs a subset of CIFAR100 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path, cifar100_data=None, cifar100_targets=None, transform=None, emb_size=64):
        """
        :param path: path to .pkl file; expected to store list of indices:
        :param cifar100_data: CIFAR-100 dataset inputs
        :param cifar100_targets: CIFAR-100 dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        # if transform is None:
        #     self.transform = \
        #         Compose([
        #             ToTensor(),
        #             Normalize(
        #                 (0.4914, 0.4822, 0.4465),
        #                 (0.2023, 0.1994, 0.2010)
        #             )
        #         ])

        if cifar100_data is None or cifar100_targets is None:
            self.data, self.targets = get_cifar100()

        else:
            self.data, self.targets = cifar100_data, cifar100_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # img = Image.fromarray(img.cpu().numpy())
        #
        # if self.transform is not None:
        #     img = self.transform(img)

        target = target

        return img, target, index


class CharacterDataset(Dataset):
    def __init__(self, file_path, chunk_len):
        """
        Dataset for next character prediction, each sample represents an input sequence of characters
         and a target sequence of characters representing to next sequence of the input
        :param file_path: path to .txt file containing the training corpus
        :param chunk_len: (int) the length of the input and target sequences
        """
        self.all_characters = string.printable
        self.vocab_size = len(self.all_characters)
        self.n_characters = len(self.all_characters)
        self.chunk_len = chunk_len

        with open(file_path, 'r') as f:
            self.text = f.read()

        self.tokenized_text = torch.zeros(len(self.text), dtype=torch.long)

        self.inputs = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)
        self.targets = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)

        self.__build_mapping()
        self.__tokenize()
        self.__preprocess_data()

    def __tokenize(self):
        for ii, char in enumerate(self.text):
            self.tokenized_text[ii] = self.char2idx[char]

    def __build_mapping(self):
        self.char2idx = dict()
        for ii, char in enumerate(self.all_characters):
            self.char2idx[char] = ii

    def __preprocess_data(self):
        for idx in range(self.__len__()):
            self.inputs[idx] = self.tokenized_text[idx:idx + self.chunk_len]
            self.targets[idx] = self.tokenized_text[idx + 1:idx + self.chunk_len + 1]

    def __len__(self):
        return max(0, len(self.text) - self.chunk_len)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], idx


def permute_label(targets, n_labels):
    dic = np.random.permutation(n_labels)
    for i in range(targets.size(0)):
        y = targets[i]
        targets[i] = dic[y]

    return targets

# def get_femnist(dist_shift=False, dp=False):
#     """
#     gets full (both train and test) EMNIST dataset inputs and labels;
#     the dataset should be first downloaded (see data/emnist/README.md)
#     :return:
#         emnist_data, emnist_targets
#     """
#     if dist_shift:
#         emnist_path = os.path.join("data", "femnist", "raw_data")
#         emnist_path_aug = os.path.join("data", "emnist_aug", "raw_data")
#     else:
#         emnist_path = os.path.join("data", "femnist", "raw_data")
#     assert os.path.isdir(emnist_path), "Download EMNIST dataset!!"
#     # Define a transform that will be applied to the images
#     d_transform = T.Compose([
#         # Resize the images to a fixed size
#         # T.ToPILImage(),
#         # T.Grayscale(),
#         # T.ColorJitter(brightness=.5, hue=.3),
#         T.RandomRotation(degrees=90),
#         T.RandomInvert(1),
#         # # Randomly flip the images horizontally
#         T.RandomHorizontalFlip(1),
#         # Convert the images to tensors
#         # T.Resize((-1, 28, 28)),
#
#         T.ToTensor()
#     ])
#     emnist_train = \
#         EMNIST(
#             root=emnist_path,
#             split="byclass",
#             download=True,
#             transform=None,
#             train=False
#         )
#     # subd = torch.utils.data.Subset(emnist_train, indices=torch.arange(10))
#     emnist_transform_train = \
#         EMNIST(
#             root=emnist_path,
#             split="byclass",
#             train=True,
#             transform=d_transform,
#             download=False
#         )
#     emnist_test = \
#         EMNIST(
#             root=emnist_path,
#             split="byclass",
#             download=True,
#             transform=T.ToTensor(),
#             train=True
#         )
#     emnist_transform_test = \
#         EMNIST(
#             root=emnist_path_aug,
#             split="byclass",
#             train=False,
#             transform=d_transform,
#             download=False
#         )
#     emnist_transform_test.transform = d_transform
#     emnist_data = \
#         torch.cat([
#             emnist_train.data,
#             emnist_test.data
#         ])
#     emnist_data_transformed = \
#         torch.cat([
#             emnist_transform_train.data,
#             emnist_transform_test.data
#         ])
#
#     emnist_targets = \
#         torch.cat([
#             emnist_train.targets,
#             emnist_test.targets
#         ])
#     if dist_shift:
#         with open('data/emnist_r/all_data/rotations.pkl', "rb") as f:
#             rotation_idx = pickle.load(f)
#             # x = emnist_data[rotation_idx]
#             # y = emnist_targets[rotation_idx]
#
#             # x = torch.rot90(x, 2, [1, 2])
#             # x = 1 - x
#             # y = permute_label(y, 62)
#             #
#             # emnist_data[rotation_idx] = x
#             # emnist_targets[rotation_idx] = y
#
#             # x_tranformed = d_transform(emnist_data_transformed)
#             # print(x_tranformed.size())
#
#             x_trans = torch.utils.data.Subset(emnist_data_transformed, rotation_idx)
#             emnist_data[rotation_idx] = x_trans[0]
#             if dp:
#                 with open('data/emnist_r/all_data/clients_permute.pkl', "rb") as f1:
#                     with open('data/emnist_r/all_data/label_projection.pkl', "rb") as f2:
#                         client_permutation = torch.LongTensor(pickle.load(f1))
#                         label_projection = torch.LongTensor(pickle.load(f2))
#
#                         emnist_targets[client_permutation] = label_projection[emnist_targets[client_permutation]]
#                         # emnist_data[rotation_idx] = x
#                         # emnist_targets[rotation_idx] = y
#             else:
#                 with open('data/emnist_r/all_data/label_projection.pkl', "rb") as f2:
#                     label_projection = torch.LongTensor(pickle.load(f2))
#                     emnist_targets[rotation_idx] = label_projection[emnist_targets[rotation_idx]]
#
#
#
#             # gray_img = T.Grayscale()
#             # jitter = T.ColorJitter(brightness=.5, hue=.3)
#             # inverter = T.RandomInvert()
#             # rotater = T.RandomRotation(degrees=(0, 180))
#             # print(torch.sum(x_trans[0][0]))
#
#             # emnist_targets[rotation_idx] = label_projection[y_i]
#             # for step, imgs in enumerate(x):
#             #     imgs = gray_img(imgs)
#             #     imgs = jitter(imgs)
#             #     imgs = inverter(imgs)
#             #     imgs = rotater(imgs)
#             #     imgs = jitter(imgs)
#             #     imgs = inverter(imgs)
#             #     imgs = rotater(imgs)
#             #     emnist_data[rotation_idx[step]] = imgs
#         #
#     return emnist_data, emnist_targets

def get_emnist(dist_shift=False, dp=False):
    """
    gets full (both train and test) EMNIST dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        emnist_data, emnist_targets
    """
    emnist_path = os.path.join("data", "emnist", "raw_data")
    assert os.path.isdir(emnist_path), "Download EMNIST dataset!!"
    # Define a transform that will be applied to the images
    emnist_train = \
        EMNIST(
            root=emnist_path,
            split="byclass",
            download=True,
            transform=None,
            train=True
        )
    emnist_test = \
        EMNIST(
            root=emnist_path,
            split="byclass",
            download=True,
            transform=T.ToTensor(),
            train=False
        )

    emnist_data = \
        torch.cat([
            emnist_train.data,
            emnist_test.data
        ])
    d_norm = Normalize((0.1307,), (0.3081,))
    # base = FEMNIST_T_Dataset(emnist_data.view(-1, 28, 28), d_base)
    emnist_targets = \
        torch.cat([
            emnist_train.targets,
            emnist_test.targets
        ])

    if dist_shift:
        with open('data/emnist/all_data/rotations.pkl', "rb") as f:
            rotation_idx = pickle.load(f)
            inverted_images = 255 - emnist_data[rotation_idx]
            flipped_images = torch.flip(inverted_images, [2])
            rotated_images = torch.rot90(flipped_images, 1, [1, 2])
            trans_data = rotated_images
            emnist_data[rotation_idx] = trans_data

            with open('data/emnist/all_data/label_projection.pkl', "rb") as f2:
                label_projection = torch.LongTensor(pickle.load(f2))
                emnist_targets[rotation_idx] = label_projection[emnist_targets[rotation_idx]]
    elif dp:
        with open('data/emnist/all_data/clients_permute.pkl', "rb") as f1:
            with open('data/emnist/all_data/label_projection.pkl', "rb") as f2:
                client_permutation = torch.LongTensor(pickle.load(f1))
                label_projection = torch.LongTensor(pickle.load(f2))
    #
    #             emnist_targets[client_permutation] = label_projection[emnist_targets[client_permutation]]
    # else:
        with open('data/emnist/all_data/rotations.pkl', "rb") as f:
            rotation_idx = pickle.load(f)
        with open('data/emnist/all_data/label_projection.pkl', "rb") as f2:
            label_projection = torch.LongTensor(pickle.load(f2))
            emnist_targets[rotation_idx] = label_projection[emnist_targets[rotation_idx]]
    emnist_data = emnist_data / 255
    # emnist_data = d_norm(emnist_data)
    return emnist_data, emnist_targets


def get_mnist(dist_shift=False, dp=False):
    """
    gets full (both train and test) EMNIST dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        emnist_data, emnist_targets
    """
    # emnist_path = os.path.join("data", "mnist", "raw_data")
    # assert os.path.isdir(emnist_path), "Download MNIST dataset!!"

    if dist_shift:
        emnist_path = os.path.join("data", "mnist", "raw_data")
        emnist_path_aug = os.path.join("data", "mnist_aug", "raw_data")
    else:
        emnist_path = os.path.join("data", "mnist", "raw_data")
    assert os.path.isdir(emnist_path), "Download MNIST dataset!!"
    # Define a transform that will be applied to the images
    d_transform = T.Compose([
        # Resize the images to a fixed size
        # T.ToPILImage(),
        T.Grayscale(),
        T.ColorJitter(brightness=.5, hue=.3),
        T.RandomInvert(),
        T.RandomRotation(degrees=(0, 180)),
        # # Randomly flip the images horizontally
        T.RandomHorizontalFlip(),
        # Convert the images to tensors
        # T.Resize((-1, 28, 28)),

        T.ToTensor()
    ])
    emnist_train = \
        MNIST(
            root=emnist_path_aug,
            download=True,
            transform=None,
            train=True
        )
    # subd = torch.utils.data.Subset(emnist_train, indices=torch.arange(10))
    emnist_transform_train = \
        MNIST(
            root=emnist_path,
            train=True,
            transform=d_transform,
            download=True
        )
    emnist_test = \
        MNIST(
            root=emnist_path_aug,
            download=True,
            transform=T.ToTensor(),
            train=True
        )
    emnist_transform_test = \
        MNIST(
            root=emnist_path,
            train=False,
            transform=d_transform,
            download=True
        )
    emnist_transform_test.transform = d_transform
    emnist_data = \
        torch.cat([
            emnist_train.data,
            emnist_test.data
        ])
    emnist_data_transformed = \
        torch.cat([
            emnist_transform_train.data,
            emnist_transform_test.data
        ])

    emnist_targets = \
        torch.cat([
            emnist_train.targets,
            emnist_test.targets
        ])

    if dist_shift:
        with open('data/mnist/all_data/rotations.pkl', "rb") as f:
            rotation_idx = pickle.load(f)
            # x = emnist_data[rotation_idx]
            # y = emnist_targets[rotation_idx]

            # x = torch.rot90(x, 2, [1, 2])
            # x = 1 - x
            # y = permute_label(y, 62)
            #
            # emnist_data[rotation_idx] = x
            # emnist_targets[rotation_idx] = y

            # x_tranformed = d_transform(emnist_data_transformed)
            # print(x_tranformed.size())

            x_trans = torch.utils.data.Subset(emnist_data_transformed, rotation_idx)
            emnist_data[rotation_idx] = x_trans[0]
            if dp:
                with open('data/mnist/clients_permute.pkl', "rb") as f1:
                    with open('data/mnist/label_projection.pkl', "rb") as f2:
                        client_permutation = torch.LongTensor(pickle.load(f1))
                        label_projection = torch.LongTensor(pickle.load(f2))

                        emnist_targets[client_permutation] = label_projection[emnist_targets[client_permutation]]
                        # emnist_data[rotation_idx] = x
                        # emnist_targets[rotation_idx] = y
            else:
                with open('data/mnist/label_projection.pkl', "rb") as f2:
                    label_projection = torch.LongTensor(pickle.load(f2))
                    emnist_targets[rotation_idx] = label_projection[emnist_targets[rotation_idx]]

            # gray_img = T.Grayscale()
            # jitter = T.ColorJitter(brightness=.5, hue=.3)
            # inverter = T.RandomInvert()
            # rotater = T.RandomRotation(degrees=(0, 180))
            # print(torch.sum(x_trans[0][0]))

            # emnist_targets[rotation_idx] = label_projection[y_i]
            # for step, imgs in enumerate(x):
            #     imgs = gray_img(imgs)
            #     imgs = jitter(imgs)
            #     imgs = inverter(imgs)
            #     imgs = rotater(imgs)
            #     imgs = jitter(imgs)
            #     imgs = inverter(imgs)
            #     imgs = rotater(imgs)
            #     emnist_data[rotation_idx[step]] = imgs
        #
    return emnist_data, emnist_targets


def get_mnist9(dist_shift=False, dp=False, leave_out=True):
    """
        gets full (both train and test) EMNIST dataset inputs and labels;
        the dataset should be first downloaded (see data/emnist/README.md)
        :return:
            emnist_data, emnist_targets
        """
    emnist_path = os.path.join("data", "mnist9", "raw_data")
    assert os.path.isdir(emnist_path), "Download MNIST dataset!!"
    d_norm = Normalize((0.1307,), (0.3081,))
    emnist_train = \
        MNIST(
            root=emnist_path,
            download=True,
            train=True
        )

    emnist_test = \
        MNIST(
            root=emnist_path,
            download=True,
            train=False
        )

    emnist_data = \
        torch.cat([
            emnist_train.data,
            emnist_test.data
        ])

    emnist_targets = \
        torch.cat([
            emnist_train.targets,
            emnist_test.targets
        ])

    if leave_out:
        idx = (emnist_targets != 1).nonzero().reshape([-1])
    else:
        idx = (emnist_targets == 1).nonzero().reshape([-1])
    print("using mnist9")
    # emnist_data = d_norm(emnist_data[idx]/255)
    emnist_data = emnist_data[idx]/255
    emnist_targets = emnist_targets[idx]

    return emnist_data, emnist_targets

def get_mnist_rep():
    mnist_path = os.path.join("data", "mnist_rep", "rep_data")
    assert os.path.isdir(mnist_path), "Download MNIST dataset!!"

    rep_all = np.load(os.path.join(mnist_path, 'x.npy'))
    label_all = np.load(os.path.join(mnist_path, 'y.npy'))

    rep_all = torch.Tensor(rep_all)

    label_all = torch.Tensor(label_all)

    return rep_all, label_all


def get_cifar10(dist_shift=False, dp=False, unseen=True):
    """
    gets full (both train and test) CIFAR10 dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        cifar10_data, cifar10_targets
    """

    cifar10_path = os.path.join("data", "cifar10", "raw_data")
    assert os.path.isdir(cifar10_path), "Download cifar10 dataset!!"
    cifar10_train = \
        CIFAR10(
            root=cifar10_path,
            train=True, download=False
        )

    cifar10_test = \
        CIFAR10(
            root=cifar10_path,
            train=False,
            download=False)

    cifar10_data = \
            torch.cat([
            torch.tensor(cifar10_train.data),
            torch.tensor(cifar10_test.data)
        ])

    cifar10_targets = \
        torch.cat([
            torch.tensor(cifar10_train.targets),
            torch.tensor(cifar10_test.targets)
        ])
    d_norm = Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
    if dist_shift:
        # cifar10_data = d_norm(cifar10_data)
        cifar10_data = cifar10_data.view(-1,3,32,32)

        with open('data/cifar10/all_data/rotations.pkl', "rb") as f:

            rotation_idx = pickle.load(f)
  
            # inverted_images = 255 - cifar10_data[rotation_idx]
            # flipped_images = torch.flip(inverted_images, [3])
            rotated_images = torch.rot90(cifar10_data[rotation_idx], 1, [2, 3])
            trans_data = rotated_images
            cifar10_data[rotation_idx] = trans_data

    if dp:
        with open('data/cifar10/rotations.pkl', "rb") as f:
            rotation_idx = pickle.load(f)
        with open('data/cifar10/label_projection.pkl', "rb") as f2:
            label_projection = torch.LongTensor(pickle.load(f2))
            cifar10_targets[rotation_idx] = label_projection[cifar10_targets[rotation_idx]]
    cifar10_data = cifar10_data / 255
    # cifar10_data = d_norm(cifar10_data)
    return cifar10_data, cifar10_targets


def get_cifar100(dist_shift=False, dp=False, unseen=True):
    """
    gets full (both train and test) CIFAR100 dataset inputs and labels;
    the dataset should be first downloaded (see data/cifar100/README.md)
    :return:
        cifar100_data, cifar100_targets
    """
    # cifar100_path = os.path.join("data", "cifar100", "raw_data")
    # assert os.path.isdir(cifar100_path), "Download cifar10 dataset!!"

    cifar100_path = os.path.join("data", "cifar100", "raw_data")
    assert os.path.isdir(cifar100_path), "Download cifar100 dataset!!"
    # d_transform = T.Compose([
    #     # Resize the images to a fixed size
    #     T.ToPILImage(),
    #     T.Grayscale(num_output_channels=3),
    #     T.RandomRotation(degrees=90),
    #     T.RandomInvert(1),
    #     T.RandomHorizontalFlip(1),
    #     T.ToTensor(),
    #
    # ])
    # d_base = T.Compose([
    #     # Resize the images to a fixed size
    #     T.ToPILImage(),
    #     T.ToTensor(),
    # ])
    cifar100_train = \
        CIFAR100(
            root=cifar100_path,
            train=True, download=False
        )

    cifar100_test = \
        CIFAR100(
            root=cifar100_path,
            train=False,
            download=False)

    cifar100_data = \
        torch.cat([
            torch.tensor(cifar100_train.data),
            torch.tensor(cifar100_test.data)
        ])

    cifar100_targets = \
        torch.cat([
            torch.tensor(cifar100_train.targets),
            torch.tensor(cifar100_test.targets)
        ])
    d_norm = Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
    a, b, c, d = cifar100_data.shape

    if dist_shift:
        cifar100_data = cifar100_data.view(a, d, b, c)


        # cifar10_data = d_norm(cifar10_data)

        with open('data/cifar100/all_data/rotations.pkl', "rb") as f:
            rotation_idx = pickle.load(f)
            cifar100_data = cifar100_data.float()
            # inverted_images = 255 - cifar100_data[rotation_idx]
            # flipped_images = torch.flip(inverted_images, [3])
            rotated_images = torch.rot90(cifar100_data[rotation_idx], 1, [2, 3])
            trans_data = rotated_images
            cifar100_data[rotation_idx] = trans_data


    if dp:
        with open('data/cifar100/rotations.pkl', "rb") as f:
            rotation_idx = pickle.load(f)
        with open('data/cifar100/label_projection.pkl', "rb") as f2:
            label_projection = torch.LongTensor(pickle.load(f2))
            cifar100_targets[rotation_idx] = label_projection[cifar100_targets[rotation_idx]]
    cifar100_data = cifar100_data / 255
    # cifar100_data = d_norm(cifar100_data)
    # cifar100_data = cifar100_data.view(-1, 3, 32, 32)
    return cifar100_data, cifar100_targets
