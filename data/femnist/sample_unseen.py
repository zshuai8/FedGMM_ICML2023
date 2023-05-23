import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
def get_loader(type_, path, batch_size, train,dist_shift=True, dp=True, inputs=None, targets=None, emb_size=64):
    """
    constructs a torch.utils.DataLoader object from the given path

    :param type_: type of the dataset; possible are `tabular`, `images` and `text`
    :param path: path to the data file
    :param batch_size:
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :param targets: tensor storing the labels; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :return: torch.utils.DataLoader
    """
    dataset = SubFEMNIST(path)
    # drop last batch, because of BatchNorm layer used in mobilenet_v2
    # drop_last = ((type_ == "cifar100") or (type_ == "cifar10")) and (len(dataset) > batch_size) and train
    drop_last = (len(dataset) > batch_size) and train

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last)
root_path = "all_data/train"
train_iterators, val_iterators, test_iterators = [], [], []

for task_id, task_dir in enumerate(tqdm(os.listdir(root_path))):

    task_data_path = os.path.join(root_path, task_dir)
    train_iterator = \
        get_loader(
            type_='femnist',
            path=os.path.join(task_data_path, f"train{'.pt'}"),
            batch_size=128,
            inputs=None,
            targets=None,
            train=True,
            emb_size=48
        )

    val_iterator = \
        get_loader(
            type_='femnist',
            path=os.path.join(task_data_path, f"train{'.pt'}"),
            batch_size=128,
            inputs=None,
            targets=None,
            train=True,
            emb_size=48
        )

    test_set = "test"

    test_iterator = \
        get_loader(
            type_='femnist',
            path=os.path.join(task_data_path, f"{test_set}{'.pt'}"),
            batch_size=128,
            inputs=None,
            targets=None,
            train=False
        )


    train_iterators.append(train_iterator)
    val_iterators.append(val_iterator)
    test_iterators.append(test_iterator)
data_list = []
targets_list = []
for i, data_loader in enumerate(test_iterators):
    for data, label, _ in data_loader:
        data_list.append(data*0.9)
        targets_list.append(label)

from sklearn.model_selection import train_test_split

data_list = torch.cat(data_list, dim=0)
targets_list = torch.cat(targets_list, dim=0)
sampled_indices = torch.randperm(len(data_list))[:2000]
sampled_data = data_list[sampled_indices]
sampled_targets = targets_list[sampled_indices]
train_data, test_data, train_targets, test_targets =\
        train_test_split(
            sampled_data,
            sampled_targets,
            train_size=0.8,
            random_state=1234
        )
save_path = os.path.join("all_data_unseen_3", "train", f"task_{0}")
os.makedirs(save_path, exist_ok=True)

save_task(
    dir_path=save_path,
    train_data=train_data,
    train_targets=train_targets,
    test_data=test_data,
    test_targets=test_targets,
    val_data=None,
    val_targets=None
)

