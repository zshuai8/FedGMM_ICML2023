import torch
import pickle
from torchvision.datasets import EMNIST
from sklearn.decomposition import PCA
from torchvision import models



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

emnist_train = \
        EMNIST(
            root="./raw_data",
            split="byclass",
            download=True,
            transform=None,
            train=False
        )
    emnist_test = \
        EMNIST(
            root="./raw_data",
            split="byclass",
            download=True,
            transform=T.ToTensor(),
            train=True
        )

    emnist_data = \
        torch.cat([
            emnist_train.data,
            emnist_test.data
        ])
    # base = FEMNIST_T_Dataset(emnist_data.view(-1, 28, 28), d_base)
    print("done normalizing")
    # base = DataLoader(base, batch_size=emnist_data.shape[0],num_workers=20)

    # for batch in base:
    #     base_data = batch
    # base_data = base_data.to(torch.uint8)
    # emnist_data = base_data
    emnist_targets = \
        torch.cat([
            emnist_train.targets,
            emnist_test.targets
        ])
    with open('./all_data/rotations.pkl', "rb") as f:
        rotation_idx = pickle.load(f)
        x_trans = emnist_data[rotation_idx]
        x_trans = random_rotation(x_trans)
        x_trans = random_invert(x_trans)
        x_trans = random_horizontal_flip(x_trans)
        emnist_data[rotation_idx] = x_trans
        emnist_data = emnist_data.view(-1,1,28,28)
    with open('data/emnist_r/all_data/label_projection.pkl', "rb") as f2:
        label_projection = torch.LongTensor(pickle.load(f2))
        emnist_targets[rotation_idx] = label_projection[emnist_targets[rotation_idx]]

    model = models.resnet18(pretrained=True)
    del model.fc
    x = model.conv1(emnist_data.float())
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    # Extract the feature maps produced by the encoder
    encoder_output = x
    """"
    " Normalization
    """
    encoder_output = encoder_output.view(encoder_output.size(0), -1)
    pca_transformer = PCA(n_components=64)
    # Fit the PCA transformer to your data
    X_pca = pca_transformer.fit_transform(encoder_output.detach().numpy())

    # Convert the resulting principal components to a PyTorch tensor
    projected = torch.from_numpy(X_pca).float().cuda()