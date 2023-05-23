import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torch.autograd import Variable
import pickle
from sklearn.decomposition import PCA





class LinearLayer(nn.Module):
    def __init__(self, input_dimension, num_classes, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        print(input_dimension, num_classes)
        self.fc = nn.Linear(input_dimension, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)


class FemnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """

    def __init__(self, num_classes, pretrain=False):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
        self.output = nn.Linear(2048, num_classes)

        if pretrain:
            save_path = os.path.join('pretrain', 'chkpts_0.pt')
            state_dict = torch.load(save_path)
            # state_dict.pop('output.weight')
            # state_dict.pop('output.bias')

            self.load_state_dict(state_dict=state_dict)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(-1,1,28,28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


class NextCharacterLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn =\
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        encoded = self.encoder(input_)
        output, _ = self.rnn(encoded)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        return output


def get_vgg11(n_classes):
    """
    creates VGG11 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.vgg11(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)

    return model


def get_squeezenet(n_classes):
    """
    creates SqueezeNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = n_classes

    return model


def get_mobilenet(n_classes):
    """
    creates MobileNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    return model


def get_resnet18(n_classes):
    """
    creates Resnet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


def get_resnet34(n_classes):
    """
    creates Resnet34 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


class FC_Classifier(nn.Module):
    def __init__(self, embedding_size, num_class):
        super(FC_Classifier, self).__init__()
        self.fc3 = nn.Linear(embedding_size, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, num_class)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        return self.fc5(h4)


class CNN_Encoder(nn.Module):
    def __init__(self, embed_size, input_size):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16

        # convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0],
                      out_channels=self.channel_mult * 1,
                      kernel_size=4,
                      stride=1,
                      padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 1, self.channel_mult * 2, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 2, self.channel_mult * 4, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 4, self.channel_mult * 8, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 8, self.channel_mult * 16, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flat_fts = self.get_flat_fts(self.conv)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.LeakyReLU(0.2),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.conv(x.view(-1, *self.input_size))
        x = x.view(-1, self.flat_fts)
        return self.linear(x)


class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size):
        super(CNN_Decoder, self).__init__()
        self.input_channel = input_size[0]
        self.input_height = input_size[1]
        self.input_width = input_size[2]
        self.input_dim = embedding_size
        self.channel_mult = 16
        self.output_channels = self.input_channel
        self.fc_output_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.BatchNorm1d(self.fc_output_dim),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult * 4,
                               4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.ReLU(True),
            # state size. self.channel_mult*32 x 4 x 4
            nn.ConvTranspose2d(self.channel_mult * 4, self.channel_mult * 2,
                               3, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.ReLU(True),
            # state size. self.channel_mult*16 x 7 x 7
            nn.ConvTranspose2d(self.channel_mult * 2, self.channel_mult * 1,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult * 1),
            nn.ReLU(True),
            # state size. self.channel_mult*8 x 14 x 14
            nn.ConvTranspose2d(self.channel_mult * 1, self.output_channels, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. self.output_channels x 28 x 28
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        x = self.deconv(x)
        return x.view(-1, self.input_channel * self.input_width * self.input_height)


class resnet_pca(nn.Module):
    def __init__(self, embedding_size, name, input_size=(1, 28, 28), dp=False):
        super().__init__()
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.encoder = CNN_Encoder(self.embedding_size, input_size=input_size)
        self.decoder = CNN_Decoder(self.embedding_size, input_size=input_size)
        self.model = models.resnet18(pretrained=True)
        del self.model.fc
        self.seq = nn.Sequential(
            self.model.conv1,
            self.model.relu,
            self.model.layer1,
            self.model.bn1,
            self.model.maxpool,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        )
        if not dp:
            if name == 'emnist':
                with open("data/emnist/all_data/PCA.pkl", 'rb') as f:
                    self.PCA_V = pickle.load(f)
            elif name == 'femnist':
                with open("data/femnist/all_data/PCA.pkl", 'rb') as f:
                    self.PCA_V = pickle.load(f)
            elif name == 'cifar10':
                with open("data/cifar10/all_data/PCA.pkl", 'rb') as f:
                    self.PCA_V = pickle.load(f)
            elif name == 'cifar100':
                with open("data/cifar100/all_data/PCA.pkl", 'rb') as f:
                    self.PCA_V = pickle.load(f)
            elif name == 'mnist9':
                with open("data/mnist9/all_data/PCA.pkl", 'rb') as f:
                    self.PCA_V = pickle.load(f)
            else:
                raise
        else:
            if name == 'emnist':
                with open("data/emnist/all_data/PCA_no_t.pkl", 'rb') as f:
                    self.PCA_V = pickle.load(f)
            elif name == 'femnist':
                with open("data/femnist/all_data/PCA_no_t.pkl", 'rb') as f:
                    self.PCA_V = pickle.load(f)
            elif name == 'cifar10':
                with open("data/cifar10/all_data/PCA_no_t.pkl", 'rb') as f:
                    self.PCA_V = pickle.load(f)
            elif name == 'cifar100':
                with open("data/cifar100/all_data/PCA_no_t.pkl", 'rb') as f:
                    self.PCA_V = pickle.load(f)
            else:
                raise
        self.PCA_V = self.PCA_V[:,:embedding_size]
    def encode(self, x):

        singular = False
        if len(x.shape) == 3 or x.shape[1] == 1:
            x = x.view(-1, 1, 28, 28)
            x = x.repeat(1, 3, 1, 1)
        else:
            x = x.view(-1, 3, 32, 32)
        if x.shape[0] == 1:
            x = x.repeat(2, 1, 1, 1)
            singular = True

        x = self.seq(x)
        # Extract the feature maps produced by the encoder
        encoder_output = x.squeeze()
        if singular:
            returned = encoder_output[0,:] @ self.PCA_V.to(x.device)
            return returned.view(1,-1)
        returned = encoder_output @ self.PCA_V.to(x.device)
        a, b = returned.shape
        return returned.view(a, b)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # z = self.encode(x)
        a,_,_,_ = x.shape
        # z = z.view(a,b,1,1)
        # return self.decode(z)
        return x.view(a,-1)


class ACnetwork(nn.Module):
    def __init__(self, embedding_size, name, input_size=(1, 28, 28)):
        super().__init__()
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.encoder = CNN_Encoder(self.embedding_size, input_size=input_size)
        self.decoder = CNN_Decoder(self.embedding_size, input_size=input_size)


    def encode(self, x):

        return self.encoder.encode(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class IDnetwork(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.placeholder = nn.Linear(1, 1)

    def encode(self, x):
        return x

    def decode(self, z):
        return z

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class cCNN_Encoder(nn.Module):
    def __init__(self, embed_size, input_size):
        super(cCNN_Encoder, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16

        # convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0],
                      out_channels=self.channel_mult * 1,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 1, self.channel_mult * 2, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 2, self.channel_mult * 4, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 4, self.channel_mult * 8, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(self.channel_mult * 8, self.channel_mult * 16, 3, 2, 1),
            # nn.BatchNorm2d(self.channel_mult * 16),
            # nn.LeakyReLU(0.2, inplace=True)
        )

        self.flat_fts = self.get_flat_fts(self.conv)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.LeakyReLU(0.2),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.conv(x.view(-1, *self.input_size))
        x = x.view(-1, self.flat_fts)
        return self.linear(x)


class cCNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size):
        super(cCNN_Decoder, self).__init__()
        self.input_channel = input_size[0]
        self.input_height = input_size[1]
        self.input_width = input_size[2]
        self.input_dim = embedding_size
        self.channel_mult = 16
        self.output_channels = self.input_channel
        self.fc_output_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.BatchNorm1d(self.fc_output_dim),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult * 4,
                               4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.ReLU(True),
            # state size. self.channel_mult*32 x 4 x 4
            nn.ConvTranspose2d(self.channel_mult * 4, self.channel_mult * 2,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.ReLU(True),
            # state size. self.channel_mult*16 x 8 x 8
            nn.ConvTranspose2d(self.channel_mult * 2, self.channel_mult * 1,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult * 1),
            nn.ReLU(True),
            # state size. self.channel_mult*8 x 16 x 16
            nn.ConvTranspose2d(self.channel_mult * 1, self.output_channels, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. self.output_channels x 32 x 32
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        x = self.deconv(x)
        return x.view(-1, self.input_channel * self.input_width * self.input_height)


class cACnetwork(nn.Module):
    def __init__(self, embedding_size, input_size=(3, 32, 32)):
        super().__init__()
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.encoder = cCNN_Encoder(self.embedding_size, input_size=input_size)
        self.decoder = cCNN_Decoder(self.embedding_size, input_size=input_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)