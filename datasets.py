import os
import urllib.request
import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.io import loadmat

num_workers = 4
lamb = 0.05

class MNIST():
    def __init__(self, batch_size, binarize=False, logit_transform=False):
        """ [-1, 1, 28, 28]
        """
        self.binarize = binarize
        self.logit_transform = logit_transform
        directory='./datasets/MNIST'
        if not os.path.exists(directory):
            os.makedirs(directory)

        kwargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.train_loader = DataLoader(
            datasets.MNIST('./datasets/MNIST', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)

        self.test_loader = DataLoader(
            datasets.MNIST('./datasets/MNIST', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False, **kwargs)

        self.dim = [1,28,28]

        if self.binarize:
            pass
        else:
            train = torch.stack([data for data, _ in
                                 list(self.train_loader.dataset)], 0).cuda()
            train = train.view(train.shape[0], -1)
            if self.logit_transform:
                train = train * 255.0
                train = (train + torch.rand_like(train)) / 256.0
                train = lamb + (1 - 2.0 * lamb) * train
                train = torch.log(train) - torch.log(1.0 - train)

            self.mean = train.mean(0)
            self.logvar = torch.log(torch.mean((train - self.mean)**2)).unsqueeze(0)

    def preprocess(self, x):
        if self.binarize:
            x = x.view([-1, np.prod(self.dim)])
            return (torch.rand_like(x).cuda() < x).to(torch.float)
        elif self.logit_transform:
            # apply uniform noise and renormalize
            x = x.view([-1, np.prod(self.dim)]) * 255.0
            x = (x + torch.rand_like(x)) / 256.0
            x = lamb + (1 - 2.0 * lamb) * x
            x = torch.log(x) - torch.log(1.0 - x)
            return x - self.mean
        else:
            return x.view([-1, np.prod(self.dim)]) - self.mean

    def unpreprocess(self, x):
        if self.binarize:
            return x.view([-1] + self.dim)
        elif self.logit_transform:
            x = x + self.mean
            x = torch.sigmoid(x)
            x = (x - lamb) / (1.0 - 2.0 * lamb)
            return x.view([-1] + self.dim)
        else:
            return (x + self.mean).view([-1] + self.dim)


class FashionMNIST():
    def __init__(self, batch_size, binarize=False, logit_transform=False):
        """ [-1, 1, 28, 28]
        """
        if binarize:
            raise NotImplementedError

        self.logit_transform = logit_transform

        directory='./datasets/FashionMNIST'
        if not os.path.exists(directory):
            os.makedirs(directory)

        kwargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.train_loader = DataLoader(
            datasets.FashionMNIST(directory, train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(
            datasets.FashionMNIST(directory, train=False, download=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False, **kwargs)

        self.dim = [1,28,28]

        train = torch.stack([data for data, _ in
                                list(self.train_loader.dataset)], 0).cuda()
        train = train.view(train.shape[0], -1)
        if self.logit_transform:
            train = train * 255.0
            train = (train + torch.rand_like(train)) / 256.0
            train = lamb + (1 - 2.0 * lamb) * train
            train = torch.log(train) - torch.log(1.0 - train)

        self.mean = train.mean(0)
        self.logvar = torch.log(torch.mean((train - self.mean)**2)).unsqueeze(0)

    def preprocess(self, x):
        if self.logit_transform:
            # apply uniform noise and renormalize
            x = x.view([-1, np.prod(self.dim)]) * 255.0
            x = (x + torch.rand_like(x)) / 256.0
            x = lamb + (1 - 2.0 * lamb) * x
            x = torch.log(x) - torch.log(1.0 - x)
            return x - self.mean
        else:
            return x.view([-1, np.prod(self.dim)]) - self.mean

    def unpreprocess(self, x):
        if self.logit_transform:
            x = x + self.mean
            x = torch.sigmoid(x)
            x = (x - lamb) / (1.0 - 2.0 * lamb)
            return x.view([-1] + self.dim)
        else:
            return (x + self.mean).view([-1] + self.dim)


class SVHN():
    def __init__(self, batch_size, binarize=False, logit_transform=False):
        """ [-1, 3, 32, 32]
        """
        if binarize:
            raise NotImplementedError

        self.logit_transform = logit_transform

        directory='./datasets/SVHN'
        if not os.path.exists(directory):
            os.makedirs(directory)

        kwargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.train_loader = DataLoader(
            datasets.SVHN(root=directory,split='train', download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(
            datasets.SVHN(root=directory, split='test', download=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False, **kwargs)

        self.dim = [3, 32, 32]

        train = torch.stack([data for data, _ in
                                list(self.train_loader.dataset)], 0).cuda()
        train = train.view(train.shape[0], -1)
        if self.logit_transform:
            train = train * 255.0
            train = (train + torch.rand_like(train)) / 256.0
            train = lamb + (1 - 2.0 * lamb) * train
            train = torch.log(train) - torch.log(1.0 - train)

        self.mean = train.mean(0)
        self.logvar = torch.log(torch.mean((train - self.mean)**2)).unsqueeze(0)

    def preprocess(self, x):
        if self.logit_transform:
            # apply uniform noise and renormalize
            x = x.view([-1, np.prod(self.dim)]) * 255.0
            x = (x + torch.rand_like(x)) / 256.0
            x = lamb + (1 - 2.0 * lamb) * x
            x = torch.log(x) - torch.log(1.0 - x)
            return x - self.mean
        else:
            return x.view([-1, np.prod(self.dim)]) - self.mean

    def unpreprocess(self, x):
        if self.logit_transform:
            x = x + self.mean
            x = torch.sigmoid(x)
            x = (x - lamb) / (1.0 - 2.0 * lamb)
            return x.view([-1] + self.dim)
        else:
            return (x + self.mean).view([-1] + self.dim)


class CIFAR10():
    def __init__(self, batch_size, binarize=False, logit_transform=False):
        """ [-1, 3, 32, 32]
        """
        if binarize:
            raise NotImplementedError

        self.logit_transform = logit_transform

        directory='./datasets/CIFAR10'
        if not os.path.exists(directory):
            os.makedirs(directory)

        kwargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}

        self.train_loader = DataLoader(
            datasets.CIFAR10(root=directory, train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(
            datasets.CIFAR10(root=directory, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False, **kwargs)

        self.dim = [3, 32, 32]

        train = torch.stack([data for data, _ in
                                list(self.train_loader.dataset)], 0).cuda()
        train = train.view(train.shape[0], -1)
        if self.logit_transform:
            train = train * 255.0
            train = (train + torch.rand_like(train)) / 256.0
            train = lamb + (1 - 2.0 * lamb) * train
            train = torch.log(train) - torch.log(1.0 - train)

        self.mean = train.mean(0)
        self.logvar = torch.log(torch.mean((train - self.mean)**2)).unsqueeze(0)

    def preprocess(self, x):
        if self.logit_transform:
            # apply uniform noise and renormalize
            x = x.view([-1, np.prod(self.dim)]) * 255.0
            x = (x + torch.rand_like(x)) / 256.0
            x = lamb + (1 - 2.0 * lamb) * x
            x = torch.log(x) - torch.log(1.0 - x)
            return x - self.mean
        else:
            return x.view([-1, np.prod(self.dim)]) - self.mean

    def unpreprocess(self, x):
        if self.logit_transform:
            x = x + self.mean
            x = torch.sigmoid(x)
            x = (x - lamb) / (1.0 - 2.0 * lamb)
            return x.view([-1] + self.dim)
        else:
            return (x + self.mean).view([-1] + self.dim)


class OMNIGLOT(Dataset):
    def __init__(self, batch_size, binarize=False, logit_transform=False):
        """ [ -1, 1, 28, 28]
        """
        if binarize:
            raise NotImplementedError

        self.logit_transform = logit_transform

        directory='./datasets/OMNIGLOT'
        if not os.path.exists(directory):
            os.makedirs(directory)
            if not os.path.exists(os.path.join(directory, 'chardata.mat')):
                print ('Downloading Omniglot images_background.zip...')
                urllib.request.urlretrieve('https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat',
                                   os.path.join(directory, 'chardata.mat'))


        data = loadmat(os.path.join(directory, 'chardata.mat'))
        # between 0~1.
        train = data['data'].swapaxes(0,1).reshape((-1, 1, 28, 28)).astype('float32')
        test = data['testdata'].swapaxes(0,1).reshape((-1, 1, 28, 28)).astype('float32')
        train_labels = np.zeros(train.shape[0])
        test_labels = np.zeros(test.shape[0])

        train_dataset = TensorDataset(torch.from_numpy(train), torch.from_numpy(train_labels))
        test_dataset = TensorDataset(torch.from_numpy(test), torch.from_numpy(test_labels))

        kwargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        self.dim = [1, 28, 28]

        train = torch.stack([data for data, _ in
                                list(self.train_loader.dataset)], 0).cuda()
        train = train.view(train.shape[0], -1)
        if self.logit_transform:
            train = train * 255.0
            train = (train + torch.rand_like(train)) / 256.0
            train = lamb + (1 - 2.0 * lamb) * train
            train = torch.log(train) - torch.log(1.0 - train)

        self.mean = train.mean(0)
        self.logvar = torch.log(torch.mean((train - self.mean)**2)).unsqueeze(0)

    def preprocess(self, x):
        if self.logit_transform:
            # apply uniform noise and renormalize
            x = x.view([-1, np.prod(self.dim)]) * 255.0
            x = (x + torch.rand_like(x)) / 256.0
            x = lamb + (1 - 2.0 * lamb) * x
            x = torch.log(x) - torch.log(1.0 - x)
            return x - self.mean
        else:
            return x.view([-1, np.prod(self.dim)]) - self.mean

    def unpreprocess(self, x):
        if self.logit_transform:
            x = x + self.mean
            x = torch.sigmoid(x)
            x = (x - lamb) / (1.0 - 2.0 * lamb)
            return x.view([-1] + self.dim)
        else:
            return (x + self.mean).view([-1] + self.dim)
