import os
from math import log

import h5py
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms

from args import args


class DataInfo():
    def __init__(self, name, channel, size):
        """Instantiates a DataInfo.

        Args:
            name: name of dataset.
            channel: number of image channels.
            size: height and width of an image.
        """
        self.name = name
        self.channel = channel
        self.size = size


class MSDSDataset(datasets.VisionDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)

        with h5py.File(f'{root}_labels.hdf5', 'r') as f:
            self.labels = torch.from_numpy(np.asarray(f['labels']))

    def __getitem__(self, index):
        # Open path as file to avoid ResourceWarning
        with open(f'{self.root}/{index:05}.png', 'rb') as f:
            img = Image.open(f)
        label = self.labels[index]
        return img, label

    def __len__(self) -> int:
        return self.labels.shape[0]


def load_dataset():
    """Load dataset.

    Returns:
        a torch dataset and its associated information.
    """
    if args.data == 'celeba32':
        data_info = DataInfo(args.data, 3, 32)
        root = os.path.join(args.data_path, 'celeba')
        transform = transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        train_set = datasets.CelebA(root,
                                    split='train',
                                    transform=transform,
                                    download=True)
        test_set = datasets.CelebA(root,
                                   split='test',
                                   transform=transform,
                                   download=True)

    elif args.data == 'mnist32':
        data_info = DataInfo(args.data, 1, 32)
        root = os.path.join(args.data_path, 'mnist')
        transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
        ])
        train_set = datasets.MNIST(root,
                                   train=True,
                                   transform=transform,
                                   download=True)
        test_set = datasets.MNIST(root,
                                  train=False,
                                  transform=transform,
                                  download=True)

    elif args.data == 'cifar10':
        data_info = DataInfo(args.data, 3, 32)
        root = os.path.join(args.data_path, 'cifar10')
        transform = transforms.ToTensor()
        train_set = datasets.CIFAR10(root,
                                     train=True,
                                     transform=transform,
                                     download=True)
        test_set = datasets.CIFAR10(root,
                                    train=False,
                                    transform=transform,
                                    download=True)

    elif args.data == 'chair600':
        data_info = DataInfo(args.data, 3, 32)
        root = os.path.join(args.data_path, 'chair600')
        transform = transforms.Compose([
            transforms.CenterCrop(300),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        train_set = datasets.ImageFolder(root, transform=transform)
        train_set, test_set = data.random_split(
            train_set, [77730, 8636],
            generator=torch.Generator().manual_seed(0))

    elif args.data == 'msds1':
        data_info = DataInfo(args.data, 3, 32)
        transform = transforms.ToTensor()
        train_set = MSDSDataset(os.path.join(args.data_path, 'msds1/train'),
                                transform=transform)
        test_set = MSDSDataset(os.path.join(args.data_path, 'msds1/test'),
                               transform=transform)

    elif args.data == 'msds2':
        data_info = DataInfo(args.data, 3, 32)
        transform = transforms.ToTensor()
        train_set = MSDSDataset(os.path.join(args.data_path, 'msds2/train'),
                                transform=transform)
        test_set = MSDSDataset(os.path.join(args.data_path, 'msds2/test'),
                               transform=transform)

    else:
        raise ValueError(f'Unknown data: {args.data}')

    assert data_info.channel == args.nchannels
    assert data_info.size == args.L

    return train_set, test_set, data_info


def get_data_batch():
    train_set, _, _ = load_dataset()
    train_loader = data.DataLoader(train_set,
                                   batch_size=args.batch_size,
                                   shuffle=True)
    dataiter = iter(train_loader)
    sample, _ = next(dataiter)
    return sample


def logit_transform(x, dequant=True, constraint=0.9, inverse=False):
    """Transforms data from [0, 1] into unbounded space.

    Restricts data into [0.05, 0.95].
    Calculates logit(restricted x).

    Args:
        x: input tensor.
        dequant: whether to do dequantization
        constraint: data constraint before logit.
        inverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
    """
    if inverse:
        logit_x = x

        # Log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(log(constraint) - log(1 - constraint))
        ldj = (F.softplus(logit_x) + F.softplus(-logit_x) -
               F.softplus(-pre_logit_scale))
        ldj = ldj.view(ldj.shape[0], -1).sum(dim=1)

        # Inverse logit transform
        x = 1 / (1 + torch.exp(-logit_x))    # [0.05, 0.95]

        # Unrestrict data
        x *= 2    # [0.1, 1.9]
        x -= 1    # [-0.9, 0.9]
        x /= constraint    # [-1, 1]
        x += 1    # [0, 2]
        x /= 2    # [0, 1]

        return x, ldj

    else:
        if dequant:
            # Dequantization
            noise = torch.rand_like(x)
            x = (x * 255 + noise) / 256

        # Restrict data
        x *= 2    # [0, 2]
        x -= 1    # [-1, 1]
        x *= constraint    # [-0.9, 0.9]
        x += 1    # [0.1, 1.9]
        x /= 2    # [0.05, 0.95]

        # Logit transform
        logit_x = torch.log(x) - torch.log(1 - x)

        # Log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(log(constraint) - log(1 - constraint))
        ldj = (F.softplus(logit_x) + F.softplus(-logit_x) -
               F.softplus(-pre_logit_scale))
        ldj = ldj.view(ldj.shape[0], -1).sum(dim=1)

        return logit_x, ldj
