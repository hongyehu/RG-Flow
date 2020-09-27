import os
from math import log

import torch
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


def load_dataset():
    """Load dataset.

    Returns:
        a torch dataset and its associated information.
    """

    if args.data == 'celeba32':
        data_info = DataInfo(args.data, 3, 32)
        transform = transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        train_set = datasets.ImageFolder(os.path.join(args.data_path,
                                                      'celeba'),
                                         transform=transform)
        [train_split, val_split] = data.random_split(train_set,
                                                     [180000, 22599])

    elif args.data == 'celeba64':
        data_info = DataInfo(args.data, 3, 64)
        transform = transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        train_set = datasets.ImageFolder(os.path.join(args.data_path,
                                                      'celeba'),
                                         transform=transform)
        [train_split, val_split] = data.random_split(train_set,
                                                     [180000, 22599])

    elif args.data == 'mnist32':
        data_info = DataInfo(args.data, 1, 32)
        transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
        ])
        train_set = datasets.MNIST(os.path.join(args.data_path, 'mnist'),
                                   train=True,
                                   download=True,
                                   transform=transform)
        [train_split, val_split] = data.random_split(train_set, [54000, 6000])

    elif args.data == 'cifar10':
        data_info = DataInfo(args.data, 3, 32)
        transform = transforms.ToTensor()
        train_set = datasets.CIFAR10(os.path.join(args.data_path, 'cifar10'),
                                     train=True,
                                     download=True,
                                     transform=transform)
        [train_split, val_split] = data.random_split(train_set, [45000, 5000])

    elif args.data == 'chair600':
        data_info = DataInfo(args.data, 3, 32)
        transform = transforms.Compose([
            transforms.CenterCrop(300),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        train_set = datasets.ImageFolder(os.path.join(args.data_path,
                                                      'chair600'),
                                         transform=transform)
        [train_split, val_split] = data.random_split(train_set, [78000, 8366])

    else:
        raise ValueError('Unknown data: {}'.format(args.data))

    assert data_info.channel == args.nchannels
    assert data_info.size == args.L

    return train_split, val_split, data_info


def get_data_batch():
    train_split, _, _ = load_dataset()
    train_loader = data.DataLoader(train_split,
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
