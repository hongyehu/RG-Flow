from math import log

import numpy as np

from .hierarchy import HierarchyBijector


def get_indices(shape, height, width, stride, dialation, offset):
    H, W = shape
    out_height = (H - dialation * (height - 1) - 1) // stride + 1
    out_width = (W - dialation * (width - 1) - 1) // stride + 1

    i0 = np.repeat(np.arange(height) * dialation, width)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(width) * dialation, height)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    return (i.transpose(1, 0) + offset) % H, (j.transpose(1, 0) + offset) % W


def mera_indices(L, kernel_size):
    index_list = []
    depth = int(log(L / kernel_size, 2) + 1)
    for i in range(depth):
        index_list.append(
            get_indices([L, L], kernel_size, kernel_size, kernel_size * 2**i,
                        2**i, 0))
        index_list.append(
            get_indices([L, L], kernel_size, kernel_size, kernel_size * 2**i,
                        2**i, kernel_size * 2**i // 2))
    indexI = [item[0] for item in index_list]
    indexJ = [item[1] for item in index_list]
    return indexI, indexJ


class MERA(HierarchyBijector):
    def __init__(self, layers, L, kernel_size, prior=None):
        indexI, indexJ = mera_indices(L, kernel_size)
        super().__init__(indexI, indexJ, layers, prior)
