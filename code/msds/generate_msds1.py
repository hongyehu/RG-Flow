#!/usr/bin/env python3

import os
from random import randint, random

import h5py
import numpy as np
from PIL import Image, ImageDraw


def generate(dir_name, num_imgs):
    # `torchvision.datasets.ImageFolder` requires a subfolder
    os.makedirs(f'./msds1/{dir_name}/0', exist_ok=True)

    labels = np.empty((num_imgs, 4, 4, 6), dtype=np.int32)
    for count in range(num_imgs):
        item_color = (
            randint(0, 191) + 32,
            randint(0, 191) + 32,
            randint(0, 191) + 32,
        )

        img = Image.new('RGB', (32, 32), (0, 0, 0))
        for i in range(4):
            for j in range(4):
                item = Image.new('RGBA', (8, 8), (255, 255, 255, 0))
                draw = ImageDraw.Draw(item)

                now_color = (
                    item_color[0] + randint(-32, 32),
                    item_color[1] + randint(-32, 32),
                    item_color[2] + randint(-32, 32),
                    255,
                )
                draw.ellipse((1, 3, 7, 5), now_color)
                now_angle = random() * 180
                item = item.rotate(now_angle,
                                   resample=Image.Resampling.BICUBIC)

                now_pos = (
                    i * 8 + randint(-1, 1),
                    j * 8 + randint(-1, 1),
                )
                img.paste(item, now_pos, mask=item)

                label = now_pos + (now_angle, ) + now_color[:3]
                labels[count, i, j, :] = label

        img.save(f'./msds1/{dir_name}/0/{count:05d}.png', compress_level=1)

    with h5py.File(f'./msds1/{dir_name}_labels.hdf5', 'w') as f:
        f.create_dataset('labels',
                         data=labels,
                         compression='gzip',
                         shuffle=True)


def main():
    generate('train', 9 * 10**4)
    generate('test', 10**4)


if __name__ == '__main__':
    main()
