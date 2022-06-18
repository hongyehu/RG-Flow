#!/usr/bin/env python3

from random import randint, random

from PIL import Image, ImageDraw


def generate(dir_name, num_imgs):
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
                item = item.rotate(random() * 180, resample=Image.BICUBIC)

                now_pos = (
                    i * 8 + randint(-1, 1),
                    j * 8 + randint(-1, 1),
                )
                img.paste(item, now_pos, mask=item)

        img.save(f'./msds1/{dir_name}/{count:05d}.png')


def main():
    generate('train', 9 * 10**4)
    generate('test', 10**4)


if __name__ == '__main__':
    main()
