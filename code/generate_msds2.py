#!/usr/bin/env python3

from random import randint, random

from PIL import Image, ImageDraw


def generate(dir_name, num_imgs):
    for count in range(num_imgs):
        item_angle = random() * 180

        img = Image.new('RGB', (32, 32), (0, 0, 0))
        for i in range(4):
            for j in range(4):
                item = Image.new('RGBA', (8, 8), (255, 255, 255, 0))
                draw = ImageDraw.Draw(item)

                now_color = (
                    randint(0, 255),
                    randint(0, 255),
                    randint(0, 255),
                    255,
                )
                draw.ellipse((1, 3, 7, 5), now_color)
                now_angle = item_angle + random() * 45
                item = item.rotate(now_angle, resample=Image.BICUBIC)

                now_pos = (
                    i * 8 + randint(-1, 1),
                    j * 8 + randint(-1, 1),
                )
                img.paste(item, now_pos, mask=item)

        img.save(f'./msds2/{dir_name}/{count:05d}.png')


def main():
    generate('train', 9 * 10**4)
    generate('test', 10**4)


if __name__ == '__main__':
    main()
