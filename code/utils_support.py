import re
from collections import namedtuple
from distutils.version import LooseVersion
from math import isinf, isnan, sqrt

import matplotlib
import numpy as np
import skimage
import skimage.io
import skimage.transform
from matplotlib import pyplot as plt

import utils

RunRow = namedtuple('RunRow',
                    ['epoch', 'batch', 'bpp', 'loss', 'loss_std', 'time'])

Run = namedtuple(
    'Run',
    ['steps', 'bpps', 'losses', 'loss_stds', 'nparams', 'time_per_step'])


def parse_run_row(s):
    match = re.compile(
        r'epoch (?P<epoch>.*?) batch (?P<batch>.*?) (bpp (?P<bpp>.*?) )?loss (?P<loss>.*?) \+- (?P<loss_std>.*?) time (?P<time>.*)'
    ).match(s)
    if not match:
        return None

    loss = float(match.group('loss'))
    if isinf(loss) or isnan(loss):
        return None

    return RunRow(
        epoch=int(match.group('epoch')),
        batch=int(match.group('batch')),
        bpp=float(match.group('bpp')),
        loss=loss,
        loss_std=float(match.group('loss_std')),
        time=float(match.group('time')),
    )


def read_log(filename):
    step = -1
    steps = []
    bpps = []
    losses = []
    loss_stds = []
    nparams = 0
    last_time = None
    time_sum = 0
    step_sum = 0
    with open(filename, 'r') as f:
        for line in f:
            match = re.compile(r'.*parameters: (.*)').match(line)
            if match:
                nparams = int(match.group(1))
                continue

            run_row = parse_run_row(line)
            if run_row:
                step += 1
                steps.append(step)
                bpps.append(run_row.bpp)
                losses.append(run_row.loss)
                loss_stds.append(run_row.loss_std)

                time = run_row.time
                if last_time is None:
                    last_time = time
                elif time < last_time or time > last_time + 60:
                    last_time = None
                else:
                    time_sum += time - last_time
                    step_sum += 1
                    last_time = time

    steps = np.array(steps, dtype=int)
    bpps = np.array(bpps)
    losses = np.array(losses)
    loss_stds = np.array(loss_stds)
    time_per_step = time_sum / step_sum

    return Run(steps, bpps, losses, loss_stds, nparams, time_per_step)


def ema(steps, losses, alpha):
    out = np.empty_like(losses)
    out[0] = losses[0]
    for i in range(1, losses.size):
        a = alpha**(steps[i] - steps[i - 1])
        out[i] = a * out[i - 1] + (1 - a) * losses[i]
    return out


def get_loss(filename, alpha=0):
    run = read_log(filename)
    steps = run.steps
    losses = run.losses
    if alpha > 0:
        losses = ema(steps, losses, alpha)
    loss = losses.min()
    return run, loss


# tight_layout() supports suptitle in matplotlib 3.3
def my_tight_layout(fig):
    if LooseVersion(matplotlib.__version__) >= LooseVersion('3.3'):
        fig.tight_layout()
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.97))


# samples: (B, H, W, C), values in (0, 1), B = array_w * array_h
def plot_samples_np(samples, array_h=None, array_w=None, fig_scale=1):
    if array_h is None and array_w is None:
        # array_h * array_w may be less than samples.shape[0]
        array_h = array_w = int(sqrt(samples.shape[0]))
    else:
        assert array_h * array_w == samples.shape[0]

    fig, axes = plt.subplots(array_h,
                             array_w,
                             figsize=(array_w * fig_scale,
                                      array_h * fig_scale),
                             sharex=True,
                             sharey=True,
                             squeeze=False)
    for i in range(array_h):
        for j in range(array_w):
            img = samples[j * array_h + i]
            if img.shape[-1] == 1:
                img = img.repeat(3, axis=-1)
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis('off')

    return fig, axes


def plot_one_sample_np(filename, sample, L):
    if len(sample.shape) == 4:
        sample = sample[0]
    sample = sample.permute(1, 2, 0).detach().cpu().numpy()
    sample = 1 / (1 + np.exp(-sample))
    sample = skimage.transform.resize(
        sample,
        (L, L),
        order=0,
        anti_aliasing=False,
    )
    sample = skimage.img_as_ubyte(sample)
    utils.ensure_dir(filename)
    skimage.io.imsave(filename, sample)
