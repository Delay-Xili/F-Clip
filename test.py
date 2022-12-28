#!/usr/bin/env python3
"""Train L-CNN
Usage:
    test.py [options] <yaml-config> <ckpt> <dataname> <datadir>
    test.py (-h | --help )

Arguments:
   <yaml-config>                   Path to the yaml hyper-parameter file
   <ckpt>                          Path to ckpt
   <dataname>                      Dataset name
   <datadir>                       Dataset dir

Options:
   -h --help                       Show this screen.
   -d --devices <devices>          Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>    Folder identifier [default: default-lr]
"""

import os
import time
import pprint
import random
import os.path as osp
import datetime
from skimage import io

import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from docopt import docopt
import FClip
from FClip.config import C, M
from FClip.datasets import collate
from FClip.datasets import LineDataset as WireframeDataset

from FClip.models import MultitaskHead, hg, hgl, hr
from FClip.models.stage_1 import FClip



_PLOT_nlines = 100
_PLOT = False
_NPZ = True
PLTOPTS = {"color": "#33FFFF", "s": 1.2, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.9, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def imshow(im):
    sizes = im.shape
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.xlim([-0.5, sizes[1] - 0.5])
    plt.ylim([sizes[0] - 0.5, -0.5])
    plt.imshow(im)


def c(x):
    return sm.to_rgba(x)


def build_model(cpu=False):
    if M.backbone == "stacked_hourglass":
        model = hg(
            depth=M.depth,
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(MultitaskHead._get_head_size(), [])),
        )
    elif M.backbone == "hourglass_lines":
        model = hgl(
            depth=M.depth,
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(MultitaskHead._get_head_size(), [])),
        )
    elif M.backbone == "hrnet":
        model = hr(
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_classes=sum(sum(MultitaskHead._get_head_size(), [])),
        )
    else:
        raise NotImplementedError

    model = FClip(model)

    if M.backbone == "hrnet":
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    if C.io.model_initialize_file:
        if cpu:
            checkpoint = torch.load(C.io.model_initialize_file, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(C.io.model_initialize_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        print('=> loading model from {}'.format(C.io.model_initialize_file))

    print("Finished constructing model!")
    return model


def main():
    args = docopt(__doc__)
    C.update(C.from_yaml(filename='config/base.yaml'))
    config_file = args["<yaml-config>"]
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    C.io.model_initialize_file = args["<ckpt>"]
    C.io.dataname = args["<dataname>"]
    C.io.datadir = args["<datadir>"]

    pprint.pprint(C, indent=4)
    bs = 1
    print("batch size: ", bs)
    print("data name: ", args["<dataname>"])
    print("data dir: ", args["<datadir>"])

    # WARNING: L-CNN is still not deterministic
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")

    # 1. dataset
    datadir = C.io.datadir
    kwargs = {
        "collate_fn": collate,
        "num_workers": C.io.num_workers,
        "pin_memory": True,
    }

    dataname = C.io.dataname
    val_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="valid", dataset=dataname), batch_size=bs, **kwargs
    )
    data_size = len(val_loader) * bs

    # 2. model
    model = build_model()
    model.cuda()

    outdir = args["--identifier"]
    print("outdir:", outdir)

    os.makedirs(f"{outdir}/npz/best", exist_ok=True)
    os.makedirs(f"{outdir}/viz", exist_ok=True)

    eval_time_ = 0
    eval_time = {
        'time_front': 0.0,
        'time_stack0': 0.0,
        'time_stack1': 0.0,
        'time_backbone': 0.0,
    }

    model.eval()
    with torch.no_grad():
        for batch_idx, (image, meta, target) in enumerate(val_loader):
            input_dict = {
                "image": image.cuda(),
            }
            eval_t = time.time()
            result = model(input_dict, isTest=True)
            eval_time_ += time.time() - eval_t

            H = result["heatmaps"]
            for i in range(image.shape[0]):
                index = batch_idx * bs + i

                npz_dict = {}
                for k, v in H.items():
                    if v is not None:
                        npz_dict[k] = v[i].cpu().numpy()
                if _NPZ:
                    np.savez(
                        f"{outdir}/npz/best/{index:06}.npz",
                        **npz_dict,
                    )
                if _PLOT:
                    lines, score = H["lines"][i].cpu().numpy() * 4, H["score"][i].cpu().numpy()
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)

                    iname = val_loader.dataset._get_im_name(index)
                    im = io.imread(iname)
                    imshow(im)
                    for (a, b), s in zip(lines[:_PLOT_nlines], score[:_PLOT_nlines]):
                        plt.plot([a[1], b[1]], [a[0], b[0]], color="orange", linewidth=0.5, zorder=s)
                        plt.scatter(a[1], a[0], **PLTOPTS)
                        plt.scatter(b[1], b[0], **PLTOPTS)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())

                    plt.savefig(f"{outdir}/viz/{index:06}.pdf", bbox_inches="tight", pad_inches=0.0, dpi=3000)
                    plt.close()

            extra_info = result["extra_info"]
            eval_time['time_front'] += extra_info['time_front']
            eval_time['time_stack0'] += extra_info['time_stack0']
            eval_time['time_stack1'] += extra_info['time_stack1']
            eval_time['time_backbone'] += extra_info['time_backbone']
            tprint(f"Validation [{batch_idx:5d}/{data_size:5d}]", " " * 25)

    with open(f"{outdir}/speed.csv", "a") as fout:
        print(f"Testing time: {data_size / eval_time_:.4f} im/s", file=fout)
        print('total time: {:.4f}'.format(eval_time_), file=fout)
        print(
            'avg time_backbone: {:.4f}\n front {:.4f}, \n stack0 {:.4f}, \n stack1 {:.4f}'.format(
                eval_time['time_backbone'] / data_size,
                eval_time['time_front'] / data_size,
                eval_time['time_stack0'] / data_size,
                eval_time['time_stack1'] / data_size,
            ), file=fout
        )


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


if __name__ == "__main__":

    main()
