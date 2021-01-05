#!/usr/bin/env python3
"""Evaluate sAP5, sAP10, sAP15 for LCNN
Usage:
    eval-sAP.py [options] <path>...
    eval-sAP.py (-h | --help )

Arguments:
    <path>                           One or more directories from train.py

Options:
   -h --help                         Show this screen.
   -c --config <config>              expe config   [default: config]
   -m --mode <mode>                  Good set name [default: shanghaiTech]
"""

import os
import glob
import numpy as np
from docopt import docopt

import FClip.utils
import FClip.metric
from FClip.config import C
from FClip.line_parsing import line_parsing_from_npz
from tqdm import tqdm


def line_center_score(path, GT, threshold=5):
    preds = sorted(glob.glob(path))
    gts = sorted(glob.glob(GT))

    n_gt = 0
    n_pt = 0
    tps, fps, scores = [], [], []

    for pred_name, gt_name in tqdm(zip(preds, gts)):
        with np.load(gt_name) as fgt:
            gt_line = fgt["lpos"][:, :, :2]

        line, score = line_parsing_from_npz(
            pred_name,
            delta=C.model.delta, nlines=C.model.nlines,
            s_nms=C.model.s_nms,
        )
        line = line * (128 / C.model.resolution)

        n_gt += len(gt_line)
        n_pt += len(line)
        tp, fp, hit = FClip.metric.msTPFP_hit(line, gt_line, threshold)
        tps.append(tp)
        fps.append(fp)
        scores.append(score)

    tps = np.concatenate(tps)
    fps = np.concatenate(fps)

    scores = np.concatenate(scores)
    index = np.argsort(-scores)
    lcnn_tp = np.cumsum(tps[index]) / n_gt
    lcnn_fp = np.cumsum(fps[index]) / n_gt

    return FClip.metric.ap(lcnn_tp, lcnn_fp)


def batch_sAP_s1(paths, GT, dataname):
    gt = GT

    def work(path):
        print(f"Working on {path}")
        return [100 * line_center_score(f"{path}/*.npz", gt, t) for t in [5, 10, 15]]

    dirs = sorted(sum([glob.glob(p) for p in paths], []))
    results = FClip.utils.parmap(work, dirs, 8)
    with open(f"{args['<path>'][0][:-14]}/sAP_{dataname}.csv", "a") as fout:
        print(f"Resolution: {C.model.resolution}", file=fout)
        for d, msAP in zip(dirs, results):
            print(f"{d[-13:]}: {msAP[0]:2.1f} {msAP[1]:2.1f} {msAP[2]:2.1f}", file=fout)


def sAP_s1(path, GT):
    sAP = [5, 10, 15]
    print(f"Working on {path}")
    print("sAP: ", sAP)
    return [100 * line_center_score(f"{path}/*.npz", GT, t) for t in sAP]


if __name__ == "__main__":

    args = docopt(__doc__)

    config_file = args["--config"]
    if config_file == "config":
        config_file = args["<path>"][-1]
        config_file = os.path.dirname(config_file)
        config_file = os.path.dirname(config_file)
        config_file = os.path.join(config_file, "config.yaml")
    print(f"load config file {config_file}")
    C.update(C.from_yaml(filename=config_file))

    GT_york = f"/home/dxl/Data/york/valid/*_label.npz"
    GT_huang = f"/home/dxl/Data/wireframe/valid/*_label.npz"

    print(args["--mode"])
    if args["--mode"] == "shanghaiTech":
        GT = GT_huang
    elif args["--mode"] == "york":
        GT = GT_york
    else:
        print(args["--mode"])
        raise ValueError("no such dataset")

    idx = int(len(args["<path>"]) / 2)

    if idx <= 32:
        batch_sAP_s1(args["<path>"], GT, args["--mode"])
    else:
        batch_sAP_s1(args["<path>"][-idx:], GT, args["--mode"])



