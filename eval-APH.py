#!/usr/bin/env python3
"""Evaluate APH for LCNN
Usage:
    eval-APH.py <src> <dst>
    eval-APH.py (-h | --help )
Examples:
    ./eval-APH.py post/RUN-ITERATION/0_010 post/RUN-ITERATION/0_010-APH
Arguments:
    <src>                Source directory that stores preprocessed npz
    <dst>                Temporary output directory
Options:
   -h --help             Show this screen.
"""

import os
import glob
import os.path as osp
import subprocess

import numpy as np
import scipy.io as sio
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
from docopt import docopt
from tqdm import tqdm
from FClip.line_parsing import line_parsing_from_npz

mpl.rcParams.update({"font.size": 18})
plt.rcParams["font.family"] = "Times New Roman"
del mpl.font_manager.weight_dict["roman"]
mpl.font_manager._rebuild()

output_size = 128
resolution = None
dataname = None
image_path = None
line_gt_path = None


# python eval-APH.py logs/path/to/npz/pth logs/output/pth/APH


def read_fclip(pred_pth, output):
    dirs = sorted(glob.glob(osp.join(pred_pth, "*.npz")))
    if osp.exists(output):
        return
    else:
        os.makedirs(output, exist_ok=True)
        print(f"line score npz path: {output}")

    for i, path in enumerate(tqdm(dirs)):
        line, score = line_parsing_from_npz(
            path,
            delta=0.8, nlines=2500,
            s_nms=0., resolution=resolution
        )
        line = line * (128 / resolution)

        np.savez(
            f"{output}/{path[-10:]}",
            lines=line,
            score=score,
        )


def read_hawp(result_path, output_dir):

    import json

    images = sorted(os.listdir(image_path))

    with open(result_path,'r') as _res:
        result_list = json.load(_res)

    os.makedirs(f"{output_dir}/npz_APH", exist_ok=True)
    print(f"line score npz path: {output_dir}/npz_APH")

    print(len(result_list))

    for res in tqdm(result_list):
        # print(res)
        filename = res['filename']
        lines_pred = np.array(res['lines_pred'], dtype=np.float32)
        scores = np.array(res['lines_score'], dtype=np.float32)
        sort_idx = np.argsort(-scores)

        lines_pred = lines_pred[sort_idx]
        scores = scores[sort_idx]
        # import pdb; pdb.set_trace()
        lines_pred[:, 0] *= 128 / float(res['width'])
        lines_pred[:, 1] *= 128 / float(res['height'])
        lines_pred[:, 2] *= 128 / float(res['width'])
        lines_pred[:, 3] *= 128 / float(res['height'])

        lines_pred = lines_pred.reshape(-1, 2, 2)
        lines_pred = lines_pred[:, :, ::-1]
        # print('======')
        # print(np.amax(lines_pred[:, :, 0]), np.amin(lines_pred[:, :, 0]))
        # print(np.amax(lines_pred[:, :, 1]), np.amin(lines_pred[:, :, 1]))

        if dataname == 'wireframe':
            img_idx = images.index(filename[:-4] + '.jpg')
        else:
            img_idx = images.index(filename[:-4] + '.png')

        np.savez(
            f"{output_dir}/npz_APH/{img_idx:06}.npz",
            lines=lines_pred,
            score=scores,
        )


def read_tplsd(src_pth, output_dir):

    images = sorted(os.listdir(image_path))
    dirs = sorted(glob.glob(osp.join(src_pth, "*.mat")))

    os.makedirs(f"{output_dir}/npz_APH", exist_ok=True)
    print(f"line score npz path: {output_dir}/npz_APH")

    for i, path in enumerate(tqdm(dirs)):

        mat = sio.loadmat(path)
        lines = mat["lines"]
        nlines = lines.reshape(-1, 2, 2)
        nlines = nlines[:, :, ::-1]

        lcnn_line = nlines * (128 / 320)
        lcnn_score = mat["score"]
        lcnn_score = np.squeeze(lcnn_score)

        filename = osp.basename(path)
        if dataname == 'wireframe':
            img_idx = images.index(filename[:-4]+'.jpg')
        else:
            img_idx = images.index(filename[:-4] + '.png')

        np.savez(
            f"{output_dir}/npz_APH/{img_idx:06}.npz",
            lines=lcnn_line,
            score=lcnn_score,
        )


def main(methods):
    args = docopt(__doc__)
    src_dir = args["<src>"]
    tar_dir = args["<dst>"]

    if methods == 'F-Clip':
        pth_apH = osp.join(tar_dir, 'npz_APH')
        read_fclip(src_dir, pth_apH)
        file_list = glob.glob(osp.join(pth_apH, "*.npz"))
        thresh = [0.1, 0.2, 0.25, 0.27, 0.3, 0.315, 0.33, 0.345, 0.36, 0.38, 0.4, 0.42, 0.45, 0.47, 0.49, 0.5, 0.52,
                  0.54, 0.56, 0.58]

    elif methods == 'LCNN':
        file_list = glob.glob(osp.join(src_dir, "*.npz"))
        thresh = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999]

    elif methods == 'HAWP':
        read_hawp(src_dir, tar_dir)
        pth_apH = osp.join(tar_dir, 'npz_APH')
        file_list = glob.glob(osp.join(pth_apH, "*.npz"))
        thresh = [0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.985, 0.99, 0.992, 0.994, 0.995,
                  0.996, 0.997, 0.998, 0.999, 0.9995]

    elif methods == 'TPLSD':
        read_tplsd(src_dir, tar_dir)
        pth_apH = osp.join(tar_dir, 'npz_APH')
        file_list = glob.glob(osp.join(pth_apH, "*.npz"))
        thresh = [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    else:
        raise ValueError()

    output_file = osp.join(tar_dir, "result.mat")
    target_dir = osp.join(tar_dir, "mat")
    os.makedirs(target_dir, exist_ok=True)
    print(f"intermediate matlab results will be saved at: {target_dir}")

    for t in thresh:
        socress = []
        for fname in file_list:
            name = fname.split("/")[-1].split(".")[0]
            mat_name = name + ".mat"
            npz = np.load(fname)
            lines = npz["lines"].reshape(-1, 4)
            scores = npz["score"]

            socress.append(scores)
            if methods in ['LCNN']:
                for j in range(len(scores) - 1):
                    if scores[j + 1] == scores[0]:
                        lines = lines[: j + 1]
                        scores = scores[: j + 1]
                        break

            idx = np.where(scores > t)[0]
            os.makedirs(osp.join(target_dir, str(t)), exist_ok=True)
            sio.savemat(osp.join(target_dir, str(t), mat_name), {"lines": lines[idx]})

    cmd = "matlab -nodisplay -nodesktop "
    cmd += '-r "dbstop if error; '
    cmd += "eval_release('{:s}', '{:s}', '{:s}', '{:s}', {:d}, '{:s}', '{:s}'); quit;\"".format(
        image_path, line_gt_path, output_file, target_dir, output_size, dataname, methods
    )
    print("Running:\n{}".format(cmd))
    os.environ["MATLABPATH"] = "matlab/"
    subprocess.call(cmd, shell=True)

    mat = sio.loadmat(output_file)
    tps = mat["sumtp"]
    fps = mat["sumfp"]
    N = mat["sumgt"]
    rcs = sorted(list((tps / N)[:, 0]))
    prs = sorted(list((tps / np.maximum(tps + fps, 1e-9))[:, 0]))[::-1]

    with open(f"{tar_dir}/aph_and_f1.csv", "a") as fout:
        print(
            f"f1 measure is: {(2 * np.array(prs) * np.array(rcs) / (np.array(prs) + np.array(rcs))).max()}",
            file=fout
        )

    recall = np.concatenate(([0.0], rcs, [1.0]))
    precision = np.concatenate(([0.0], prs, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    with open(f"{tar_dir}/aph_and_f1.csv", "a") as fout:
        print(f"AP-H is: {np.sum((recall[i + 1] - recall[i]) * precision[i + 1])}", file=fout)

    f = interpolate.interp1d(rcs, prs, kind="cubic", bounds_error=False)
    x = np.arange(0, 1, 0.01) * rcs[-1]
    y = f(x)
    plt.plot(x, y, linewidth=3, label="F-Clip")

    f_scores = np.linspace(0.2, 0.8, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color="green", alpha=0.3)
        plt.annotate("f={0:0.1}".format(f_score), xy=(0.9, y[45] + 0.02), alpha=0.4)

    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.0, step=0.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.yticks(np.arange(0, 1.0, step=0.1))
    plt.legend(loc=3)
    plt.title("PR Curve for APH")
    plt.savefig(f"{tar_dir}/apH.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{tar_dir}/apH.svg", format="svg", bbox_inches="tight")
    plt.close()
    # plt.show()


def config_global(resolu, dataset):
    global resolution, dataname, image_path, line_gt_path
    resolution = resolu
    dataname = dataset  # york, shanghaiTech

    if dataname == 'shanghaiTech':
        image_path = "/home/dxl/Data/wireframe/valid-images/"
        line_gt_path = "/home/dxl/Data/wireframe/valid/"
    elif dataname == 'york':
        image_path = "/home/dxl/Data/york/valid-image/"
        line_gt_path = "/home/dxl/Data/york/valid/"
    else:
        raise ValueError


if __name__ == "__main__":

    resolution_ = 128
    dataset = 'york'  # york, shanghaiTech
    methods = 'F-Clip'  # ['F-Clip', 'LCNN', 'HAWP', 'TPLSD']

    config_global(resolution_, dataset)

    plt.tight_layout()
    main(methods)
