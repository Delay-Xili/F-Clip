
import scipy.io as sio
import os
import glob
import numpy as np
from docopt import docopt

import matplotlib.pyplot as plt
from scipy import interpolate

import FClip.utils
import FClip.metric
from FClip.config import C
from FClip.line_parsing import line_parsing_from_npz
from tqdm import tqdm


C.update(C.from_yaml(filename="config/fclip_HG1_320.yaml"))

_resolution = 128
_num = 462

_delta = 0.8
_nlines = 1000
_s_nms = 0.
_s_nms_d = 0.
_error = 0
# _display = {"name": 's_nms delta:', "value": _s_nms_d}
_display = None


def line_score_for_tplsd(path, GT, threshold=5):
    preds = sorted(glob.glob(path))[:_num]
    # gts = sorted(glob.glob(GT))
    gts = sorted(glob.glob(GT))[:_num]

    # print(path)

    n_gt = 0
    n_pt = 0
    lcnn_tp, lcnn_fp, lcnn_scores = [], [], []
    for pred_name, gt_name in tqdm(zip(preds, gts)):
        mat = sio.loadmat(pred_name)
        lines = mat["lines"]
        nlines = lines.reshape(-1, 2, 2)
        nlines = nlines[:, :, ::-1]

        lcnn_line = nlines * (128 / 320)
        lcnn_score = mat["score"]
        lcnn_score = np.squeeze(lcnn_score)

        # print(nlines)

        with np.load(gt_name) as fgt:
            gt_line = fgt["lpos"][:, :, :2]

        n_gt += len(gt_line)
        # for i in range(len(lcnn_line)):
        #     if i > 0 and (lcnn_line[i] == lcnn_line[0]).all():
        #         lcnn_line = lcnn_line[:i]
        #         lcnn_score = lcnn_score[:i]
        #         break

        n_pt += len(lcnn_line)
        tp, fp = FClip.metric.msTPFP(lcnn_line, gt_line, threshold)
        lcnn_tp.append(tp)
        lcnn_fp.append(fp)
        lcnn_scores.append(lcnn_score)

    lcnn_tp = np.concatenate(lcnn_tp)
    lcnn_fp = np.concatenate(lcnn_fp)

    lcnn_scores = np.concatenate(lcnn_scores)
    lcnn_index = np.argsort(-lcnn_scores)
    lcnn_tp = np.cumsum(lcnn_tp[lcnn_index]) / n_gt
    lcnn_fp = np.cumsum(lcnn_fp[lcnn_index]) / n_gt

    return lcnn_tp, lcnn_fp, n_gt


def line_score(path, GT, threshold=5):
    preds = sorted(glob.glob(path))[:_num]
    # gts = sorted(glob.glob(GT))
    gts = sorted(glob.glob(GT))[:_num]

    n_gt = 0
    n_pt = 0
    lcnn_tp, lcnn_fp, lcnn_scores = [], [], []
    for pred_name, gt_name in tqdm(zip(preds, gts)):
        with np.load(pred_name) as fpred:
            lcnn_line = fpred["lines"][:, :, :2] * (128 / _resolution)
            lcnn_score = fpred["score"]
            # lcnn_score = fpred["lscore"]

        with np.load(gt_name) as fgt:
            gt_line = fgt["lpos"][:, :, :2]

        n_gt += len(gt_line)
        for i in range(len(lcnn_line)):
            if i > 0 and (lcnn_line[i] == lcnn_line[0]).all():
                lcnn_line = lcnn_line[:i]
                lcnn_score = lcnn_score[:i]
                break

        n_pt += len(lcnn_line)
        tp, fp = FClip.metric.msTPFP(lcnn_line, gt_line, threshold)
        lcnn_tp.append(tp)
        lcnn_fp.append(fp)
        lcnn_scores.append(lcnn_score)

    lcnn_tp = np.concatenate(lcnn_tp)
    lcnn_fp = np.concatenate(lcnn_fp)

    lcnn_scores = np.concatenate(lcnn_scores)
    lcnn_index = np.argsort(-lcnn_scores)
    lcnn_tp = np.cumsum(lcnn_tp[lcnn_index]) / n_gt
    lcnn_fp = np.cumsum(lcnn_fp[lcnn_index]) / n_gt

    return lcnn_tp, lcnn_fp, n_gt


def line_center_score(path, GT, threshold=5):
    preds = sorted(glob.glob(path))[:_num]
    # gts = sorted(glob.glob(GT))
    gts = sorted(glob.glob(GT))[:_num]

    n_gt = 0
    n_pt = 0
    tps, fps, scores = [], [], []

    # ltype = "slcnn+junc"
    # ang_type = None
    ltype = "slcnn"

    for pred_name, gt_name in tqdm(zip(preds, gts)):
        with np.load(gt_name) as fgt:
            gt_line = fgt["lpos"][:, :, :2]

        line, score = line_parsing_from_npz(
            pred_name, ltype, ang_type="radian",
            delta=_delta, nlines=_nlines,
            s_nms=_s_nms,
            nms_d=_s_nms_d,
            error=_error,
        )
        line = line * (128 / _resolution)

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

    return lcnn_tp, lcnn_fp, n_gt


def draw_sap_pr_curve():

    data_name = 'huangkun'   # 'huangkun', 'york'

    GT_york = f"/home/dxl/Data/york/valid/*_label.npz"
    GT_huang = f"/home/dxl/Data/wireframe/valid/*_label.npz"

    name = ['F-Clip', 'TP-LSD', 'HAWP', 'LCNN']
    color = ['r', 'b', 'g', 'darkorange']
    npz_pth_wf = [
        "logs/slcnn/stage2-June-/DataAug/hr/201107-041812-slcnn-DataAug3w-hr220-bs6-crop0.9-1.6/npz/006588000",
        "/home/dxl/Code/TP-LSD/log/test_wireframe/mat/lmbd0.5/0.01",
        "/home/dxl/Code/tmp/hawp/outputs/hawp/wireframe_test.npz",
        "/home/dxl/Code/lcnn/logs/default_ckpt/npz_wireframe"
    ]

    npz_pth_york = [
        "logs/slcnn/stage2-June-/DataAug/hr/201107-041812-slcnn-DataAug3w-hr220-bs6-crop0.9-1.6/york/best_ckpt",
        "/home/dxl/Code/TP-LSD/log/test_york/mat/lmbd0.5/0.01",
        "/home/dxl/Code/tmp/hawp/outputs/hawp/york_test.npz",
        "/home/dxl/Code/lcnn/logs/default_ckpt/npz_york"
    ]

    if data_name == 'york':
        npz_pth = npz_pth_york
        GT = GT_york
    elif data_name == 'huangkun':
        npz_pth = npz_pth_wf
        GT = GT_huang
    else:
        raise ValueError()

    for k, pth in enumerate(npz_pth):

        if name[k] == "LCNN":
            tps, fps, n_gts = line_score(pth+"/*.npz", GT, 10)
        elif name[k] == 'TP-LSD':
            tps, fps, n_gts = line_score_for_tplsd(pth+"/*.mat", GT, 10)
        elif name[k] == "F-Clip":
            global _nlines, _s_nms
            _nlines = 5000
            _s_nms = 2.
            tps, fps, n_gts = line_center_score(pth + "/*.npz", GT, 10)
            _nlines = 1000
            _s_nms = 0.
        elif name[k] == "HAWP":
            with np.load(pth) as npz:
                tps = npz['tps']
                fps = npz['fps']
        else:
            raise ValueError()

        rcs = sorted(list(tps))
        prs = sorted(list(tps / np.maximum(tps + fps, 1e-9)))[::-1]

        print("sAP-10:", FClip.metric.ap(tps, fps))

        f = interpolate.interp1d(rcs, prs, bounds_error=False)
        x = np.arange(0, 1, 0.01) * rcs[-1]
        y = f(x)
        plt.plot(x, y, c=color[k], linewidth=3, label=name[k])

    f_scores = np.linspace(0.2, 0.8, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color="green", alpha=0.3)
        plt.annotate("f={0:0.1}".format(f_score), xy=(0.9, y[45] + 0.02), alpha=0.4)

    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.0, step=0.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.yticks(np.arange(0, 1.0, step=0.1))
    plt.legend(loc=3)
    plt.title("PR Curve for sAP-10")
    plt.savefig(f"sAP_pr_curve_{data_name}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"sAP_pr_curve_{data_name}.svg", format="svg", bbox_inches="tight")
    plt.close()


def draw_aph_pr_curve():


    pass


if __name__ == '__main__':
    draw_sap_pr_curve()