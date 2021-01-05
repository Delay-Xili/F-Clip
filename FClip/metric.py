import numpy as np


def ap(tp, fp):
    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[i + 1] - recall[i]) * precision[i + 1])


def APJ(vert_pred, vert_gt, max_distance, im_ids):
    if len(vert_pred) == 0:
        return 0

    vert_pred = np.array(vert_pred)
    vert_gt = np.array(vert_gt)

    confidence = vert_pred[:, -1]
    idx = np.argsort(-confidence)
    vert_pred = vert_pred[idx, :]
    im_ids = im_ids[idx]
    n_gt = sum(len(gt) for gt in vert_gt)

    nd = len(im_ids)
    tp, fp = np.zeros(nd, dtype=np.float), np.zeros(nd, dtype=np.float)
    hit = [[False for _ in j] for j in vert_gt]

    for i in range(nd):
        gt_juns = vert_gt[im_ids[i]]
        pred_juns = vert_pred[i][:-1]
        if len(gt_juns) == 0:
            continue
        dists = np.linalg.norm((pred_juns[None, :] - gt_juns), axis=1)
        choice = np.argmin(dists)
        dist = np.min(dists)
        if dist < max_distance and not hit[im_ids[i]][choice]:
            tp[i] = 1
            hit[im_ids[i]][choice] = True
        else:
            fp[i] = 1

    tp = np.cumsum(tp) / n_gt
    fp = np.cumsum(fp) / n_gt
    return ap(tp, fp)


def mAPJ(pred, truth, distances, im_ids):
    return sum(APJ(pred, truth, d, im_ids) for d in distances) / len(distances) * 100
    # return [APJ(pred, truth, d, im_ids) * 100 for d in distances]


def msTPFP(line_pred, line_gt, threshold):
    diff = ((line_pred[:, None, :, None] - line_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )
    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(len(line_gt), np.bool)
    tp = np.zeros(len(line_pred), np.float)
    fp = np.zeros(len(line_pred), np.float)
    for i in range(len(line_pred)):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def msTPFP_hit(line_pred, line_gt, threshold):
    diff = ((line_pred[:, None, :, None] - line_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )
    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(len(line_gt), np.bool)
    tp = np.zeros(len(line_pred), np.float)
    fp = np.zeros(len(line_pred), np.float)
    for i in range(len(line_pred)):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp, hit


def msAP(line_pred, line_gt, threshold):
    tp, fp = msTPFP(line_pred, line_gt, threshold)
    tp = np.cumsum(tp) / len(line_gt)
    fp = np.cumsum(fp) / len(line_gt)
    return ap(tp, fp)
