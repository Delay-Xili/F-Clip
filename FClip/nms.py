import numpy as np
import torch
import torch.nn.functional as F


def non_maximum_suppression(a, delta=0., kernel=3):
    ap = F.max_pool2d(a, kernel, stride=1, padding=int((kernel-1)/2))
    mask = (a == ap).float().clamp(min=0.0)
    mask_n = (~mask.bool()).float() * delta
    return a * mask + a * mask_n


def structure_nms(lines, scores, threshold=2, delta=0.0):
    diff = ((lines[:, None, :, None] - lines[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )

    idx = diff <= threshold
    ii = np.eye(len(idx)).astype(np.bool)
    idx[ii] = False

    i = 1
    k = lines.shape[0]
    hit = idx[0]
    while i < k - 2:
        if hit[i]:
            i += 1
            continue
        else:
            hit[i+1:] = hit[i+1:] | idx[i, i+1:]
            i += 1

    # hit = hit * delta
    keep = ~hit
    drop = hit * delta
    scale = keep + drop

    return lines, scores * scale


def structure_nms_torch(lines, scores, threshold=2):
    diff = ((lines[:, None, :, None] - lines[:, None]) ** 2).sum(-1)
    diff0, diff1 = diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    diff = torch.cat([diff0[..., None], diff1[..., None]], -1)
    diff, _ = torch.min(diff, -1)

    idx = diff <= threshold
    ii = torch.eye(len(lines), device=lines.device).bool()
    idx[ii] = False

    i = 1
    k = lines.shape[0]
    hit = idx[0]
    while i < k - 2:
        if hit[i]:
            i += 1
            continue
        else:
            hit[i + 1:] = hit[i + 1:] | idx[i, i + 1:]
            i += 1

    drop = ~hit
    return lines[drop], scores[drop]
