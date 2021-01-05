import numpy as np
import torch


def pline(x1, y1, x2, y2, x, y):
    """the L2 distance from n(x,y) to line n1 n2"""
    px = x2 - x1
    py = y2 - y1
    dd = px * px + py * py
    u = ((x - x1) * px + (y - y1) * py) / max(1e-9, float(dd))
    dx = x1 + u * px - x
    dy = y1 + u * py - y
    return dx * dx + dy * dy


def psegment(x1, y1, x2, y2, x, y):
    px = x2 - x1
    py = y2 - y1
    dd = px * px + py * py
    u = max(min(((x - x1) * px + (y - y1) * py) / float(dd), 1), 0)
    dx = x1 + u * px - x
    dy = y1 + u * py - y
    return dx * dx + dy * dy


def plambda(x1, y1, x2, y2, x, y):
    """
    project the n(x,y) on line n1(x1, y1), n2(x2, y2), the plambda will return the
    ratio of line (n n1) for line (n1, n2)
    :return:
    """
    px = x2 - x1
    py = y2 - y1
    dd = px * px + py * py
    return ((x - x1) * px + (y - y1) * py) / max(1e-9, float(dd))


def postprocess(lines, scores, threshold=0.01, tol=1e9, do_clip=False):
    nlines, nscores = [], []
    for (p, q), score in zip(lines, scores):
        start, end = 0, 1
        for a, b in nlines:
            if (
                min(
                    max(pline(*p, *q, *a), pline(*p, *q, *b)),
                    max(pline(*a, *b, *p), pline(*a, *b, *q)),
                )
                > threshold ** 2
            ):
                continue
            lambda_a = plambda(*p, *q, *a)
            lambda_b = plambda(*p, *q, *b)
            if lambda_a > lambda_b:
                lambda_a, lambda_b = lambda_b, lambda_a
            lambda_a -= tol
            lambda_b += tol

            # case 1: skip (if not do_clip)
            # current line cover the existent line which has higher score.
            if start < lambda_a and lambda_b < end:
                # start = 10  # drop
                continue

            # not intersect
            # no overlap
            if lambda_b < start or lambda_a > end:
                continue

            # cover
            # the current line be covered by line with higher score
            if lambda_a <= start and end <= lambda_b:
                start = 10  # drop
                break

            # case 2 & 3:
            if lambda_a <= start and start <= lambda_b:
                start = lambda_b
            if lambda_a <= end and end <= lambda_b:
                end = lambda_a

            if start >= end:
                break

        if start >= end:
            continue
        nlines.append(np.array([p + (q - p) * start, p + (q - p) * end]))
        nscores.append(score)
    return np.array(nlines), np.array(nscores)


def acc_postprocess(gtlines, lines, scores, threshold=1, is_cover=True, is_becover=True, overlap_fraction=0):

    lines1 = gtlines.copy()
    lines2 = lines.copy()

    def get_u_line(_lines, points):
        """N lines to M points u score and l2 distance"""
        p = _lines[:, 1] - _lines[:, 0]  # (N, 2)
        dd = np.sum(p ** 2, -1)   # (N,)
        pv1 = points[:, None] - _lines[:, 0][None]  # (M, N, 2)
        inner_u = np.sum(pv1 * p[None], -1)  # (M, N)
        u = inner_u / np.clip(dd[None], 1e-9, 1e9)  # (M, N)

        v1p = - points[:, None] + _lines[:, 0][None]  # (M, N, 2)
        d = v1p + u[:, :, None] * p[None]  # (M, N ,2)

        pline = np.sum(d ** 2, -1)  # (M, N)

        return u.transpose(), pline.transpose()  # (N, M)

    lambda_a12, pqa12 = get_u_line(lines1, lines2[:, 0])
    lambda_b12, pqb12 = get_u_line(lines1, lines2[:, 1])
    dis12 = np.concatenate([pqa12[:, :, None], pqb12[:, :, None]], -1)
    dist12 = np.amax(dis12, -1)

    lambda_a21, pqa21 = get_u_line(lines2, lines1[:, 0])
    lambda_b21, pqb21 = get_u_line(lines2, lines1[:, 1])
    dis21 = np.concatenate([pqa21[:, :, None], pqb21[:, :, None]], -1)
    dist21 = np.amax(dis21, -1)

    distmin = np.concatenate([dist12[:, :, None], np.transpose(dist21)[:, :, None]], -1)
    distm = np.amin(distmin, -1)

    mask = distm < threshold
    if (lines1 == lines2).all():
        diag = np.eye(len(mask)).astype(np.bool)
        mask[diag] = False

    k = 0
    hit = np.zeros((len(mask),)).astype(np.bool)
    lambda_a = lambda_a12  # each row means all u for one line
    lambda_b = lambda_b12
    while k < len(mask) - 2:

        if hit[k]:
            k += 1
            continue
        else:
            cline = mask[k, k+1:]
            cline_ab = np.concatenate([lambda_a[k, k+1:][None], lambda_b[k, k+1:][None]], 0)
            cline_a = np.amin(cline_ab, 0)
            cline_b = np.amax(cline_ab, 0)

            cover = (cline_a > 0) & (cline_b < 1)
            be_covered = (cline_a < 0) & (cline_b > 1)

            overlap1 = (cline_a < 0) & (cline_b > overlap_fraction)
            overlap2 = (cline_a < 1 - overlap_fraction) & (cline_b > 1)
            overlap = overlap1 | overlap2

            if is_cover:
                remove = cover  # cover | be_covered
            else:
                remove = np.zeros_like(cover).astype(np.bool)

            if is_becover:
                remove = remove | be_covered

            if overlap_fraction > 0:
                remove = remove | overlap

            hit[k+1:] = hit[k+1:] | (cline & remove)
            k += 1

    drop = ~hit
    return lines[drop], scores[drop]


def acc_postprocess_torch(gtlines, lines, scores, threshold=1, is_cover=True, is_becover=True, overlap_fraction=0):
    lines1 = gtlines.clone()
    lines2 = lines.clone()

    def get_u_line(_lines, points):
        """N lines to M points u score and l2 distance"""
        p = _lines[:, 1] - _lines[:, 0]  # (N, 2)
        dd = (p ** 2).sum(-1) # (N,)
        pv1 = points[:, None] - _lines[:, 0][None]  # (M, N, 2)
        inner_u = torch.sum(pv1 * p[None], -1)  # (M, N)
        u = inner_u / dd[None].clamp(1e-9, 1e9)  # (M, N)

        v1p = - points[:, None] + _lines[:, 0][None]  # (M, N, 2)
        d = v1p + u[:, :, None] * p[None]  # (M, N ,2)
        pline = (d ** 2).sum(-1)  # (M, N)

        return u.transpose(0, 1), pline.transpose(0, 1)  # (N, M)

    lambda_a12, pqa12 = get_u_line(lines1, lines2[:, 0])
    lambda_b12, pqb12 = get_u_line(lines1, lines2[:, 1])
    dis12 = torch.cat([pqa12[:, :, None], pqb12[:, :, None]], -1)
    dist12, _ = torch.max(dis12, -1)

    lambda_a21, pqa21 = get_u_line(lines2, lines1[:, 0])
    lambda_b21, pqb21 = get_u_line(lines2, lines1[:, 1])
    dis21 = torch.cat([pqa21[:, :, None], pqb21[:, :, None]], -1)
    dist21, _ = torch.max(dis21, -1)

    distmin = torch.cat([dist12[:, :, None], dist21.transpose(0, 1)[:, :, None]], -1)
    distm, _ = torch.min(distmin, -1)

    mask = distm < threshold
    if (lines1 == lines2).all():
        diag = torch.eye(len(mask)).bool()
        # diag = np.eye(len(mask)).astype(np.bool)
        mask[diag] = False

    k = 0
    hit = torch.zeros((len(mask),), device=mask.device).bool()
    lambda_a = lambda_a12  # each row means all u for one line
    lambda_b = lambda_b12

    while k < len(mask) - 2:

        if hit[k]:
            k += 1
            continue
        else:
            cline = mask[k, k+1:]
            cline_ab = torch.cat([lambda_a[k, k+1:][None], lambda_b[k, k+1:][None]], 0)
            cline_a, _ = torch.min(cline_ab, 0)
            cline_b, _ = torch.max(cline_ab, 0)

            cover = (cline_a > 0) & (cline_b < 1)
            be_covered = (cline_a < 0) & (cline_b > 1)

            overlap1 = (cline_a < 0) & (cline_b > overlap_fraction)
            overlap2 = (cline_a < 1 - overlap_fraction) & (cline_b > 1)
            overlap = overlap1 | overlap2

            if is_cover:
                remove = cover  # cover | be_covered
            else:
                remove = torch.zeros_like(cover).bool()

            if is_becover:
                remove = remove | be_covered

            if overlap_fraction > 0:
                remove = remove | overlap

            hit[k+1:] = hit[k+1:] | (cline & remove)
            k += 1

    drop = ~hit
    return lines[drop], scores[drop]

