import torch
import torch.nn.functional as F


def lt(logits):
    """long tail function"""
    logits_ = torch.relu(logits)
    # return 1 - torch.exp(-logits_)
    return torch.exp(-logits_)


def lt_loss(logits, target, mask=None):
    """long tail distribution loss"""
    logits = lt(logits)
    loss = torch.abs(logits - target)

    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)


def logl1loss(logits, target, mask=None):

    loss = torch.abs(torch.log(logits) - torch.log(target))
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def l12loss(logits, target, mask=None, loss="L1"):
    if loss == "L1":
        loss = torch.abs(logits - target)
    elif loss == "L2":
        loss = (logits - target) ** 2
    else:
        raise ValueError("no such loss")
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)


def angle_l1_loss(logits, target, scale=1.0, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) * scale + offset
    loss = (logp - target).abs()
    loss = torch.cat([loss[..., None], 1.0 - loss[..., None]], -1)
    loss, _ = torch.min(loss, dim=-1)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)


def sigmoid_l1_loss(logits, target, scale=1.0, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) * scale + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)


def sigmoid_l1_loss3(logits, target, scale=1.0, offset=0.0, mask=None):
    # mask: (bs, c, h, w)
    # logits: (bs, c, h, w)
    # target: (bs, c, h, w)
    logp = torch.sigmoid(logits) * scale + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(3, True).mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(3).mean(2).mean(1)


def sigmoid_l1_lossn(logits, target, scale=1.0, offset=0.0, mask=None):
    # mask: (bs, ...)
    # logits: (bs, ...)
    # target: (bs, ...)
    bs = logits.shape[0]
    logp = torch.sigmoid(logits) * scale + offset
    loss = torch.abs(logp - target)

    loss = loss.reshape(bs, -1)
    w = mask.reshape(bs, -1)

    loss = loss * w

    return loss.sum(1) / w.sum(1)


def l1_lossn(logits, target, mask=None):
    # mask: (bs, ...)
    # logits: (bs, ...)
    # target: (bs, ...)
    bs = logits.shape[0]
    loss = torch.abs(logits - target)

    loss = loss.reshape(bs, -1)
    w = mask.reshape(bs, -1)

    loss = loss * w

    return loss.sum(1) / w.sum(1)


def ce_loss(logits, label, alpha=None):
    """
    merge 2 cls and multi-cls ce loss
    :param logits: [cls_num, bs, h, w]
    :param label: [cls_num, bs, h, w]
    :return:
    """

    cls_num, bs, h, w = logits.shape
    label = label.reshape(-1, bs, h, w)
    if cls_num == 2:
        label = torch.cat([1 - label, label], 0)  # label: [2, bs, h, w]

    nlogp = F.log_softmax(logits, 0)
    if cls_num == 2:
        loss = label * nlogp
        return -loss.sum(0).mean(2).mean(1)
        # loss = label * torch.log(p.clamp(1e-12, 1.0))
        # return -loss.sum(0).sum(2).sum(1).div(label[1].sum(2).sum(1))
    else:
        loss = label * nlogp
        w = label.sum(0).sum(2).sum(1)
        loss_ = -loss.sum(0).sum(2).sum(1)
        return loss_ / w


def focal_loss(logits, label, alpha):
    """
    TODO
    another version of focal loss from corner net
    only for two cls
    :param logits: [cls_num, bs, h, w]
    :param label: [bs, h, w]
    :return:
    """
    cls_num, bs, h, w = logits.shape
    assert cls_num == 2
    # label = label.reshape(-1, bs, h, w)

    # label = torch.cat([1 - label, label], 0)  # label: [2, bs, h, w]

    # alpha = M.focal_alpha
    # beta = 4.
    logp = F.log_softmax(logits, 0)
    p = F.softmax(logits, 0)

    loss = label * (p[0] ** alpha) * logp[1] + \
           (1 - label) * (p[1] ** alpha) * logp[0]
    mask = label.sum(2).sum(1)
    return -loss.sum(2).sum(1) / mask


def sigmoid_focal_loss(logits, labels, alpha):
    """

    :param logits: (len_resolu, ang_resolu, bs, h, w)
    :param labels: (len_resolu, ang_resolu, bs, h, w)
    :param alpha:
    :return:
    """

    # logp = F.logsigmoid(logits)
    p = torch.sigmoid(logits)

    loss = labels * ((1-p) ** alpha) * torch.log(p.clamp(1e-12, 1)) + \
           (1 - labels) * (p ** alpha) * torch.log((1-p).clamp(1e-12, 1))

    mask = labels.sum(4).sum(3).sum(1).sum(0)
    return -loss.sum(4).sum(3).sum(1).sum(0) / mask


def nms_ce_loss(logits, positive, loss_weight):
    nlogp = -F.log_softmax(logits, dim=0)
    loss_m = positive * nlogp[1] + (1 - positive) * nlogp[0]

    ap = F.avg_pool2d(loss_m, 3, stride=1, padding=1)

    w = positive.mean(2, True).mean(1, True)
    w[w == 0] = 1

    loss1 = ((positive * ap) * (positive / w)).mean(2).mean(1)
    loss2 = loss_m.mean(2).mean(1)

    return loss1 / loss_weight + loss2


def cls_acc(logits, label, mask):

    cls_num, bs, h, w = logits.shape
    label = label.reshape(-1, bs, h, w)
    if cls_num == 2:
        label = torch.cat([1 - label, label], 0)  # label: [2, bs, h, w]

    predict = logits.argmax(0)
    if cls_num == 2:
        # mask = label[1]
        gt = label.argmax(0)
        acc = ((predict == gt).float() * mask).sum(2).sum(1)
        w = mask.sum(2).sum(1)

        return acc / w
    else:
        # mask = label.sum(0)
        gt = label.argmax(0)
        acc = ((predict == gt).float() * mask).sum(2).sum(1)
        w = mask.sum(2).sum(1)

        return acc / w


# =================================================== #

def balanced_positive_negative_sampler(matched_idxs, batch_size_per_image, positive_fraction):
    """
    Arguments:
        matched idxs: list of tensors containing -1, 0 or positive values.
            Each tensor corresponds to a specific image.
            -1 values are ignored, 0 are considered as negatives and > 0 as
            positives.

    Returns:
        pos_idx (list[tensor])
        neg_idx (list[tensor])

    Returns two lists of binary masks for each image.
    The first list contains the positive elements that were selected,
    and the second list the negative example.
    """
    pos_idx = []
    neg_idx = []
    for matched_idxs_per_image in matched_idxs:
        positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
        negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

        num_pos = int(batch_size_per_image * positive_fraction)
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        num_neg = batch_size_per_image - num_pos
        # protect against not enough negative examples
        num_neg = min(negative.numel(), num_neg)

        # randomly select positive and negative examples
        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx_per_image = positive[perm1]
        neg_idx_per_image = negative[perm2]

        # create binary mask from indices
        pos_idx_per_image_mask = torch.zeros_like(
            matched_idxs_per_image, dtype=torch.bool
        )
        neg_idx_per_image_mask = torch.zeros_like(
            matched_idxs_per_image, dtype=torch.bool
        )
        pos_idx_per_image_mask[pos_idx_per_image] = 1
        neg_idx_per_image_mask[neg_idx_per_image] = 1

        pos_idx.append(pos_idx_per_image_mask)
        neg_idx.append(neg_idx_per_image_mask)

    return pos_idx, neg_idx


def anchor_loss(logits, labels, batch_size_per_image, positive_fraction):

    batch, num_cls, h, w = logits.shape

    lleng_label = [labels[i].flatten() for i in range(batch)]
    pos_idx, neg_idx = balanced_positive_negative_sampler(lleng_label, batch_size_per_image, positive_fraction)
    sample_ind = torch.cat([(pos_idx[i] | neg_idx[i]).unsqueeze(0) for i in range(batch)], 0)

    cls_loss = F.binary_cross_entropy_with_logits(logits.reshape(batch, -1)[sample_ind], labels.reshape(batch, -1)[sample_ind])

    return cls_loss
