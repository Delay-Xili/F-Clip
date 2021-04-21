from collections import OrderedDict, defaultdict

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import torch

from FClip.line_parsing import OneStageLineParsing
from FClip.config import M
from FClip.losses import ce_loss, sigmoid_l1_loss, focal_loss, l12loss
from FClip.nms import structure_nms_torch


class FClip(nn.Module):
    def __init__(self, backbone):
        super(FClip, self).__init__()
        self.backbone = backbone
        self.M_dic = M.to_dict()
        self._get_head_size()

    def _get_head_size(self):

        head_size = []
        for h in self.M_dic['head']['order']:
            head_size.append([self.M_dic['head'][h]['head_size']])

        self.head_off = np.cumsum([sum(h) for h in head_size])

    def lcmap_head(self, output, target):
        name = "lcmap"

        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx-1]

        pred = output[s: self.head_off[offidx]].reshape(self.M_dic['head'][name]['head_size'], batch, row, col)

        if self.M_dic['head'][name]['loss'] == "Focal_loss":
            alpha = self.M_dic['head'][name]['focal_alpha']
            loss = focal_loss(pred, target, alpha)
        elif self.M_dic['head'][name]['loss'] == "CE":
            loss = ce_loss(pred, target, None)
        else:
            raise NotImplementedError

        weight = self.M_dic['head'][name]['loss_weight']
        return pred.permute(1, 0, 2, 3).softmax(1)[:, 1], loss * weight

    def lcoff_head(self, output, target, mask):
        name = 'lcoff'

        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]

        pred = output[s: self.head_off[offidx]].reshape(self.M_dic['head'][name]['head_size'], batch, row, col)

        loss = sum(
            sigmoid_l1_loss(pred[j], target[j], offset=-0.5, mask=mask)
            for j in range(2)
        )

        weight = self.M_dic['head'][name]['loss_weight']
        return pred.permute(1, 0, 2, 3).sigmoid() - 0.5, loss * weight

    def lleng_head(self, output, target, mask):
        name = 'lleng'

        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]

        pred = output[s: self.head_off[offidx]].reshape(batch, row, col)

        if self.M_dic['head'][name]['loss'] == "sigmoid_L1":
            loss = sigmoid_l1_loss(pred, target, mask=mask)
            pred = pred.sigmoid()
        elif self.M_dic['head'][name]['loss'] == "L1":
            loss = l12loss(pred, target, mask=mask)
            pred = pred.clamp(0., 1.)
        else:
            raise NotImplementedError

        weight = self.M_dic['head'][name]['loss_weight']
        return pred, loss * weight

    def angle_head(self, output, target, mask):
        name = 'angle'

        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]

        pred = output[s: self.head_off[offidx]].reshape(batch, row, col)

        if self.M_dic['head'][name]['loss'] == "sigmoid_L1":
            loss = sigmoid_l1_loss(pred, target, mask=mask)
            pred = pred.sigmoid()
        elif self.M_dic['head'][name]['loss'] == "L1":
            loss = l12loss(pred, target, mask=mask)
            pred = pred.clamp(0., 1.)
        else:
            raise NotImplementedError

        weight = self.M_dic['head'][name]['loss_weight']
        return pred, loss * weight

    def jmap_head(self, output, target, n_jtyp):
        name = "jmap"
        _, batch, row, col = output.shape

        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]
        pred = output[s: self.head_off[offidx]].reshape(n_jtyp, self.M_dic['head'][name]['head_size'], batch, row, col)

        if self.M_dic['head'][name]['loss'] == "Focal_loss":
            alpha = self.M_dic['head'][name]['focal_alpha']
            loss = sum(
                    focal_loss(pred[i], target[i], alpha) for i in range(n_jtyp)
                )
        elif self.M_dic['head'][name]['loss'] == "CE":
            loss = sum(
                    ce_loss(pred[i], target[i], None) for i in range(n_jtyp)
                )
        else:
            raise NotImplementedError

        weight = self.M_dic['head'][name]['loss_weight']
        return pred.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1], loss * weight

    def joff_head(self, output, target, n_jtyp, mask):
        name = "joff"

        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]

        pred = output[s: self.head_off[offidx]].reshape(
            n_jtyp, self.M_dic['head'][name]['head_size'], batch, row, col)

        loss = sum(
                    sigmoid_l1_loss(pred[i, j], target[i, j], scale=1.0, offset=-0.5, mask=mask[i])
                    for i in range(n_jtyp)
                    for j in range(2)
                )
        weight = self.M_dic['head'][name]['loss_weight']
        return pred.permute(2, 0, 1, 3, 4).sigmoid() - 0.5, loss * weight

    def lmap_head(self, output, target):
        name = "lmap"
        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]
        pred = output[s: self.head_off[offidx]].reshape(batch, row, col)

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                .mean(2)
                .mean(1)
        )

        weight = self.M_dic['head'][name]['loss_weight']
        return pred.sigmoid(), loss * weight

    def forward(self, input_dict, isTest=False):

        if isTest:
            return self.test_forward(input_dict)
        else:
            return self.trainval_forward(input_dict)

    def test_forward(self, input_dict):

        extra_info = {
            'time_front': 0.0,
            'time_stack0': 0.0,
            'time_stack1': 0.0,
            'time_backbone': 0.0,
        }

        extra_info['time_backbone'] = time.time()
        image = input_dict["image"]
        outputs, feature, backbone_time = self.backbone(image)
        extra_info['time_front'] = backbone_time['time_front']
        extra_info['time_stack0'] = backbone_time['time_stack0']
        extra_info['time_stack1'] = backbone_time['time_stack1']
        extra_info['time_backbone'] = time.time() - extra_info['time_backbone']

        output = outputs[0]

        heatmap = {}
        heatmap["lcmap"] = output[:, 0:                self.head_off[0]].softmax(1)[:, 1]
        heatmap["lcoff"] = output[:, self.head_off[0]: self.head_off[1]].sigmoid() - 0.5
        heatmap["lleng"] = output[:, self.head_off[1]: self.head_off[2]].sigmoid()
        heatmap["angle"] = output[:, self.head_off[2]: self.head_off[3]].sigmoid()

        parsing = True
        if parsing:
            lines, scores = [], []
            for k in range(output.shape[0]):
                line, score = OneStageLineParsing.fclip_torch(
                    lcmap=heatmap["lcmap"][k],
                    lcoff=heatmap["lcoff"][k],
                    lleng=heatmap["lleng"][k],
                    angle=heatmap["angle"][k],
                    delta=M.delta,
                    resolution=M.resolution
                )
                if M.s_nms > 0:
                    line, score = structure_nms_torch(line, score, M.s_nms)
                lines.append(line[None])
                scores.append(score[None])

            heatmap["lines"] = torch.cat(lines)
            heatmap["score"] = torch.cat(scores)
        return {'heatmaps': heatmap, 'extra_info': extra_info}

    def trainval_forward(self, input_dict):

        image = input_dict["image"]
        outputs, feature, backbone_time = self.backbone(image)
        result = {"feature": feature}
        batch, channel, row, col = outputs[0].shape
        T = input_dict["target"].copy()
        n_jtyp = 1

        T["lcoff"] = T["lcoff"].permute(1, 0, 2, 3)

        losses = []
        accuracy = []
        for stack, output in enumerate(outputs):
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()

            L = OrderedDict()
            Acc = OrderedDict()
            heatmap = {}
            lcmap, L["lcmap"] = self.lcmap_head(output, T["lcmap"])
            lcoff, L["lcoff"] = self.lcoff_head(output, T["lcoff"], mask=T["lcmap"])
            heatmap["lcmap"] = lcmap
            heatmap["lcoff"] = lcoff

            lleng, L["lleng"] = self.lleng_head(output, T["lleng"], mask=T["lcmap"])
            angle, L["angle"] = self.angle_head(output, T["angle"], mask=T["lcmap"])
            heatmap["lleng"] = lleng
            heatmap["angle"] = angle

            losses.append(L)
            accuracy.append(Acc)

            if stack == 0 and input_dict["do_evaluation"]:
                result["heatmaps"] = heatmap

        result["losses"] = losses
        result["accuracy"] = accuracy

        return result


