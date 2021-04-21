"""
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) Yichao Zhou (LCNN)
(c) YANG, Wei
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from FClip.config import M
import time

__all__ = ["HourglassNet", "hg"]


class BottleneckLine(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckLine, self).__init__()

        # self.ks = (M.line_kernel, 1)
        # self.padding = (int(M.line_kernel / 2), 0)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.conv2D = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        # self.conv2v = nn.Conv2d(planes, planes, kernel_size=(M.line_kernel, 1), padding=(int(M.line_kernel / 2), 0))
        # self.conv2h = nn.Conv2d(planes, planes, kernel_size=(1, M.line_kernel), padding=(0, int(M.line_kernel / 2)))

        self.conv2, self.merge = self.build_line_layers(planes)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck2D.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def build_line_layers(self, planes):
        layer = []
        if "s" in M.line.mode:
            layer.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1))

        if "v" in M.line.mode:
            layer.append(nn.Conv2d(planes, planes, kernel_size=(M.line_kernel, 1), padding=(int(M.line_kernel / 2), 0)))

        if "h" in M.line.mode:
            layer.append(nn.Conv2d(planes, planes, kernel_size=(1, M.line_kernel), padding=(0, int(M.line_kernel / 2))))

        assert len(layer) > 0

        if M.merge == 'cat':
            merge = nn.Conv2d(planes * len(layer), planes, kernel_size=1)
        elif M.merge == 'maxpool':
            ll = len(M.line.mode)
            merge = nn.MaxPool3d((ll, 1, 1), stride=(ll, 1, 1))
        else:
            raise ValueError()

        return nn.ModuleList(layer), merge

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)

        if M.merge == 'cat':
            tt = torch.cat([conv(out) for conv in self.conv2], dim=1)
        elif M.merge == 'maxpool':
            tt = torch.cat([torch.unsqueeze(conv(out), 2) for conv in self.conv2], dim=2)
        else:
            raise ValueError()
        out = self.merge(tt)
        out = torch.squeeze(out, 2)

        # print(out.size())
        # exit()

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck2D.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck1D_v(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1D_v, self).__init__()

        self.ks = (M.line_kernel, 1)
        self.padding = (int(M.line_kernel / 2), 0)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=self.ks, stride=stride, padding=self.padding)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck2D.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck1D_h(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1D_h, self).__init__()

        self.ks = (1, M.line_kernel)
        self.padding = (0, int(M.line_kernel / 2))

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=self.ks, stride=stride, padding=self.padding)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck2D.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


block_dicts = {
    "Bottleneck2D": Bottleneck2D,
    "Bottleneck1D_v": Bottleneck1D_v,
    "Bottleneck1D_h": Bottleneck1D_h,
    "BottleneckLine": BottleneckLine,
}


class HourglassNet(nn.Module):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self, head, depth, num_stacks, num_blocks, num_classes):
        super(HourglassNet, self).__init__()

        block2D = Bottleneck2D
        # block1D_v = Bottleneck1D_v
        # block1D_h = Bottleneck1D_h

        branch_blocks = []
        for key in M.branch_blocks:
            branch_blocks.append(block_dicts[key])

        self.inplanes = M.inplanes
        self.num_feats = self.inplanes * block2D.expansion
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block2D, self.inplanes, 1)
        self.layer2 = self._make_residual(block2D, self.inplanes, 1)
        self.layer3 = self._make_residual(block2D, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        # build hourglass modules
        ch = self.num_feats * block2D.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        merge_fc = []
        for i in range(num_stacks):
            sub_hg, sub_res, sub_fc = [], [], []
            for bb in branch_blocks:
                sub_hg.append(Hourglass(bb, num_blocks, self.num_feats, depth))
                sub_res.append(self._make_residual(bb, self.num_feats, num_blocks))
                sub_fc.append(self._make_fc(ch, ch))
            hg.append(nn.ModuleList(sub_hg))
            res.append(nn.ModuleList(sub_res))
            fc.append(nn.ModuleList(sub_fc))
            merge_fc.append(self._make_fc(int(ch*len(sub_fc)), ch))

            score.append(head(ch, num_classes))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.merge_fc = nn.ModuleList(merge_fc)

        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        bn = nn.BatchNorm2d(outplanes)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):

        extra_info = {
            'time_front': 0.0,
            'time_stack0': 0.0,
            'time_stack1': 0.0,
        }

        t = time.time()
        out = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        extra_info['time_front'] = time.time() - t

        for i in range(self.num_stacks):
            t = time.time()
            feat = []
            for j in range(len(self.hg[i])):
                y = self.hg[i][j](x)
                y = self.res[i][j](y)
                y = self.fc[i][j](y)
                feat.append(y)

            y = self.merge_fc[i](torch.cat(feat, dim=1))
            score = self.score[i](y)
            out.append(score)

            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

            extra_info[f"time_stack{i}"] = time.time() - t

        return out[::-1], y, extra_info  # out_vps[::-1]


def hg(**kwargs):
    model = HourglassNet(
        # Bottleneck2D,
        head=kwargs.get("head", lambda c_in, c_out: nn.Conv2d(c_in, c_out, 1)),
        depth=kwargs["depth"],
        num_stacks=kwargs["num_stacks"],
        num_blocks=kwargs["num_blocks"],
        num_classes=kwargs["num_classes"],
    )
    return model


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")
