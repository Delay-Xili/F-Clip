import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import io
import torch.nn.functional as F

from FClip.config import C
from FClip.line_parsing import OneStageLineParsing


def get_global():
    _scale = C.model.resolution / 128
    _len_scale = 64 * _scale
    _ang_scale = 180
    _pscale = 512 / C.model.resolution
    return _scale, _len_scale, _ang_scale, _pscale


class VisualizeResults(object):
    def __init__(self):
        cmap = plt.get_cmap("jet")
        norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        self.sm = sm
        self.train_loss = []  # (iteration, loss)
        self.val_loss = []

    def plot_loss(self, path):
        plt.figure()
        train_loss = np.array(self.train_loss)
        plt.plot(train_loss[:, 0], train_loss[:, 1], 'b-', label='train-loss')
        val_loss = np.array(self.val_loss)
        plt.plot(val_loss[:, 0], val_loss[:, 1], 'r-', label='val-loss')
        plt.ylim([0, 2])
        plt.legend()
        plt.savefig(f"{path}/loss.jpg"), plt.close()

    def c(self, x):
        return self.sm.to_rgba(x)

    def imshow(self, im):
        plt.close()
        plt.tight_layout()
        plt.imshow(im)
        plt.colorbar(self.sm, fraction=0.046)
        plt.xlim([0, im.shape[0]])
        plt.ylim([im.shape[0], 0])

    def plot_samples(self, fn, i, result, target, meta, prefix):
        """

        :param fn:
        :param i:
        :param result:  {
                           'lcmap': [N, H, W]
                           'lcoff': [N, 2, H, W]
                           'lleng': [N, H, W]
                           'angle': [N, H, W]
                         }
                         or
                         {
                           'lcmap': [N, H, W]
                           'lcoff': [N, 2, H, W]
                           'lleng':   [N, C, H, W]
                           'angle':   [N, C, H, W]
                           'len_off': [N, C, H, W]
                           'ang_off': [N, C, H, W]
                         }
                         or
                         {
                            'anchor': [N, len_reoslu, ang_resolu, h, w]
                            'anchoff': [N, 2, len_reoslu, ang_resolu, h, w]
                            'lcoff': [N, 2, H, W]
                         }
        :param meta:   [meta0, meta1, meta2,...metaN-1]

        :param prefix:
        :return:
        """
        img = io.imread(fn)

        self.imshow(img), plt.savefig(f"{prefix}_img.jpg"), plt.close()

        self.draw_fclip(i, result, img, meta, prefix)

        # self.draw_jmap(result, target, i, prefix)
        # self.draw_lmap(result, target, i, prefix)

    def draw_fclip(self, i, result, image, meta, prefix):
        _scale, _len_scale, _ang_scale, _pscale = get_global()

        if "lleng" in result.keys():
            lcmap, lcoff, lleng, angle = result["lcmap"][i], result["lcoff"][i], result["lleng"][i], result["angle"][i]

            lines, scores = OneStageLineParsing.fclip_torch(
                lcmap, lcoff, lleng, angle,
                delta=C.model.delta, nlines=300, ang_type=C.model.ang_type, resolution=C.model.resolution
            )

        scale = 1.0 / scores[0]
        scores = scores * scale

        lpre, lpre_label = meta[i]["lpre"], meta[i]["lpre_label"]
        lpos = lpre[lpre_label == 1]

        lines = lines.cpu().numpy() * _pscale
        lpos = lpos.cpu().numpy() * _pscale
        scores = scores.cpu().numpy()

        plt.subplot(121)
        plt.imshow(image)
        self.plot_wireframe(lines, scores)

        plt.subplot(122)
        plt.imshow(image)
        self.plot_wireframe(lpos, np.ones(lpos.shape[0]))

        plt.savefig(f"{prefix}.jpg", dpi=200), plt.close()

    def plot_wireframe(self, lines, scores, isline=True, isjunc=True, alpha=None):
        for (v0, v1), s in zip(lines, scores):
            if isline:
                plt.plot([v0[1], v1[1]], [v0[0], v1[0]], linewidth=3,
                         c=self.c(s), alpha=alpha)
            if isjunc:
                plt.scatter(v0[1], v0[0], c="blue", s=16, zorder=100)
                plt.scatter(v1[1], v1[0], c="blue", s=16, zorder=100)





