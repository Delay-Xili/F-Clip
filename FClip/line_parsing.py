import torch
import numpy as np

from FClip.nms import non_maximum_suppression, structure_nms


class PointParsing():

    @staticmethod
    def jheatmap_torch(jmap, joff, delta=0.8, K=1000, kernel=3, joff_type="raw", resolution=128):
        h, w = jmap.shape
        lcmap = non_maximum_suppression(jmap[None, ...], delta, kernel).reshape(-1)
        score, index = torch.topk(lcmap, k=int(K))

        if joff is not None:
            lcoff = joff.reshape(2, -1)
            if joff_type == "raw":
                y = (index // w).float() + lcoff[0][index] + 0.5
                x = (index % w).float() + lcoff[1][index] + 0.5
            elif joff_type == "gaussian":
                y = (index // w).float() + lcoff[0][index]
                x = (index % w).float() + lcoff[1][index]
            else:
                raise NotImplementedError
        else:
            y = (index // w).float()
            x = (index % w).float()

        yx = torch.cat([y[..., None], x[..., None]], dim=-1).clamp(0, resolution - 1e-6)

        return yx, score, index

    @staticmethod
    def jheatmap_numpy(jmap, joff, delta=0.8, K=1000, kernel=3, resolution=128):

        jmap = torch.from_numpy(jmap)
        if joff is not None:
            joff = torch.from_numpy(joff)
        xy, score, index = PointParsing.jheatmap_torch(jmap, joff, delta, K, kernel, resolution=resolution)
        v = torch.cat([xy, score[:, None]], 1)
        return v.numpy()


class OneStageLineParsing():
    # @staticmethod
    # def get_resolution():
    #     return C.model.resolution

    @staticmethod
    def fclip_numpy(lcmap, lcoff, lleng, angle, delta=0.8, nlines=1000, ang_type="radian", kernel=3, resolution=128):
        lcmap = torch.from_numpy(lcmap)
        lcoff = torch.from_numpy(lcoff)
        lleng = torch.from_numpy(lleng)
        angle = torch.from_numpy(angle)

        lines, scores = OneStageLineParsing.fclip_torch(lcmap, lcoff, lleng, angle, delta, nlines, ang_type, kernel, resolution=resolution)

        return lines.numpy(), scores.numpy()

    @staticmethod
    def fclip_torch(lcmap, lcoff, lleng, angle, delta=0.8, nlines=1000, ang_type="radian", kernel=3, resolution=128):

        xy, score, index = PointParsing.jheatmap_torch(lcmap, lcoff, delta, nlines, kernel, resolution=resolution)
        lines = OneStageLineParsing.fclip_merge(xy, index, lleng, angle, ang_type, resolution=resolution)

        return lines, score

    @staticmethod
    def fclip_merge(xy, xy_idx, length_regress, angle_regress, ang_type="radian", resolution=128):
        """
        :param xy: (K, 2)
        :param xy_idx: (K,)
        :param length_regress: (H, W)
        :param angle_regress:  (H, W)
        :param ang_type
        :param resolution
        :return:
        """
        # resolution = OneStageLineParsing.get_resolution()
        xy_idx = xy_idx.reshape(-1)
        lleng_regress = length_regress.reshape(-1)[xy_idx]  # (K,)
        angle_regress = angle_regress.reshape(-1)[xy_idx]   # (K,)

        lengths = lleng_regress * (resolution / 2)
        if ang_type == "cosine":
            angles = angle_regress * 2 - 1
        elif ang_type == "radian":
            angles = torch.cos(angle_regress * np.pi)
        else:
            raise NotImplementedError
        angles1 = -torch.sqrt(1-angles**2)
        direction = torch.cat([angles1[:, None], angles[:, None]], 1)  # (K, 2)
        v1 = (xy + direction * lengths[:, None]).clamp(0, resolution)
        v2 = (xy - direction * lengths[:, None]).clamp(0, resolution)

        return torch.cat([v1[:, None], v2[:, None]], 1)


def line_parsing_from_npz(
        npz_name, ang_type="radian",
        delta=0.8, nlines=1000, kernel=3,
        s_nms=0, resolution=128
):
    # -------line parsing----
    with np.load(npz_name) as fpred:
        lcmap = fpred["lcmap"]
        lcoff = fpred["lcoff"]
        lleng = fpred["lleng"]
        angle = fpred["angle"]
        line, score = OneStageLineParsing.fclip_numpy(
            lcmap, lcoff, lleng, angle, delta, nlines, ang_type, kernel, resolution=resolution
        )

    # ---------step 2 remove line by structure nms ----
    if s_nms > 0:
        line, score = structure_nms(line, score, s_nms)


    return line, score
