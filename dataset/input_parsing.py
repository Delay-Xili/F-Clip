import numpy as np


def offset_wrapper(lcoff, lcmap, threshold=3):
    """translate the offset to gaussian mode"""

    _lcoff = lcoff.copy()

    lcidx = np.argwhere(lcmap == 1)

    u, v = np.meshgrid(np.arange(len(lcoff)), np.arange(len(lcoff)))
    u, v = u.reshape(-1), v.reshape(-1)
    uv = np.concatenate([u[:, None], v[:, None]], 1)

    dist = np.sqrt(np.sum((uv[:, None] - lcidx[None]) ** 2, -1))

    val, ddx = np.min(dist, 1), np.argmin(dist, 1)

    for k, xy in enumerate(uv):
        if val[k] <= threshold:

            ixy = lcidx[ddx[k]]
            _lcoff[:, xy[0], xy[1]] = lcoff[:, ixy[0], ixy[1]] + ixy - xy + 0.5  # xy + _lcoff[xy] = ixy + lcoff[ixy]

    return _lcoff


class WireframeHuangKun():

    @staticmethod
    def la2drdc(lcmap, lleng, angle, ang_type):

        if ang_type == "radian":
            angle_ = lcmap * np.cos(angle * np.pi)
        elif ang_type == "cosine":
            angle_ = lcmap * (angle * 2 - 1)
        else:
            raise ValueError()

        dc = lleng * angle_
        dr = lleng * (-np.sqrt(1-angle_**2))

        return dr, dc

    @staticmethod
    def fclip_parsing(npz_name, ang_type="radian"):
        with np.load(npz_name) as npz:
            lcmap_ = npz["lcmap"]
            lcoff_ = npz["lcoff"]
            lleng_ = np.clip(npz["lleng"], 0, 64 - 1e-4)
            angle_ = np.clip(npz["angle"], -1 + 1e-4, 1 - 1e-4)

            lcmap = lcmap_
            lcoff = lcoff_
            lleng = lleng_ / 64
            if ang_type == "cosine":
                angle = (angle_ + 1) * lcmap_ / 2
            elif ang_type == "radian":
                angle = lcmap * np.arccos(angle_) / np.pi
            else:
                raise NotImplementedError

            return lcmap, lcoff, lleng, angle
