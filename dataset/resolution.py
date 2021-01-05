import numpy as np
import numpy.linalg as LA
import skimage
import cv2


class ResizeResolution():

    @staticmethod
    def resize(lpos, image, resolu, ang_type="radian"):
        """resize the heatmap"""

        if resolu < 128:
            image_ = cv2.resize(image, (resolu * 4, resolu * 4))
        elif resolu == 128:
            image_ = image
        else:
            raise ValueError("not support!")

        lcmap, lcoff, lleng, angle = ResizeResolution.resolution_fclip(lpos, resolu, ang_type)

        return image_, lcmap, lcoff, lleng, angle

    @staticmethod
    def resolution_fclip(lpos, resolu, ang_type="radian"):
        heatmap_scale = (resolu, resolu)
        scale = resolu / 128

        lines = lpos * scale

        lcmap = np.zeros(heatmap_scale, dtype=np.float32)  # (resolu, resolu)
        lcoff = np.zeros((2,) + heatmap_scale, dtype=np.float32)  # (2, resolu, resolu)
        lleng = np.zeros(heatmap_scale, dtype=np.float32)  # (resolu, resolu)
        angle = np.zeros(heatmap_scale, dtype=np.float32)  # (resolu, resolu)

        lines[:, :, 0] = np.clip(lines[:, :, 0], 0, heatmap_scale[0] - 1e-4)
        lines[:, :, 1] = np.clip(lines[:, :, 1], 0, heatmap_scale[1] - 1e-4)

        for v0, v1 in lines:
            v = (v0 + v1) / 2
            vint = tuple(map(int, v))
            lcmap[vint] = 1
            lcoff[:, vint[0], vint[1]] = v - vint - 0.5
            lleng[vint] = np.sqrt(np.sum((v0 - v1) ** 2)) / 2  # L

            if v0[0] <= v[0]:
                vv = v0
            else:
                vv = v1
            # the angle under the image coordinate system (r, c)
            # theta means the component along the c direction on the unit vector
            if np.sqrt(np.sum((vv - v) ** 2)) <= 1e-4:
                continue
            angle[vint] = np.sum((vv - v) * np.array([0., 1.])) / np.sqrt(np.sum((vv - v) ** 2))  # theta

        lleng_ = np.clip(lleng, 0, 64 * scale - 1e-4) / (64 * scale)

        if ang_type == "cosine":
            angle = (angle + 1) * lcmap / 2
        elif ang_type == "radian":
            angle = lcmap * np.arccos(angle) / np.pi
        else:
            raise NotImplementedError

        return lcmap, lcoff, lleng_, angle

    @staticmethod
    def resolution_lcnn(npz_dic, resolution, dataset="yichao"):
        heatmap_scale = (resolution, resolution)
        scale = resolution / 128
        # npz["jmap"]: [J, H, W]    Junction heat map
        # npz["joff"]: [J, 2, H, W] Junction offset within each pixel
        # npz["lmap"]: [H, W]       Line heat map with anti-aliasing
        # npz["junc"]: [Na, 3]      Junction coordinates
        # npz["Lpos"]: [M, 2]       Positive lines represented with junction indices
        # npz["Lneg"]: [M, 2]       Negative lines represented with junction indices
        # npz["lpos"]: [Np, 2, 3]   Positive lines represented with junction coordinates
        # npz["lneg"]: [Nn, 2, 3]   Negative lines represented with junction coordinates

        def to_int(x):
            return tuple(map(int, x))

        npz = {}
        npz["junc"] = npz_dic["junc"]
        npz["lpos"] = npz_dic["lpos"]
        npz["lneg"] = npz_dic["lneg"]
        npz["Lpos"] = npz_dic["Lpos"]
        npz["Lneg"] = npz_dic["Lneg"]

        npz["junc"][:, :2] = npz["junc"][:, :2] * scale
        npz["lpos"][:, :, :2] = npz["lpos"][:, :, :2] * scale
        npz["lneg"][:, :, :2] = npz["lneg"][:, :, :2] * scale
        # print(npz["lneg"])

        J = 1 if dataset == "huangkun" else 2

        jmap = np.zeros((J,) + heatmap_scale, dtype=np.float32)  # (2, 256, 256)
        joff = np.zeros((J, 2) + heatmap_scale, dtype=np.float32)  # (2, 2, 256, 256)
        lmap = np.zeros(heatmap_scale, dtype=np.float32)  # (256, 256)

        for node in npz["junc"]:
            vint = to_int(node[:2])
            jmap[int(node[2])][vint] = 1
            if resolution <= 256:
                off = node[:2] - vint - 0.5
                if LA.norm(joff[int(node[2]), :, vint[0], vint[1]]) == 0 or LA.norm(
                        joff[int(node[2]), :, vint[0], vint[1]]) > LA.norm(off):
                    joff[int(node[2]), :, vint[0], vint[1]] = off
                # joff[int(node[2]), :, vint[0], vint[1]] = node[:2] - vint - 0.5

        npz["jmap"] = jmap
        npz["joff"] = joff

        for [v0, v1] in npz["lpos"]:
            rr, cc, value = skimage.draw.line_aa(*to_int(v0[:2]), *to_int(v1[:2]))
            lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

        npz["lmap"] = lmap

        return npz
