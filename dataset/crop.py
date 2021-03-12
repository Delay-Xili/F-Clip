import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from skimage import io
import cv2
import warnings


def randomCrop(s, resolution):
    th, tw = int(resolution * s), int(resolution * s)
    i, j = random.randint(0, resolution - th), random.randint(0, resolution - tw)
    # x1, y1 = j, i
    # x2, y2 = j + tw, i + th
    return i, j, th, tw


class CropAugmentation():

    @staticmethod
    def random_crop_augmentation(image, lines, s, resolution=128):
        """resolution should be the size of the label in npz file,
        'NOT' the augmentation resolution of the input."""
        _pscale = 512 / resolution

        if 0.2 <= s < 0.5:
            warnings.warn("Maybe got a all zeros lcmap, crop a region of no line!")
        elif s < 0.2:
            raise ValueError("risk! high probability to crop a region of no line!")
        else:
            pass

        if s == 1:
            #
            x1, x2, y1, y2 = 0, 128, 0, 128
            iscrop = False
            image_ = image
        elif s < 1.:
            # crop
            i, j, th, tw = randomCrop(s, resolution)
            x1, y1 = j, i
            x2, y2 = j + tw, i + th
            iscrop = True
            crop_image = image[int(y1 * _pscale):int(y2 * _pscale), int(x1 * _pscale):int(x2 * _pscale), :]
            image_ = cv2.resize(crop_image, (512, 512))
        elif s > 1.:
            # shrink
            s_ = 1. / s
            i, j, th, tw = randomCrop(s_, resolution)
            x1, y1 = j, i
            x2, y2 = j + tw, i + th

            image_ = np.zeros_like(image)
            image_[int(y1 * _pscale):int(y2 * _pscale), int(x1 * _pscale):int(x2 * _pscale), :] = \
                cv2.resize(image, (int(th * _pscale), int(tw * _pscale)))

            iscrop = False
            lines = lines * s_
            lines[:, :, 0], lines[:, :, 1] = lines[:, :, 0] + i, lines[:, :, 1] + j
        else:
            raise ValueError("")

        lcmap, lcoff, lleng, angle, cropped_lines = CropAugmentation.line_crop_and_heatmap(lines, x1, x2, y1, y2, iscrop)

        if lcmap is None and s < 1.:
            x1, x2, y1, y2 = 0, 128, 0, 128
            iscrop = False
            image_ = image
            lcmap, lcoff, lleng, angle, cropped_lines = CropAugmentation.line_crop_and_heatmap(lines, x1, x2, y1, y2, iscrop)

        return image_, lcmap, lcoff, lleng, angle, cropped_lines, (x1, x2, y1, y2)

    @staticmethod
    def viz_crop(image, crop_region, lines, cropped_lines, prefix, isShrink):
        resolu = 128
        _pscale = 512 / resolu
        x1, x2, y1, y2 = crop_region
        plt.subplot(121)
        plt.imshow(image)
        for v0, v1 in lines:
            v0, v1 = _pscale * v0, _pscale * v1
            plt.plot([v0[1], v1[1]], [v0[0], v1[0]], linewidth=3, c="red")
            plt.scatter(v0[1], v0[0], c="blue", s=64, zorder=100)
            plt.scatter(v1[1], v1[0], c="blue", s=64, zorder=100)

        plt.subplot(122)
        plt.imshow(image)
        # if isShrink:
        #     lpos = cropped_lines.copy()
        # else:
        #     lpos = np.concatenate([cropped_lines[:, :, 0:1] * (y2 - y1) / resolu + y1,
        #                            cropped_lines[:, :, 1:2] * (x2 - x1) / resolu + x1], -1)
        #
        # for v0, v1 in lpos:
        #     v0, v1 = _pscale * v0, _pscale * v1
        #     plt.plot([v0[1], v1[1]], [v0[0], v1[0]], linewidth=3, c="red")
        #     plt.scatter(v0[1], v0[0], c="blue", s=64, zorder=100)
        #     plt.scatter(v1[1], v1[0], c="blue", s=64, zorder=100)

        x1, x2, y1, y2 = _pscale * x1, _pscale * x2, _pscale * y1, _pscale * y2
        plt.plot([x1, x1], [y1, y2], linewidth=3, c="yellow")
        plt.plot([x1, x2], [y1, y1], linewidth=3, c="yellow")
        plt.plot([x1, x2], [y2, y2], linewidth=3, c="yellow")
        plt.plot([x2, x2], [y1, y2], linewidth=3, c="yellow")

        # ids = np.random.randint(0, 100)
        plt.savefig(f"{prefix}_{x1}_{x2}_{y1}_{y2}.jpg", dpi=100), plt.close()
        print(f"{prefix}_{x1}_{x2}_{y1}_{y2}.jpg")
        # exit()

    @staticmethod
    def line_crop_and_heatmap(lines, x1, x2, y1, y2, iscrop=True, ang_type="radian"):

        heatmap_scale = (128, 128)
        if iscrop:
            lines = CropAugmentation.line_crop(lines, x1, x2, y1, y2)
        if len(lines) == 0:

            return None, None, None, None, None

        lcmap = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)
        lcoff = np.zeros((2,) + heatmap_scale, dtype=np.float32)  # (2, 128, 128)
        lleng = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)
        angle = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)

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

        lleng_ = np.clip(lleng, 0, 64 - 1e-4) / 64
        # angle_ = np.clip(angle, -1 + 1e-4, 1 - 1e-4) / np.pi
        if ang_type == "cosine":
            angle_ = (angle + 1) * lcmap / 2
        elif ang_type == "radian":
            angle_ = lcmap * np.arccos(angle) / np.pi
        else:
            raise NotImplementedError

        return lcmap, lcoff, lleng_, angle_, lines

    @staticmethod
    def line_crop(lines, x1, x2, y1, y2, resolu=128):

        # with np.load(label_name) as label:
        #     lpos = label["lpos"]

        lines = torch.from_numpy(lines)

        # it will cause bug wihtout the 1e-8 on numerator and denominator.
        # TODO
        k = ((lines[:, 0, 0] - lines[:, 1, 0]) + 1e-8) / ((lines[:, 0, 1] - lines[:, 1, 1]) + 1e-8)  # (y1-y2) / (x1-x2)
        b = lines[:, 0, 0] - k * lines[:, 0, 1]
        kb = torch.cat([k[:, None], b[:, None]], 1)

        n1y, n1x = kb[:, 0:1] * x1 + kb[:, 1:2], torch.ones_like(kb[:, 0:1]) * x1
        n1 = torch.cat([n1y, n1x], 1)
        n1_s = (n1y <= y2) & (n1y >= y1) & (n1x <= x2) & (n1x >= x1)

        n2y, n2x = kb[:, 0:1] * x2 + kb[:, 1:2], torch.ones_like(kb[:, 0:1]) * x2
        n2 = torch.cat([n2y, n2x], 1)
        n2_s = (n2y <= y2) & (n2y >= y1) & (n2x <= x2) & (n2x >= x1)

        n3y, n3x = torch.ones_like(kb[:, 0:1]) * y1, (y1 - kb[:, 1:2]) / kb[:, 0:1]
        n3 = torch.cat([n3y, n3x], 1)
        n3_s = (n3y <= y2) & (n3y >= y1) & (n3x <= x2) & (n3x >= x1)

        n4y, n4x = torch.ones_like(kb[:, 0:1]) * y2, (y2 - kb[:, 1:2]) / kb[:, 0:1]
        n4 = torch.cat([n4y, n4x], 1)
        n4_s = (n4y <= y2) & (n4y >= y1) & (n4x <= x2) & (n4x >= x1)

        nodes = torch.cat([n1[:, None], n2[:, None], n3[:, None], n4[:, None]], 1)  # (N, 4, 2)
        n_s = torch.cat([n1_s, n2_s, n3_s, n4_s], 1)  # (N, 4)

        ly1, lx1, ly2, lx2 = lines[:, 0, 0], lines[:, 0, 1], lines[:, 1, 0], lines[:, 1, 1]
        ln1_s = (lx1 < x2) & (lx1 > x1) & (ly1 < y2) & (ly1 > y1)  # satisfied node 1
        ln2_s = (lx2 < x2) & (lx2 > x1) & (ly2 < y2) & (ly2 > y1)  # satisfied node 2
        ln_s = torch.cat([ln1_s[:, None], ln2_s[:, None]], 1)  # (N, 2)

        l_s = ln_s.long().sum(1)

        idx2 = l_s == 2  # the idx of line which's two endpoints all in crop region
        idx1 = l_s == 1  # the idx of line which's one endpoint all in crop region
        idx0 = l_s == 0  # the idx of line which's zero endpoint all in crop region

        if len(idx2) > 0:
            lines2 = lines[idx2]
            lines2 = torch.cat([(lines2[:, :, 0:1] - y1) * resolu / (y2 - y1),
                                     (lines2[:, :, 1:2] - x1) * resolu / (x2 - x1)], -1)  # (k, 2, 2)
        else:
            lines2 = torch.zeros((0, 2, 2))

        if len(idx1) > 0:
            # step1
            # get the vector from inner endpoint to outer endpoint (in "direction")
            sat_lines = lines[idx1]  # (k, 2, 2)
            sat_idx = ln_s[idx1]  # (k, 2)
            unsat_idx = ~sat_idx
            sat_line_node = sat_lines.reshape(-1, 2)[sat_idx.reshape(-1)]  # (k, 2)
            unsat_line_node = sat_lines.reshape(-1, 2)[unsat_idx.reshape(-1)]  # (k, 2)
            direction = unsat_line_node - sat_line_node

            # step2
            # get the boundary point for each satisfied lines (nodes1 and n_s1),
            # if the boundary points not satisfied, drop the corresponding line and its boundary points.
            nodes1 = nodes[idx1]  # (k, 4, 2)
            n_s1 = n_s[idx1]  # (k, 4)
            good_idx = n_s1.sum(-1) == 2
            if nodes1.shape[0] != good_idx.sum():
                sat_line_node = sat_line_node[good_idx]
                unsat_line_node = unsat_line_node[good_idx]
                direction = direction[good_idx]
                nodes1 = nodes1[good_idx]
                n_s1 = n_s1[good_idx]
            sat_nodes = nodes1.reshape(-1, 2)[n_s1.reshape(-1)]
            sat_nodes = sat_nodes.reshape(-1, 2, 2)

            # step3
            # select the boundary point using inner product
            sat_direction = sat_nodes - sat_line_node[:, None]
            dot_product = (direction[:, None] * sat_direction).sum(-1)  # (K, 2)
            node_idx = dot_product > 0
            good_idx = node_idx.sum(-1) == 1
            if node_idx.shape[0] != good_idx.sum():
                direction = direction[good_idx]
                sat_direction = sat_direction[good_idx]
                dot_product = dot_product[good_idx]
                node_idx = node_idx[good_idx]
                sat_nodes = sat_nodes[good_idx]
                sat_line_node = sat_line_node[good_idx]

            # step4 merge, sift and scale
            nodes_s = sat_nodes.reshape(-1, 2)[node_idx.reshape(-1)]  # (K, 2)
            lines1 = torch.cat([sat_line_node[:, None], nodes_s[:, None]], 1)  # (K, 2, 2)
            lines1 = torch.cat([(lines1[:, :, 0:1] - y1) * resolu / (y2 - y1),
                                (lines1[:, :, 1:2] - x1) * resolu / (x2 - x1)], -1)  # (k, 2, 2)

        else:
            lines1 = torch.zeros((0, 2, 2))

        if len(idx0) > 0:

            lines00 = lines[idx0]
            node0 = nodes[idx0]  # (K, 4, 2)
            n_s0 = n_s[idx0]     # (K, 4)

            n_idx = n_s0.sum(-1) == 2

            nodes_s = node0[n_idx]  # (k, 4, 2)
            n_ss = n_s0[n_idx]  # (k, 4)
            lines00 = lines00[n_idx]

            lines0 = nodes_s.reshape(-1, 2)[n_ss.reshape(-1)]
            lines0 = lines0.reshape(-1, 2, 2)
            assert nodes_s.shape[0] == lines0.shape[0]

            index = []
            for i in range(len(lines0)):
                v0, v1 = lines0[i]  # clip points
                v00, v11 = lines00[i]  # endpoints

                if v00[1] <= v11[1]:
                    x1_, x2_ = v00[1], v11[1]
                else:
                    x1_, x2_ = v11[1], v00[1]

                if v00[0] <= v11[0]:
                    y1_, y2_ = v00[0], v11[0]
                else:
                    y1_, y2_ = v11[0], v00[0]

                if x1_ < v0[1] < x2_ and y1_ < v0[0] < y2_:
                    index.append(True)
                else:
                    index.append(False)
            index = torch.from_numpy(np.array(index))
            if index.sum() == 0:
                lines0 = torch.zeros((0, 2, 2))
            else:
                lines0 = lines0[index]
                lines0 = torch.cat([(lines0[:, :, 0:1] - y1) * resolu / (y2 - y1),
                                    (lines0[:, :, 1:2] - x1) * resolu / (x2 - x1)], -1)

        else:
            lines0 = torch.zeros((0, 2, 2))
        # lines0 = torch.zeros((0, 2, 2))

        new_lines = torch.cat([lines0, lines1, lines2], 0)

        return new_lines.numpy()
