#!/usr/bin/env python
"""Process Huang's wireframe dataset for L-CNN network
Usage:
    dataset/wireframe.py <src> <dst>
    dataset/wireframe.py (-h | --help )

Examples:
    python dataset/wireframe.py /datadir/wireframe Good/wireframe
    python dataset/wireframe_line.py /home/dxl/Data/wireframe_raw

Arguments:
    <src>                Original Good directory of Huang's wireframe dataset
    <dst>                Directory of the output

Options:
   -h --help             Show this screen.
"""

import os
import sys
import json
from itertools import combinations

import cv2
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from docopt import docopt
from scipy.ndimage import zoom

try:
    sys.path.append(".")
    sys.path.append("..")
    from FClip.utils import parmap
except Exception:
    raise


def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))


def save_heatmap(prefix, image, lines):
    im_rescale = (512, 512)
    heatmap_scale = (128, 128)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]

    lcmap = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)
    lcoff = np.zeros((2,) + heatmap_scale, dtype=np.float32)  # (2, 128, 128)
    lleng = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)
    angle = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)

    # the coordinate of lines can not equal to 128 (less than 128).
    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    lines = lines[:, :, ::-1]  # change position of x and y --> (r, c)

    for v0, v1 in lines:
        v = (v0 + v1) / 2
        vint = to_int(v)
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

        # the junction coordinate(image coordinate system) of line can be recovered by follows:
        # direction = [-sqrt(1-theta^2), theta]
        # (-sqrt(1-theta^2) means the component along the r direction on the unit vector, it always negative.)
        # center = coordinate(lcmap) + offset + 0.5
        # J = center (+-) direction * lleng  (+-) means two end points

    image = cv2.resize(image, im_rescale)

    # plt.figure()
    # plt.imshow(image)
    # for v0, v1 in lines:
    #     plt.plot([v0[1] * 4, v1[1] * 4], [v0[0] * 4, v1[0] * 4])
    # plt.savefig(f"dataset/{os.path.basename(prefix)}_line.png", dpi=200), plt.close()
    # return

    # coor = np.argwhere(lcmap == 1)
    # for yx in coor:
    #     offset = lcoff[:, int(yx[0]), int(yx[1])]
    #     length = lleng[int(yx[0]), int(yx[1])]
    #     theta = angle[int(yx[0]), int(yx[1])]
    #
    #     center = yx + offset
    #     d = np.array([-np.sqrt(1-theta**2), theta])
    #     plt.scatter(center[1]*4, center[0]*4, c="b")
    #
    #     plt.arrow(center[1]*4, center[0]*4, d[1]*length*4, d[0]*length*4,
    #               length_includes_head=True,
    #               head_width=15, head_length=25, fc='r', ec='b')

    # plt.savefig(f"{prefix}_line.png", dpi=200), plt.close()

    # plt.subplot(122), \
    # plt.imshow(image)
    # coor = np.argwhere(lcmap == 1)
    # for yx in coor:
    #     offset = lcoff[:, int(yx[0]), int(yx[1])]
    #     length = lleng[int(yx[0]), int(yx[1])]
    #     theta = angle[int(yx[0]), int(yx[1])]
    #
    #     center = yx + offset
    #     d = np.array([-np.sqrt(1-theta**2), theta])
    #
    #     n0 = center + d * length
    #     n1 = center - d * length
    #     plt.plot([n0[1] * 4, n1[1] * 4], [n0[0] * 4, n1[0] * 4])
    # plt.savefig(f"{prefix}_line.png", dpi=100), plt.close()

    np.savez_compressed(
        f"{prefix}_line.npz",
        # aspect_ratio=image.shape[1] / image.shape[0],
        lcmap=lcmap,
        lcoff=lcoff,
        lleng=lleng,
        angle=angle,
    )
    cv2.imwrite(f"{prefix}.png", image)


def coor_rot90(coordinates, center, k):

    # !!!rotate the coordinates 90 degree anticlockwise on image!!!!

    # (x, y) --> (p-q+y, p+q-x) means point (x,y) rotate 90 degree clockwise along center (p,q)
    # but, the y direction of coordinates is inverse, not up but down.
    # so it equals to rotate the coordinate anticlockwise.

    # coordinares: [n, 2]; center: (p, q) rotation center.
    # coordinates and center should follow the (x, y) order, not (h, w).
    new_coor = coordinates.copy()
    p, q = center
    for i in range(k):
        x = p - q + new_coor[:, 1:2]
        y = p + q - new_coor[:, 0:1]
        new_coor = np.concatenate([x, y], 1)
    return new_coor


def prepare_rotation(image, lines):
    heatmap_scale = (512, 512)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]

    # the coordinate of lines can not equal to 128 (less than 128).
    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)

    im = cv2.resize(image, heatmap_scale)

    return im, lines


def main():
    args = docopt(__doc__)
    data_root = args["<src>"]
    data_output = args["<dst>"]

    os.makedirs(data_output, exist_ok=True)
    for batch in ["train", "valid"]:  # "train", "valid"
        anno_file = os.path.join(data_root, f"{batch}.json")

        with open(anno_file, "r") as f:
            dataset = json.load(f)

        def handle(data):
            im = cv2.imread(os.path.join(data_root, "images", data["filename"]))
            prefix = data["filename"].split(".")[0]
            lines = np.array(data["lines"]).reshape(-1, 2, 2)
            os.makedirs(os.path.join(data_output, batch), exist_ok=True)
            path = os.path.join(data_output, batch, prefix)

            # lines0 = lines.copy()
            # save_heatmap(f"{path}_0", im[::, ::], lines0)

            if batch != "valid":
                # lines1 = lines.copy()
                # lines1[:, :, 0] = im.shape[1] - lines1[:, :, 0]
                # im1 = im[::, ::-1]
                # save_heatmap(f"{path}_1", im1, lines1)

                # lines2 = lines.copy()
                # lines2[:, :, 1] = im.shape[0] - lines2[:, :, 1]
                # im2 = im[::-1, ::]
                # save_heatmap(f"{path}_2", im2, lines2)
                #
                # lines3 = lines.copy()
                # lines3[:, :, 0] = im.shape[1] - lines3[:, :, 0]
                # lines3[:, :, 1] = im.shape[0] - lines3[:, :, 1]
                # im3 = im[::-1, ::-1]
                # save_heatmap(f"{path}_3", im3, lines3)
                #
                # im4, lines4 = prepare_rotation(im, lines.copy())
                # lines4 = coor_rot90(lines4.reshape((-1, 2)), (im4.shape[1] / 2, im4.shape[0] / 2), 1)  # rot90 on anticlockwise
                # im4 = np.rot90(im4, k=1)  # rot90 on anticlockwise
                # save_heatmap(f"{path}_4", im4, lines4.reshape((-1, 2, 2)))
                #
                # im5, lines5 = prepare_rotation(im, lines.copy())
                # lines5 = coor_rot90(lines5.reshape((-1, 2)), (im5.shape[1] / 2, im5.shape[0] / 2), 3)  # rot90 on clockwise
                # im5 = np.rot90(im5, k=-1)  # rot90 on clockwise
                # save_heatmap(f"{path}_5", im5, lines5.reshape((-1, 2, 2)))

                linesf = lines.copy()
                linesf[:, :, 0] = im.shape[1] - linesf[:, :, 0]
                im1 = im[::, ::-1]
                im6, lines6 = prepare_rotation(im1, linesf.copy())
                lines6 = coor_rot90(lines6.reshape((-1, 2)), (im6.shape[1] / 2, im6.shape[0] / 2), 1)  # rot90 on anticlockwise
                im6 = np.rot90(im6, k=1)  # rot90 on anticlockwise
                save_heatmap(f"{path}_6", im6, lines6.reshape((-1, 2, 2)))

                im7, lines7 = prepare_rotation(im1, linesf.copy())
                lines7 = coor_rot90(lines7.reshape((-1, 2)), (im7.shape[1] / 2, im7.shape[0] / 2), 3)  # rot90 on clockwise
                im7 = np.rot90(im7, k=-1)  # rot90 on clockwise
                save_heatmap(f"{path}_7", im7, lines7.reshape((-1, 2, 2)))

                # exit()

            print("Finishing", os.path.join(data_output, batch, prefix))

        # handle(dataset[0])
        # multiprocessing the function of handle with augment 'dataset'.
        parmap(handle, dataset, 16)


if __name__ == "__main__":

    main()

