import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))


def save_heatmap(prefix, image, lines):
    im_rescale = (512, 512)
    heatmap_scale = (128, 128)

    # fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]

    lcmap = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)
    lcoff = np.zeros((2,) + heatmap_scale, dtype=np.float32)  # (2, 128, 128)
    lleng = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)
    angle = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)

    # the coordinate of lines can not equal to 128 (less than 128).
    # lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    # lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    # lines = lines[:, :, ::-1]  # change position of x and y --> (r, c)

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

    # image = cv2.resize(image, im_rescale)

    # plt.figure()
    # plt.imshow(image)
    # for v0, v1 in lines:
    #     plt.plot([v0[1] * 4, v1[1] * 4], [v0[0] * 4, v1[0] * 4])
    # plt.savefig(f"{prefix[-8:]}_line_gt.png", dpi=200), plt.close()
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
    # plt.savefig(f"{prefix[-8:]}_line.png", dpi=100), plt.close()

    np.savez_compressed(
        f"{prefix}_line.npz",
        # aspect_ratio=image.shape[1] / image.shape[0],
        lcmap=lcmap,
        lcoff=lcoff,
        lleng=lleng,
        angle=angle,
    )
    # cv2.imwrite(f"{prefix}.png", image)


if __name__ == '__main__':

    root = "/home/dxl/Data/york/valid/"

    filelist = glob.glob(f"{root}/*_label.npz")

    for file in filelist:
        with np.load(file) as npz:
            lines = npz["lpos"][:, :, :2]
        # image = cv2.imread(file.replace("_label.npz", ".png"))
        image = None
        save_heatmap(file[:-10], image, lines)
        print(file)

