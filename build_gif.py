import imageio
import os.path as osp
import os
import torch
from skimage import transform, io
from tqdm import tqdm
import numpy as np
import cv2


def build_video(images_pth, output):
    img = cv2.imread(images_pth[0])
    imgInfo = img.shape
    size = (256, int(256 * imgInfo[0] / imgInfo[1]))
    print(size)

    fps = 24
    videoWrite = cv2.VideoWriter(f'{output}/demo.mp4', -1, fps, size)# 根据图片的大小，创建写入对象 （文件名，支持的编码器，5帧，视频大小（图片大小））

    for fileName in images_pth:
        img = cv2.imread(fileName)
        img = cv2.resize(img, size)
        videoWrite.write(img)
    print('end!')


def build_gif(images_pth, output):
    img = io.imread(images_pth[0])
    size = (256, int(256 * img.shape[1] / img.shape[0]))
    gif_images = []
    for i, path in tqdm(enumerate(images_pth)):
        if i % 2 == 0:
            image = transform.resize(io.imread(path), size) * 255
            gif_images.append(image.astype(np.uint8))
    imageio.mimsave(f"{output}/demo_raw.gif", gif_images, fps=12)


if __name__ == '__main__':

    root = r"/Users/xilidai/Downloads/temp/demo_results/indoor/HR_t0.40"
    out = r"/Users/xilidai/Downloads/temp/demo_results/indoor"

    print(f"gif ing {root}")
    img_paths = sorted(os.listdir(f"{root}"))
    image_pth = [f"{root}/{pth}" for pth in img_paths]

    build_gif(image_pth[1:], out)
    # build_video(image_pth[1:], out)
