#!/usr/bin/env python
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#  Revision author: Siyu Huang
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import argparse
import glob
import numpy as np
import os
import time
import cv2 as cv
import torch
os.environ['KMP_DUPLICATE_LIB_OK']='True'

threshold = 0.4


myjet = np.array([[0., 0., 0.5],
                  [0., 0., 0.99910873],
                  [0., 0.37843137, 1.],
                  [0., 0.83333333, 1.],
                  [0.30044276, 1., 0.66729918],
                  [0.66729918, 1., 0.30044276],
                  [1., 0.90123457, 0.],
                  [1., 0.48002905, 0.],
                  [0.99910873, 0.07334786, 0.],
                  [0.5, 0., 0.]])


class VideoStreamer(object):
    """ Class to help process image streams. Three types of possible inputs:"
      1.) USB Webcam.
      2.) A directory of images (files in directory matching 'img_glob').
      3.) A video file, such as an .mp4 or .avi file.
    """

    def __init__(self, basedir, camid, skip, img_glob):
        self.cap = []
        self.camera = False
        self.video_file = False
        self.listing = []
        self.i = 0
        self.skip = skip
        self.needsort = False
        # If the "basedir" string is the word camera, then use a webcam.
        if basedir == "camera/" or basedir == "camera":
            print('==> Processing Webcam Input.')
            self.cap = cv.VideoCapture(camid)
            self.listing = range(0, self.maxlen)
            self.camera = True
        else:
            # Try to open as a video.
            self.cap = cv.VideoCapture(basedir)
            lastbit = basedir[-4:len(basedir)]
            if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
                raise IOError('Cannot open movie file')
            elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
                print('==> Processing Video Input.')
                num_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
                self.listing = range(0, num_frames)
                self.listing = self.listing[::self.skip]
                self.camera = True
                self.video_file = True
                self.maxlen = len(self.listing)
            else:
                print('==> Processing Image Directory Input.')
                minname_len = 1000000
                maxname_len = 0
                self.index = []
                search = os.path.join(basedir, img_glob)
                self.listing = glob.glob(search)
                for imname in self.listing:
                    name = imname.split('/')[-1]
                    if len(name) > maxname_len:
                        maxname_len = len(name)
                    if (len(name)) < minname_len:
                        minname_len = len(name)
                if (minname_len) != maxname_len:
                    for imname in self.listing:
                        name = imname.split('/')[-1]
                        name = name.rjust(maxname_len, '0')
                        self.index.append(name)
                    self.needsort = True
                else:
                    self.index = self.listing
                self.ordername = np.argsort(self.index)
                self.maxlen = len(self.ordername)
                if self.maxlen == 0:
                    raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

    def read_image(self, index):
        """ Read image as grayscale and resize to img_size.
        Inputs
          impath: Path to input image.
          img_size: (W, H) tuple specifying resize size.
        Returns
          grayim: float32 numpy array sized H x W with values in range [0, 1].
        """
        if self.needsort:
            impath = self.listing[self.ordername[index]]
        else:
            impath = self.listing[index]

        image = cv.imread(impath)
        grayim = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        # Image is resized via opencv.
        # interp = cv.INTER_AREA
        return grayim, image

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
           image: Next H x W image.
           status: True or False depending whether image was loaded.
        """
        if self.i == self.maxlen:
            return (None, None, False)
        if self.camera:
            ret, image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
                return (None, None, False)
            if self.video_file:
                self.cap.set(cv.CAP_PROP_POS_FRAMES, self.listing[self.i])
            # input_image = cv.resize(image, (self.sizer[1], self.sizer[0]),
            #                         interpolation=cv.INTER_AREA)
            input_image = image
            input_image = cv.cvtColor(input_image, cv.COLOR_RGB2GRAY)
        else:
            # image_file = self.listing[self.i]
            input_image, image = self.read_image(self.i)
        # Increment internal counter.
        self.i = self.i + 1
        return (input_image, image, True)


class FClipDetect:
    def __init__(self, modeluse, ckpt=None):
        from test import build_model, C, M

        if modeluse == 'HG1_D3':
            config_file = 'config/fclip_HG1_D3.yaml'
        elif modeluse == 'HG1':
            config_file = 'config/fclip_HG1.yaml'
        elif modeluse == 'HG2':
            config_file = 'config/fclip_HG2.yaml'
        elif modeluse == 'HG2_LB':
            config_file = 'config/fclip_HG2_LB.yaml'
        elif modeluse == 'HR':
            config_file = 'config/fclip_HR.yaml'
        else:
            raise ValueError("")

        C.update(C.from_yaml(filename='config/base.yaml'))
        C.update(C.from_yaml(filename=config_file))
        M.update(C.model)
        C.io.model_initialize_file = ckpt

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        iscpu = False if torch.cuda.is_available() else True
        print('Using device: ', self.device)

        self.model = build_model(cpu=iscpu)
        self.model.to(self.device)
        self.input_resolution = (M.resolution * 4, M.resolution * 4)
        self.image_mean = M.image.mean
        self.image_stddev = M.image.stddev

    def detect(self, img):
        H_img, W_img = img.shape[:2]
        inp = cv.resize(img, self.input_resolution)
        inp = inp[:, :, ::-1]  # convert BGR to RGB
        H, W, C = inp.shape

        inp = (inp.astype(np.float32) - self.image_mean) / self.image_stddev
        inp = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0).to(device=self.device)
        input_dict = {"image": inp}
        with torch.no_grad():
            outputs = self.model(input_dict, isTest=True)
            lines = outputs["heatmaps"]["lines"][0] * 4
            score = outputs["heatmaps"]["score"][0]
            lines = lines[score > threshold]

        lines[:, :, 0] = lines[:, :, 0] * H_img / H
        lines[:, :, 1] = lines[:, :, 1] * W_img / W
        if torch.cuda.is_available():
            return lines.cpu().numpy()
        else:
            return lines.numpy()


if __name__ == '__main__':

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Line Demo.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory or movie file or "camera" (for webcam).')
    parser.add_argument('--model', type=str, default='HR',
                        help='choose the pretrained model (option: HG1, HG2, HR).')
    parser.add_argument('--output_dir', type=str, default='logs/demo_results',
                        help='output directory.')
    parser.add_argument('--ckpt', type=str, default='ckpt',
                        help='directory to checkpoint file.')
    parser.add_argument('--camid', type=int, default=0,
                        help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
    parser.add_argument('--img_glob', type=str, default='*.png',
                        help='Glob match if directory of images is specified (default: \'*.png\').')
    parser.add_argument('--skip', type=int, default=1,
                        help='Images to skip if input is movie or directory (default: 1).')
    parser.add_argument('--waitkey', type=int, default=1,
                        help='OpenCV waitkey time in ms (default: 1).')
    parser.add_argument('--display', type=bool, default=False,
                        help='Whether to create a window to display the demo or not.')
    opt = parser.parse_args()
    print(opt)
    print(opt.output_dir)
    os.makedirs(opt.output_dir, exist_ok=True)

    print('==> Loading video.')
    # This class helps load input images from different sources.
    vs = VideoStreamer(opt.input, opt.camid, opt.skip, opt.img_glob)
    print('==> Successfully loaded video.')

    detector = FClipDetect(opt.model, opt.ckpt)

    # Create a window to display the demo.
    if opt.display:
        win = 'Line Tracker'
        cv.namedWindow(win)

    print('==> Running Demo.')
    t_begin = time.time()
    frame = 0
    while True:

        start = time.time()
        img, oriimg, status = vs.next_frame()  # gray
        print("\r", end="")
        print(f"Processing: {vs.i}", end="")
        if status is False:
            break

        # Get points and descriptors.
        start1 = time.time()
        lines = detector.detect(oriimg)
        end1 = time.time()

        out = oriimg
        for i in range(lines.shape[0]):
            # print(lines[i])
            start_coor = (int(lines[i][0][1]), int(lines[i][0][0]))
            end_coor = (int(lines[i][1][1]), int(lines[i][1][0]))
            cv.line(out, start_coor, end_coor, (0, 0, 255), 2, lineType=16)  # red
            # cv.line(out, lines[i, 0, ::-1], lines[i, 1, ::-1], (110, 215, 245), 2, lineType=16)

        cv.imwrite(f"{opt.output_dir}/{vs.i:04}.png", out)

        # Display visualization image to screen.
        if opt.display:
            cv.imshow(win, out)
            key = cv.waitKey(opt.waitkey) & 0xFF
            if key == ord('q'):
                print('Quitting, \'q\' pressed.')
                break

        end = time.time()

        net_t = (1. / float(end1 - start))
        total_t = (1. / float(end - start))
        frame = frame + 1

    if opt.display:
        # Close any remaining windows.
        cv.destroyAllWindows()
    t_end = time.time()
    print("Total time spent:%f" % (t_end - t_begin))
    print("Average frame rate:%f" % (frame / (t_end - t_begin)))

    print('==> Finshed Demo.')
