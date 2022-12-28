import os
import time
import shutil
import os.path as osp
from timeit import default_timer as timer

import numpy as np
import torch
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from FClip.utils import recursive_to, ModelPrinter
from FClip.config import C
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'


class Trainer(object):
    def __init__(self, device, model, optimizer, lr_scheduler, train_loader, val_loader, out, iteration=0, epoch=0, bml=1e1000):

        from FClip.visualize import VisualizeResults
        self.device = device

        self.model = model
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = C.model.batch_size
        self.eval_batch_size = C.model.eval_batch_size

        self.validation_interval = C.io.validation_interval

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.epoch = epoch
        self.iteration = iteration
        self.max_epoch = C.optim.max_epoch
        self.lr_decay_epoch = C.optim.lr_decay_epoch
        self.num_stacks = C.model.num_stacks
        self.mean_loss = self.best_mean_loss = bml

        self.loss_labels = None
        self.acc_label = None
        self.avg_metrics = None
        self.metrics = np.zeros(0)
        self.visual = VisualizeResults()
        self.printer = ModelPrinter(out)

    def _loss(self, result):
        losses = result["losses"]
        accuracy = result["accuracy"]
        # Don't move loss label to other place.
        # If I want to change the loss, I just need to change this function.
        if self.loss_labels is None:
            self.loss_labels = ["sum"] + list(losses[0].keys())
            self.acc_label = ["Acc"] + list(accuracy[0].keys())
            self.metrics = np.zeros([self.num_stacks, len(self.loss_labels)+len(self.acc_label)])

            self.printer.loss_head(loss_labels=self.loss_labels+self.acc_label)

        total_loss = 0
        for i in range(self.num_stacks):
            for j, name in enumerate(self.loss_labels):
                if name == "sum":
                    continue
                if name not in losses[i]:
                    assert i != 0
                    continue
                loss = losses[i][name].mean()
                self.metrics[i, 0] += loss.item()
                self.metrics[i, j] += loss.item()
                total_loss += loss

        for i in range(self.num_stacks):
            for j, name in enumerate(self.acc_label, len(self.loss_labels)):
                if name == "Acc":
                    continue
                if name not in accuracy[i]:
                    assert i != 0
                    continue
                acc = accuracy[i][name].mean()
                self.metrics[i, j] += acc.item()

        return total_loss

    def validate(self, isviz=True, isnpz=True, isckpt=True):
        self.printer.tprint("Running validation...", " " * 55)
        training = self.model.training
        self.model.eval()

        if isviz:
            viz = osp.join(self.out, "viz", f"{self.iteration * self.batch_size:09d}")
            osp.exists(viz) or os.makedirs(viz)
        if isnpz:
            npz = osp.join(self.out, "npz", f"{self.iteration * self.batch_size:09d}")
            osp.exists(npz) or os.makedirs(npz)

        total_loss = 0
        self.metrics[...] = 0
        with torch.no_grad():
            for batch_idx, (image, meta, target) in enumerate(self.val_loader):
                input_dict = {
                    "image": recursive_to(image, self.device),
                    "meta": recursive_to(meta, self.device),
                    "target": recursive_to(target, self.device),
                    "do_evaluation": True,
                }

                result = self.model(input_dict)
                total_loss += self._loss(result)

                H = result["heatmaps"]
                for i in range(image.shape[0]):
                    index = batch_idx * self.eval_batch_size + i
                    if isnpz:
                        npz_dict = {}
                        for k, v in H.items():
                            if v is not None:
                                npz_dict[k] = v[i].cpu().numpy()
                        np.savez(
                            f"{npz}/{index:06}.npz",
                            **npz_dict,
                        )

                    if index >= C.io.visual_num:
                        continue
                    if isviz:
                        fn = self.val_loader.dataset._get_im_name(index)
                        self.visual.plot_samples(fn, i, H, target, meta, f"{viz}/{index:06}")
                self.printer.tprint(f"Validation [{batch_idx:5d}/{len(self.val_loader):5d}]", " " * 25)

        self.printer.valid_log(len(self.val_loader), self.epoch, self.iteration, self.batch_size, self.metrics[0])
        self.mean_loss = total_loss / len(self.val_loader)

        if isckpt:
            torch.save(
                {
                    "iteration": self.iteration,
                    "arch": self.model.__class__.__name__,
                    "optim_state_dict": self.optim.state_dict(),
                    "model_state_dict": self.model.state_dict(),
                    "best_mean_loss": self.best_mean_loss,
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                },
                osp.join(self.out, "checkpoint_lastest.pth.tar"),
            )
            shutil.copy(
                osp.join(self.out, "checkpoint_lastest.pth.tar"),
                osp.join(npz, "checkpoint.pth.tar"),
            )
            if self.mean_loss < self.best_mean_loss:
                self.best_mean_loss = self.mean_loss
                shutil.copy(
                    osp.join(self.out, "checkpoint_lastest.pth.tar"),
                    osp.join(self.out, "checkpoint_best.pth.tar"),
                )

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        time = timer()

        for batch_idx, (image, meta, target) in enumerate(self.train_loader):

            self.optim.zero_grad()
            self.metrics[...] = 0

            input_dict = {
                "image": recursive_to(image, self.device),
                "meta": recursive_to(meta, self.device),
                "target": recursive_to(target, self.device),
                "do_evaluation": False,
            }
            result = self.model(input_dict)

            loss = self._loss(result)
            if np.isnan(loss.item()):
                print("\n")
                print(self.metrics[0])
                raise ValueError("loss is nan while training")
            loss.backward()
            self.optim.step()

            if self.avg_metrics is None:
                self.avg_metrics = self.metrics
            else:
                self.avg_metrics[0, :len(self.loss_labels)] = self.avg_metrics[0, :len(self.loss_labels)] * 0.9 + \
                                                              self.metrics[0, :len(self.loss_labels)] * 0.1
                if len(self.loss_labels) < self.avg_metrics.shape[1]:
                    self.avg_metrics[0, len(self.loss_labels):] = self.metrics[0, len(self.loss_labels):]

            if self.iteration % 4 == 0:
                self.printer.train_log(self.epoch, self.iteration, self.batch_size, time, self.avg_metrics)

                time = timer()
            num_images = self.batch_size * self.iteration
            if num_images % self.validation_interval == 0 or num_images == 60:
                # record training loss
                if num_images > 0:
                    self.printer.valid_log(1, self.epoch, self.iteration, self.batch_size, self.avg_metrics[0],
                                           csv_name="train_loss.csv", isprint=False)
                    self.validate()
                    time = timer()

            self.iteration += 1

    def train(self):
        plt.rcParams["figure.figsize"] = (24, 24)
        epoch_size = len(self.train_loader)
        start_epoch = self.iteration // epoch_size
        for self.epoch in range(start_epoch, self.max_epoch):
            self.train_epoch()
            self.lr_scheduler.step()
