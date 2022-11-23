import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from attack import Attack
from train_utils import AverageMeter


class Interpolate(Attack):
    r"""
    Optimization to generate Interpolated Sample

    """

    def __init__(self, model, steps=40, random_start=True, lr=0.1, norm_mode="clamp"):
        super().__init__("Interpolate", model)
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ["default", "targeted"]
        self.lr = lr
        self.losses = AverageMeter()
        self.norm_mode = norm_mode

    def min_max_norm(self, x):
        ## normalize
        x = (x - x.min()) / (x.max() - x.min())
        return x

    def forward(self, images, tar_logits):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        tar_logits = tar_logits.clone().detach().to(self.device)

        loss = nn.MSELoss(size_average=None, reduce=None, reduction="mean")
        adv_images = images.clone().detach()
        adv_images.requires_grad = True
        optimizer = optim.Adam([adv_images], lr=self.lr, betas=(0.5, 0.999))

        cost_prev = 1e10
        for _ in range(self.steps):

            adv_images.requires_grad = True
            outputs, _ = self.model(adv_images)

            # Calculate loss
            cost = loss(outputs, tar_logits)

            if np.abs(cost_prev - cost.item()) < 1e-3:
                break  ## Break out of loop
            else:
                cost_prev = cost.item()

            self.losses.update(torch.abs(cost).item(), n=images.size(0))

            optimizer.zero_grad()

            cost.backward()
            optimizer.step()

            if self.norm_mode == "clamp":
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            elif self.norm_mode == "min_max":
                adv_images = self.min_max_norm(adv_images).detach()

        return adv_images
