import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    def __init__(self, targer):
        super(ContentLoss, self).__init__()
        self.target = targer.detach()
        self.loss = None

    def forward(self, d_in):
        self.loss = F.mse_loss(d_in, self.target)
        return d_in

    def update(self, target):
        self.target = target.detach()
