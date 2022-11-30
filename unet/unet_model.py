""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, dropout=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        p = 0.0
        if dropout:
            p = 0.2

        # [32, 64, 128, 192, 192]
        self.inc = DoubleConv(n_channels, 32, p=p)
        self.down1 = Down(32, 64, p=p)
        self.down2 = Down(64, 128, p=p)
        self.down3 = Down(128, 192, p=p)
        factor = 2 if bilinear else 1
        self.down4 = Down(192, 192 // factor, p=p, p=p)
        self.up1 = Up(192 + (192 // factor), 192 // factor, bilinear, p=p)
        self.up2 = Up(128 + (192 // factor), 128 // factor, bilinear, p=p)
        self.up3 = Up(64 + (128 // factor), 64 // factor, bilinear, p=p)
        self.up4 = Up(32 + (64 // factor), 32, bilinear, p=p)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4_ = self.up1(x5, x4)
        x3_ = self.up2(x4_, x3)
        x2_ = self.up3(x3_, x2)
        x1_ = self.up4(x2_, x1)
        logits = self.outc(x1_)
        return logits, x2_
