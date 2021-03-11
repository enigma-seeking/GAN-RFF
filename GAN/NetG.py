import numpy as np
from math import pi
import math
import cmath
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import complexLayers as cl
import complexFunctions as cf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Net_G(nn.Module):
# 三层卷积层
    def __init__(self):
        super(Net_G, self).__init__()
        self.conv1 = cl.ComplexConv2d(1, 1, 1, 1)
        self.conv2 = cl.ComplexConv2d(1, 1, 1, 1)
        self.conv3 = cl.ComplexConv2d(1, 1, 1, 1)
# ComplexConv2d 卷积核为一，步长为一，不补零   进去和出来的长度相同
    # TODO

    def forward(self, x):
        xr = x[:, :, :, 0:: 2]  # 4维 从第一维每隔2个取一维
        # imaginary part to zero
        xi = x[:, :, :, 1:: 2]  # 4维 从第二维每隔2个取一维
        xr, xi = self.conv1(xr, xi)
        x_br = xr.detach().cpu()
        x_bi = xi.detach().cpu()
        xr, xi = self.conv2(xr.to(device), xi.to(device))
        # BN = cl.ComplexBatchNorm2d(1)
        # xr, xi = BN(xr, xi)
        # xr, xi = cf.complex_relu(xr, xi)
        xr, xi = self.conv3(xr.to(device), xi.to(device))
        # xr, xi = BN(xr, xi)
        xr = xr.detach().cpu()
        xi = xi.detach().cpu()
        xr = xr + x_br
        xi = xi + x_bi

        xr, xi = self.conv1(xr.to(device), xi.to(device))
        x_br = xr.detach().cpu()
        x_bi = xi.detach().cpu()
        xr, xi = self.conv2(xr, xi)
        # BN = cl.ComplexBatchNorm2d(1)
        # xr, xi = BN(xr, xi)
        # xr, xi = cf.complex_relu(xr, xi)
        xr, xi = self.conv3(xr.to(device), xi.to(device))
        # xr, xi = BN(xr, xi)
        xr = xr.detach().cpu()
        xi = xi.detach().cpu()
        xr = xr + x_br
        xi = xi + x_bi
        xr, xi = self.conv3(xr.to(device), xi.to(device))
        xr, xi = self.conv1(xr.to(device), xi.to(device))
        x_br = xr.detach().cpu()
        x_bi = xi.detach().cpu()
        xr, xi = self.conv2(xr, xi)
        # BN = cl.ComplexBatchNorm2d(1)
        # xr, xi = BN(xr, xi)
        # xr, xi = cf.complex_relu(xr, xi)
        xr, xi = self.conv3(xr.to(device), xi.to(device))
        # xr, xi = BN(xr, xi)
        xr = xr.detach().cpu()
        xi = xi.detach().cpu()
        xr = xr + x_br
        xi = xi + x_bi
        xr, xi = self.conv3(xr.to(device), xi.to(device))
        xr, xi = self.conv1(xr.to(device), xi.to(device))
        x_br = xr.detach().cpu()
        x_bi = xi.detach().cpu()
        xr, xi = self.conv2(xr, xi)
        # BN = cl.ComplexBatchNorm2d(1)
        # xr, xi = BN(xr, xi)
        # xr, xi = cf.complex_relu(xr, xi)
        xr, xi = self.conv3(xr.to(device), xi.to(device))
        # xr, xi = BN(xr, xi)
        xr = xr.detach().cpu()
        xi = xi.detach().cpu()
        xr = xr + x_br
        xi = xi + x_bi
        xr, xi = self.conv3(xr.to(device), xi.to(device))
        #xr, xi = BN(xr, xi)  # 目前改的可能有点问题
        # xr, xi = cf.complex_relu(xr, xi)
        return xr, xi