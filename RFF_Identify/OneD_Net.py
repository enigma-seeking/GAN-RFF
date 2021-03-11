import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import complexLayers as cl
import complexFunctions as cf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = cl.ComplexConv2d(1, 1, 1, 1)
        self.conv2 = cl.ComplexConv2d(1, 1, 1, 1)
        self.conv3 = cl.ComplexConv2d(1, 1, 1, 1)
        self.BN = cl.NaiveComplexBatchNorm2d(100)
        self.fc1 = nn.Linear(800, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 1)
        # init.xavier_uniform(self.fc4.weight)
        # init.constant(self.fc4.bias, 0.1)
        self.dropout = torch.nn.Dropout(p=0.5)


    def forward(self, x):
        xr1 = x[:,:,:, 0:: 2]  # 4维 从第一维每隔2个取一维
        # imaginary part to zero
        xi1 = x[:,:,:, 1:: 2]  # 4维 从第二维每隔2个取一维
        xr, xi = self.conv1(xr1, xi1)
        x_br = xr.detach().cpu()
        x_bi = xi.detach().cpu()
        xr, xi = self.conv2(xr.to(device), xi.to(device))
        BN = cl.ComplexBatchNorm2d(1)
        xr, xi = BN(xr, xi)
        xr, xi = cf.complex_relu(xr, xi)
        xr, xi = self.conv3(xr.to(device), xi.to(device))
        xr, xi = BN(xr, xi)
        xr = xr.detach().cpu()
        xi = xi.detach().cpu()
        xr = xr + x_br
        xi = xi + x_bi
        xr, xi = cf.complex_relu(xr.to(device), xi.to(device))

        xr, xi = self.conv2(xr.to(device), xi.to(device))
        xr = xr.detach().cpu().numpy()
        # 修改数据类型  CUDA tensor在这里应该是指变量而不是张量格式的数据改成numpy
        xi = xi.detach().cpu().numpy()

        xr = xr.reshape(40000)  # 变成一个一维 400*100

        xi = xi.reshape(40000)
        x = np.zeros(80000)  # 100个sample，每个sample800个点
        for i in range(40000):
            x[2 * i] = xr[i]  # 按行out输入到一行的out_oneline中
            x[2 * i + 1] = xi[i]
        x = x.reshape(100, 800)
        x = torch.FloatTensor(x).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = F.sigmoid(self.fc4(x))

        return x