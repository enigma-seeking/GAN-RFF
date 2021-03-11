import torch
import torch.nn as nn
import torch.optim as opt
from math import pi
import cmath
from RFF_Identify import TwoD_Net
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import GAN_data_maker as process
from GAN import NetG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 只能单GPU运行
if(torch.cuda.is_available()):
    print("Use GPU")
else:
    print("Use CPU")
model = TwoD_Net.Net()
model_G = NetG.Net_G()
model.cuda()
model_G.cuda()
IQreal_Tphase = 1 * pi / 12  # 合法发射机
IQfake_Tphase = 1 * pi / 6  # 非法发射机
IQreal_Rphase = -1 * pi / 12  # 合法接收机
IQfake_Rphase = -1 * pi / 6  # 非法接收机
real_Talpha = 0.5
fake_Talpha = -0.5
real_Ralpha = 0.05
fake_Ralpha = -0.05
# print(model)
# 定义损失函数
loss_fc = nn.BCELoss()
# 采用随机梯度下降SGD
optimizer = opt.Adam(params=model.parameters(), lr=0.001)
optimizer_G = opt.Adam(params=model_G.parameters(), lr=1e-6)
# 记录每次的损失值
loss_list_D = []
loss_list_G = []
# 记录训练次数
x = []
start_time = time.time()
train_data = pd.DataFrame(np.load("GAN_test_data.npy"))
fake_pre_data = np.load("../RFF_Identify/fDATA_rand_point.npy")
fake_pre_data = process.data_union(fake_pre_data)
fake_pre_data = fake_pre_data.reshape(10000,900)
model_G.load_state_dict(torch.load('GAN_G.pkl'))
model.load_state_dict(torch.load('../RFF_Identify/Leg_Net.pkl'))
test_correct = 0
SNR = 30
for n in range(20):
    G_input = torch.FloatTensor(fake_pre_data[n % 100 * 100:(n % 100 + 1) * 100, :]).reshape(100, 1, -1, 900).cuda()
    outxr, outxi = model_G(G_input)
    outxr = outxr.detach().cpu().numpy()
    outxi = outxi.detach().cpu().numpy()
    xr = outxr.reshape(45000)  # 变成一个一维 400*100
    xi = outxi.reshape(45000)
    out_oneline = np.zeros(90000)  # 100个sample，每个sample800个点
    for i in range(45000):
        out_oneline[2 * i] = xr[i]  # 按行out输入到一行的out_oneline中
        out_oneline[2 * i + 1] = xi[i]
    out_data_later = process.data_to_complex_line(out_oneline)  # 重新转化为复数
    X_bar1 = process.Normalization1(out_data_later )
    out_data_later = process.Normalization1(process.data_process(X_bar1, 1/20, 1/50,IQfake_Tphase, fake_Talpha, IQreal_Rphase, real_Ralpha,SNR))
    out_data_later = process.data_union(out_data_later)
    G_output = out_data_later.reshape(100, 900)
    fake_lable = np.zeros((100, 1))  # 反转标签训练
    G_output = pd.DataFrame(np.hstack([fake_lable, G_output]))
    # batch_data = train_data.sample(n=100, replace=False)
    # 标签值
    batch_Gy = torch.from_numpy(G_output.iloc[:, 0].values).float()
    batch_Gx = torch.from_numpy(G_output.iloc[:, 1::].values).float() \
        .view(-1, 1, 30, 30)
    batch_Gy = batch_Gy.cuda()
    batch_Gx = batch_Gx.cuda()

    # 图片信息，一条数据784维将其转化为通道数为1，大小28*28的图片。
    # 1.前向传播c
    pred = (model(batch_Gx) >0.5).cpu().float()  #自己已经确认过一遍了，再找师兄老师再确认一遍，这个出来的就是判决符合要求的为1，
    # 像在这里，想要伪装为真数据，那么输出值要小于0.5（因为反转标签了），pred中伪装成功的都为1，所以下面对于ones就可以了
    # print(pred.shape)
    test_correct += pred.eq(torch.ones((100, 1)).view_as(pred)).sum().item()

accuracy = test_correct/2000
# print(prediction)
# # print(batch_y)
print("第%d组测试集，准确率为%.3f" % (1, accuracy))

