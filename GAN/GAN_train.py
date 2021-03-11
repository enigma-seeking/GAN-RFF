import torch
import torch.nn as nn
from math import pi
import torch.optim as opt
import cmath
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import GAN_data_maker as process
from GAN import NetG, TwoD_NetD
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 只能单GPU运行
if(torch.cuda.is_available()):
    print("Use GPU")
else:
    print("Use CPU")
model_D = TwoD_NetD.Net_D()
model_G = NetG.Net_G()
model_D.cuda()
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
optimizer_D = opt.Adam(params=model_D.parameters(), lr=1e-2)
optimizer_G = opt.Adam(params=model_G.parameters(), lr=1e-7)
# 记录每次的损失值
loss_list_D = []
loss_list_G = []
# 记录训练次数
x = []
start_time = time.time()
train_data = pd.DataFrame(np.load("GAN_train_data.npy"))
fake_pre_data = np.load("../RFF_Identify/fDATA_rand_point.npy")
fake_pre_data = process.data_union(fake_pre_data)
fake_pre_data = fake_pre_data.reshape(10000,900)
model_D.load_state_dict(torch.load('../RFF_Identify/Net_D.pkl'))
SNR = 30
for n in range(150):
    # 每次随机读取30条数据
    G_input = torch.FloatTensor(fake_pre_data[n%100 * 100:(n%100 + 1) * 100, :]).reshape(100, 1, -1, 900).cuda()
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
    X_bar1 = process.Normalization1(out_data_later)
    out_data_later = process.Normalization1(process.data_process(X_bar1, 1 / 20, 1 / 40, IQfake_Tphase, fake_Talpha, IQfake_Rphase,
                                          fake_Ralpha,SNR))
    out_data_later = process.data_union(out_data_later)
    G_output = out_data_later.reshape(100, 900)
    fake_lable = np.zeros((100,1))  #反转标签训练
    G_output = pd.DataFrame(np.hstack([fake_lable, G_output]))
    # batch_data = train_data.sample(n=100, replace=False)
    # 标签值
    batch_Gy = torch.from_numpy(G_output.iloc[:, 0].values).float()
    batch_Gx = torch.from_numpy(G_output.iloc[:, 1::].values).float() \
        .view(-1, 1, 30, 30)
    batch_Gy = batch_Gy.cuda()
    batch_Gx = batch_Gx.cuda()

    batch_data = train_data.sample(n=100, replace=False)
    # 标签值
    batch_y = torch.from_numpy(batch_data.iloc[:, 0].values).float()
    batch_x = torch.from_numpy(batch_data.iloc[:, 1::].values).float() \
        .view(-1, 1, 30, 30)
    batch_y = batch_y.cuda()
    batch_x = batch_x.cuda()
    # 图片信息，一条数据784维将其转化为通道数为1，大小28*28的图片。
    for jd in range(1):
        # 1.前向传播c

        prediction_fake = model_D.forward(batch_Gx)
        prediction_real = model_D.forward(batch_x)
        # print('Prediction value is :', prediction)
        # print('Y value is :', batch_y)
        # 2.计算损失值
        fake_loss = loss_fc(prediction_fake, batch_Gy)
        real_loss = loss_fc(prediction_real, batch_y)
        # 反向传播
        optimizer_D.zero_grad()
        real_loss.backward()
        fake_loss.backward()

        # 更新权重
        optimizer_D.step()
        D_loss = fake_loss+real_loss
        print("第%d次训练，D_loss为%.3f" % (n, D_loss))


    loss_list_D.append(D_loss)
    for jg in range(3):
        # 1.前向传播c
        optimizer_G.zero_grad()
        g_loss = loss_fc(model_D.forward(batch_Gx), batch_y)  # 对于G来说，希望是假数据变成标签为1
        g_loss.backward()  # 反向传播
        optimizer_G.step()  # 优化器单次优化G
        print("第%d次训练，G_loss为%.3f" % (n, g_loss))
    x.append(n)
    loss_list_G.append(g_loss)
end_time = time.time()
print('Time cost', end_time - start_time, 's')
# 保存模型参数
torch.save(model_D.state_dict(), 'GAN_D.pkl')
# print("已保存模型")
plt.figure('D_loss')
# 可以将损失值进行绘制
plt.plot(x, loss_list_D, "r-")
plt.savefig('D_Loss.png')
plt.show()

torch.save(model_G.state_dict(), 'GAN_G.pkl')
# print("已保存模型")
plt.figure('G_loss')
# 可以将损失值进行绘制
plt.plot(x, loss_list_G, "r-")
plt.savefig('G_Loss.png')
plt.show()
print("结束")
