from math import pi
import copy
import numpy as np
import cmath
import torch
import torch.nn as nn
import GAN_data_maker as process
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import matplotlib.pyplot as plt
from complexLayers import ComplexConv2d
from torch.nn import Module, Parameter, init, Sequential
from torch.nn import Conv2d
from GAN import NetG
import matplotlib as mpl

import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 只能单GPU运行
print("是否使用GPU加速:")
print(torch.cuda.is_available())

def random_point():
    # 定义采样点长度
    size = 45000

    a = np.random.randint(0, 4, size)  # 生成4000000个采样样本点，0,1,2,3

    # 相位选择向量，00对应π/4；01对应3π/4；11对应5π/4；10对应7π/4
    phase_map = ([1 * pi / 4, 3 * pi / 4, 5 * pi / 4, 7 * pi / 4])
    phase = np.zeros(size)
    # data = np.zeros(size)
    # 对应相位与样本点
    for i in range(size):
        t = a[i]
        phase[i] = phase_map[t]
    data = np.cos(phase) + 1j * np.sin(phase);
    return data


def data_process(data,ft,IQtran_phase,Talpha):
    data = (1 + Talpha) * data.real * np.exp(1j * IQtran_phase) + 1j * data.imag * np.exp(-1j * IQtran_phase)
    sample_phase = np.zeros(450)
    for i in range(450):  # 产生与每组内100个点做循环加相偏的100个相位
        sample_phase[i] = (2 * i * pi) * (ft)  # 样本点相位偏移0到99π/f
    for k in range(data.size // 450):  # 将40万个点分为4000组100个点
        data_phase = data[k * 450:(k + 1) * 450]
        data[k * 450:(k + 1) * 450] = data_phase * np.exp(1j * sample_phase)


    return data

rDATA_rand_point = random_point()  #合法发射机样本随机点，未经过射频前端
fDATA_rand_point = random_point()
IQreal_Tphase = 1*pi/12  #合法发射机
IQfake_Tphase = 1*pi/6  #非法发射机
IQreal_Rphase = -1*pi/12  #合法接收机
IQfake_Rphase = -1*pi/6  #非法接收机
real_Talpha = 0.1
fake_Talpha = -0.1
real_Ralpha = 0.05
fake_Ralpha = -0.05
r = data_process(rDATA_rand_point,1/30,IQreal_Tphase,real_Talpha)  #合法发射机  H_BA  1/50,, IQreal_Rphase,real_Ralpha
f = data_process(fDATA_rand_point,1/20,IQfake_Tphase,fake_Talpha)  #1/50,  , IQreal_Rphase,real_Ralpha


# plt.scatter(f.real[0:450],f.imag[0:450])
# plt.xticks(fontsize = 8.5)
# plt.yticks(fontsize = 8.5)
# plt.xlim(-1.6, 1.6)
# plt.ylim(-1.6, 1.6)
# #plt.title(u"准确率变化对比图")
# plt.xlabel(u"同相分量",fontsize = 13)
# plt.ylabel(u"正交分量",fontsize = 13)
# plt.savefig('AR.png')
# plt.close()

net_G = NetG.Net_G().to(device)
net_G.load_state_dict(torch.load("GAN/GAN_G.pkl", map_location=device))

##Y1##
fake_pre_data = np.load("RFF_Identify/fDATA_rand_point.npy")  # GAN使用假数据的QPSK信号点与合法接收机判别器D使用的假数据信号点一致
print("fake_pre_data", fake_pre_data)
X1 = fake_pre_data[0:45000]


# print("X1",X1.size)
# SNR = 30
# snr = 10 ** (SNR / 10.0)  # 比例SNR与dB为单位的snr之间的转换
# an = np.random.randn(fake_pre_data.size) / cmath.sqrt(2 * snr)
# bn = np.random.randn(fake_pre_data.size) / cmath.sqrt(2 * snr)
# N = an + 1j * bn
#
#
# fake_pre_data_next = fake_pre_data+N
# fake_pre_data[3000000:4000000] = fake_pre_data[3000000:4000000] *np.exp(1j*pi/4)
# # print("shape:",fake_pre_data.shape)
# plt.scatter(fake_pre_data.real[0:4000000],fake_pre_data.imag[0:4000000])
# plt.xticks(fontsize = 8.5)
# plt.yticks(fontsize = 8.5)
# plt.xlim(-1.6, 1.6)
# plt.ylim(-1.6, 1.6)
# #plt.title(u"准确率变化对比图")
# plt.xlabel(u"同相分量",fontsize = 13)
# plt.ylabel(u"正交分量",fontsize = 13)
# plt.savefig('fake_pre_data.png')
# plt.close()
fake_pre_data = process.data_union(fake_pre_data)  # 划分为实数
# plt.scatter(fake_pre_data.real[0:8000000],fake_pre_data.imag[0:8000000])
# plt.xticks(fontsize = 8.5)
# plt.yticks(fontsize = 8.5)
# plt.xlim(-1.6, 1.6)
# plt.ylim(-1.6, 1.6)
# #plt.title(u"准确率变化对比图")
# plt.xlabel(u"同相分量",fontsize = 13)
# plt.ylabel(u"正交分量",fontsize = 13)
# plt.savefig('fake_pre_data2.png')
# plt.close()

def data_splite(data, train_rate, test_rate, eval_rate, Select_fun='Train'):
    size = int(data.size)
    if (Select_fun == 'Train'):
        select_data = data[0:int(size * train_rate)]
    elif (Select_fun == 'Test'):
        select_data = data[int(size * train_rate):int(size * (train_rate + test_rate))]
    elif (Select_fun == 'Eval'):
        select_data = data[int(size * (1 - eval_rate)):size]
    return select_data


train_data_fake = data_splite(fake_pre_data, 0.8, 0.1, 0.1, Select_fun='Train')
train_f_data_Ginput = train_data_fake.reshape((int(10000 * 0.8), 900))

input = torch.FloatTensor(train_f_data_Ginput[0:100, :]).reshape(100, 1, -1, 900).to(device)  # 只过第一个batch的sample
outxr, outxi = net_G(input)
outxr = outxr.detach().cpu().numpy()
outxi = outxi.detach().cpu().numpy()
xr = outxr.reshape(45000)
xi = outxi.reshape(45000)
out_oneline = np.zeros(90000)  # 100个sample，每个sample800个点
for i in range(45000):
    out_oneline[2 * i] = xr[i]  # 按行out输入到一行的out_oneline中
    out_oneline[2 * i + 1] = xi[i]
    # print("out_oneline.shape",out_oneline.shape)
out_data_later = process.data_to_complex_line(out_oneline)
Y1 = out_data_later
plt.scatter(Y1.real[0:450], Y1.imag[0:450])
plt.xticks(fontsize=8.5)
plt.yticks(fontsize=8.5)
# plt.title(u"准确率变化对比图")
plt.xlabel(u"同相分量", fontsize=13)
plt.ylabel(u"正交分量", fontsize=13)
plt.savefig('G_final.png')
plt.close()
modulus1 = np.zeros(Y1.size)

for k in range(Y1.size):
    modulus1[k] = abs(Y1[k]) ** 2
#     modulus2[k] = abs(Y2[k]) ** 2
for k in range(Y1.size):
    modulus1[k] = abs(Y1[k])**2
pw1 = modulus1.mean()
# pw2 = modulus2.mean()
#
X_bar1 = Y1 / cmath.sqrt(pw1)
# X_bar2 = Y2 / cmath.sqrt(pw2)

Y1 = data_process(X_bar1,1/20,IQfake_Tphase,fake_Talpha) #,1/50  , IQreal_Rphase,real_Ralpha
plt.scatter(Y1.real[0:45000], Y1.imag[0:45000],c = 'r',marker = 'o',label="G_AR")
plt.scatter(r.real[0:450],r.imag[0:450],c = 'b',marker = 'x',label="R")
plt.scatter(f.real[0:450],f.imag[0:450],c = 'g',marker = 'v',label="AR")
plt.xticks(fontsize=8.5)
plt.yticks(fontsize=8.5)
plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)
# plt.title(u"准确率变化对比图")
plt.legend(loc='upper right')
plt.savefig('G_AR--R--AR.png')
plt.close()

# plt.xticks(fontsize = 8.5)
# plt.yticks(fontsize = 8.5)
# plt.xlim(-1.6, 1.6)
# plt.ylim(-1.6, 1.6)
# #plt.title(u"准确率变化对比图")
# plt.xlabel(u"同相分量",fontsize = 13)
# plt.ylabel(u"正交分量",fontsize = 13)
# plt.savefig('R.png')
# plt.close()