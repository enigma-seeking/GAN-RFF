from math import pi
import GAN_data_maker as process
import numpy as np
import cmath
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import matplotlib.pyplot as plt
from NetG import Net_G
from complexLayers import ComplexConv2d
from torch.nn import Module, Parameter, init, Sequential
from torch.nn import Conv2d
import matplotlib as  mpl
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 只能单GPU运行
print("是否使用GPU加速:")
print(torch.cuda.is_available())
def data_process(data, tran_phase, rec_phase, H, f):
    ## 2.数据处理函数 传参分别为：处理的数据；发射机相位；接收机相位；收发双方信道 ##
    sample_phase = np.zeros(100)
    data = copy.copy(data)
    for i in range(100):  # 产生与每组内100个点做循环加相偏的100个相位
        sample_phase[i] = (i) * pi / f  # 样本点相位偏移0到99π/50
    for k in range(data.size // 100):  # 将40万个点分为4000组100个点
        data_phase = np.zeros(100)
        data_phase = data[k * 100:(k + 1) * 100]
        # 在200000个样本点中每100个样本点加一次0到99π/50的循环
        data[k * 100:(k + 1) * 100] = data_phase * (np.cos(sample_phase) + 1j * np.sin(sample_phase))
    # data[k*100+i] = data[k*100+i]*(np.cos(sample_phase[i])+1j*np.sin(sample_phase[i]))
    # print("data后：",data[k*100:(k+1)*100])
    # print(k*100+i)
    # print(data)
    # print(data.size)

    ##（二）合法发射机与非法发射机相位偏移T_B,T_M ##
    tran_phase = tran_phase  # 调用发射机相位参数，传参
    data = data * (np.cos(tran_phase) + 1j * np.sin(tran_phase))
    # data_1和data_2水平组合
    # data=np.hstack((data_1,data_2))

    ##（三）瑞利信道H_BA，H_MA 500个sample内固定不变##

    # l=int (data.size)
    # H=rice_matrix(5,1,l)
    # ah = np.random.randn(1)  # 信道的实部
    # #     # #print(ah)
    # bh = np.random.randn(1)  # 信道的虚部
    # H = (1 / cmath.sqrt(2)) * (ah + 1j * bh)
    data = data * H

    # 连接data_1，data_2,水平组合
    # data = np.hstack((data_1,data_2))
    # print(data)

    ##（四）高斯白噪声N ##
    # 信噪比为30dB

    SNR = 30
    snr = 10 ** (SNR / 10.0)  # 比例SNR与dB为单位的snr之间的转换
    an = np.random.randn(data.size) / cmath.sqrt(2 * snr)
    bn = np.random.randn(data.size) / cmath.sqrt(2 * snr)
    N = an + 1j * bn
    # print("N:")
    # print(N)

    data = data + N  # 每个样本点上都加上不一样的高斯白噪声
    # print(data)

    ##（五）合法接收机的相位偏移R_A，并提取实部虚部形成矩阵##
    rec_phase = rec_phase  # 合法接收机相位，传参
    DATA = data * (np.cos(rec_phase) + 1j * np.sin(rec_phase))
    # print("DATA:")
    # print(DATA)
    return DATA
H_ME = np.load("H_ME.npy")
fake_Tphase = 1 * pi / 6  # 非法发射机
fake_Rphase = 1 * pi / 5  # 非法接收机


net_G = Net_G().to(device)

net_G.load_state_dict(torch.load("0.888net_G_cov.pkl", map_location=device))

##Y1##
fake_pre_data_complex = np.load("fDATA_rand_point.npy")  # GAN使用假数据的QPSK信号点与合法接收机判别器D使用的假数据信号点一致
#print("fake_pre_data",fake_pre_data)
X1 = fake_pre_data_complex[0:400]   #取的训练集的第一个
X2 = fake_pre_data_complex[400:800]   #和第二个sample

#print("X1",X1.size)

#print("shape:",fake_pre_data.shape)
fake_pre_data = process.data_union(fake_pre_data_complex) #划分为实数

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
train_f_data_Ginput = train_data_fake.reshape((int(10000*0.8), 800))

input = torch.FloatTensor(train_f_data_Ginput[0:100, :]).reshape(100,1,-1,800).to(device)
outxr,outxi = net_G(input)
outxr = outxr.detach().cpu().numpy()
outxi = outxi.detach().cpu().numpy()
xr = outxr.reshape(40000)
xi = outxi.reshape(40000)
out_oneline = np.zeros(80000)  #100个sample，每个sample800个点
for i in range(40000):
    out_oneline[2*i] = xr[i]   #按行out输入到一行的out_oneline中
    out_oneline[2*i+1] = xi[i] 
    # print("out_oneline.shape",out_oneline.shape)
out_data_later = process.data_to_complex_line(out_oneline)
#print("out_data_later",out_data_later)
modulus = np.zeros(out_data_later.size)
for k in range(out_data_later.size):
    modulus[k] = abs(out_data_later[k])**2
#print("modulus",modulus)    
pw = modulus.mean()    
X_bar = out_data_later/cmath.sqrt(pw)
plt.scatter(X_bar.real,X_bar.imag,marker=',')
plt.scatter(fake_pre_data_complex.real,fake_pre_data_complex.imag,marker='+')
plt.xticks(fontsize = 8.5)
plt.yticks(fontsize = 8.5)
plt.xlabel(u"同相分量",fontsize = 13)
plt.ylabel(u"正交分量",fontsize = 13)
plt.savefig('GX_influence.png')
plt.close()  
