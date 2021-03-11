##本文件画出通过G之后的非法假样本信道估计与信道均衡后对初始信号点的影响##

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
from NetG import Net_G
import matplotlib as  mpl
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 只能单GPU运行
print("是否使用GPU加速:")
print(torch.cuda.is_available())

H_ME = np.load("H_ME.npy")

def data_process(data, ft, fr, IQtran_phase, Talpha, IQrec_phase, Ralpha,H):
    ## 2.数据处理函数 传参分别为：处理的数据；发射机相位；接收机相位；收发双方信道 ##
    data = (1 + Talpha) * data.real * np.exp(1j * IQtran_phase) + 1j * data.imag * np.exp(-1j * IQtran_phase)

    #

    ##（三）瑞利信道H_BA，H_MA ##
    # H_leg = []
    # for i in range(400):
    #     ah = np.random.randn(1)  # 信道的实部
    #     # #print(ah)
    #     bh = np.random.randn(1)  # 信道的虚部
    #     H = (1 / cmath.sqrt(2)) * (ah + 1j * bh)
    #     H_leg = H_leg+[H]*10000   #append适合加一个数，加列表用加号
    # #H_leg.append(H)
    # H_leg = np.array(H_leg)
    # data = data * H_leg  #不报错，但是显存爆了
    # ah = np.random.randn(1)  # 信道的实部
    # #     # #print(ah)
    # bh = np.random.randn(1)  # 信道的虚部
    # H = (1 / cmath.sqrt(2)) * (ah + 1j * bh)
    # H = -0.13176909 - 0.94339763j
    data = data * H
    ##（四）高斯白噪声N ##
    # 信噪比为30dB

    SNR = 30
    snr = 10 ** (SNR / 10.0)  # 比例SNR与dB为单位的snr之间的转换
    an = np.random.randn(data.size) / cmath.sqrt(2 * snr)
    bn = np.random.randn(data.size) / cmath.sqrt(2 * snr)
    N = an + 1j * bn
    data = data + N  # 每个样本点上都加上不一样的高斯白噪声
    # print(data)

    # 接收端
    K1 = ((1 + Ralpha) * np.exp(-1j * IQrec_phase) + np.exp(1j * IQrec_phase)) / 2
    K2 = ((1 + Ralpha) * np.exp(1j * IQrec_phase) - np.exp(-1j * IQrec_phase)) / 2
    I = K1 * data
    Q = K2 * (data.conjugate())
    sample_phase = np.zeros(400)
    for i in range(400):  # 产生与每组内100个点做循环加相偏的100个相位
        sample_phase[i] = (2 * i * pi) * (ft - fr)  # 样本点相位偏移0到99π/f
    for k in range(data.size // 400):  # 将40万个点分为4000组100个点
        data_phaseI = I[k * 400:(k + 1) * 400]
        data_phaseQ = Q[k * 400:(k + 1) * 400]
        # 在200000个样本点中每100个样本点加一次0到99π/50的循环
        data[k * 400:(k + 1) * 400] = data_phaseI * np.exp(1j * sample_phase) + data_phaseQ * np.exp(-1j * sample_phase)
    ##同步误差
    data = data * np.exp(1j * (pi * np.random.randint(-30, 30) / 180))

    # print("DATA:")
    # print(DATA)
    return data


IQreal_Tphase = 1 * pi / 12  # 合法发射机
IQfake_Tphase = 1 * pi / 6  # 非法发射机
IQreal_Rphase = -1 * pi / 12  # 合法接收机
IQfake_Rphase = -1 * pi / 6  # 非法接收机
real_Talpha = 0.5
fake_Talpha = -0.5
real_Ralpha = 0.05
fake_Ralpha = -0.05
net_G = Net_G().to(device)

net_G.load_state_dict(torch.load("0.996net_G_cov.pkl", map_location=device))

##Y1##
fake_pre_data = np.load("fDATA_rand_point.npy")  # GAN使用假数据的QPSK信号点与合法接收机判别器D使用的假数据信号点一致
print("fake_pre_data",fake_pre_data)
X1 = fake_pre_data[0:20000]   #取的训练集的第一个
X2 = fake_pre_data[20000:40000]  #和第二个sample

#print("X1",X1.size)

#print("shape:",fake_pre_data.shape)
fake_pre_data = process.data_union(fake_pre_data) #划分为实数

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

input = torch.FloatTensor(train_f_data_Ginput[0:100, :]).reshape(100,1,-1,800).to(device)  #只过第一个batch的sample
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
Y1 = out_data_later[0:20000]
Y2 = out_data_later[20000:40000]
plt.scatter(Y1.real[0:40000],Y1.imag[0:40000])
plt.xticks(fontsize = 8.5)
plt.yticks(fontsize = 8.5)
#plt.title(u"准确率变化对比图")
plt.xlabel(u"同相分量",fontsize = 13)
plt.ylabel(u"正交分量",fontsize = 13)
plt.savefig('G_final.png')
plt.close()
modulus1 = np.zeros(Y1.size)
modulus2 = np.zeros(Y2.size)
for k in range(Y2.size):
    modulus1[k] = abs(Y1[k])**2
    modulus2[k] = abs(Y2[k])**2
# for k in range(Y1.size):
#     modulus1[k] = abs(Y1[k])**2
pw1 = modulus1.mean()
pw2 = modulus2.mean()

X_bar1 = Y1/cmath.sqrt(pw1)
X_bar2 = Y2/cmath.sqrt(pw2)




Y1 = data_process(X_bar1, 1/20, 1/40,IQfake_Tphase, fake_Talpha, IQfake_Rphase, fake_Ralpha,H_ME)
Y2 = data_process(X_bar2, 1/20, 1/40,IQfake_Tphase, fake_Talpha, IQfake_Rphase, fake_Ralpha,H_ME)
'''
Y1 = out_data_later[0:400]
#print("Y1",Y1)
Y2 = out_data_later[400:800]   
#print("Y2",Y2) 
#print("Y1",Y1)
#print("Y2",Y2.size)
'''
H1 = Y1/X1  #这个H包含了H、G、发射机接收机相偏
#print("H1",H1)
X = Y2/H_ME #均衡后的得到的还原的X
#print("X",X)
#print("X2",X2)
plt.scatter(Y2.real[0:40000],Y2.imag[0:40000])
plt.xticks(fontsize = 8.5)
plt.yticks(fontsize = 8.5)
#plt.title(u"准确率变化对比图")
plt.xlabel(u"同相分量",fontsize = 13)
plt.ylabel(u"正交分量",fontsize = 13)
plt.savefig('R_final.png')
plt.close()
'''
sample_phase = np.zeros(100)
for i in range(100): #产生与每组内100个点做循环加相偏的100个相位
    sample_phase[i] = (i)*pi/50  #样本点相位偏移0到99π/50


for k in range(Y1.size//100):  #将40万个点分为4000组100个点
    data_phase1 = np.zeros(100)
    data_phase1 = Y1[k*100:(k+1)*100]
    #在200000个样本点中每100个样本点加一次0到99π/50的循环
    #print("data_phase1",data_phase1)
    #print("sample_phase",sample_phase)    
    Y1[k*100:(k+1)*100] = data_phase1 / (np.cos(sample_phase)+1j * np.sin(sample_phase))
    #print("Y1",Y1[k*100:(k+1)*100])
#print("Y1",Y1)
Y1_1 = Y1 / (np.cos(fake_Tphase)+1j * np.sin(fake_Tphase))
Y1_2 = Y1_1 / (np.cos(fake_Rphase)+1j * np.sin(fake_Rphase))
H1 = Y1_2/X1  #这个H包含了H、G及噪声

#print("H1",H1)

for k in range(Y2.size//100):  #将40万个点分为4000组100个点
    data_phase2 = np.zeros(100)
    data_phase2 = Y2[k*100:(k+1)*100]
    #print("sample_phase",sample_phase)
    #在200000个样本点中每100个样本点加一次0到99π/50的循环 
    Y2[k*100:(k+1)*100] = data_phase2 / (np.cos(sample_phase)+1j * np.sin(sample_phase))
Y2_1 = Y2 / (np.cos(fake_Tphase)+1j * np.sin(fake_Tphase))
Y2_2 = Y2_1 / (np.cos(fake_Rphase)+1j * np.sin(fake_Rphase))  
X = Y2_2/H1  #此为第二个sample信道均衡后还原的信号
#print("X",X)
#print("X2",X2)
'''
X2_pre_digital_point = np.zeros(40000)
for i in range(20000):
    if X2[i].real > 0 and X2[i].imag > 0:
        X2_pre_digital_point[2*i] = 0
        X2_pre_digital_point[2*i+1] = 0
    elif X2[i].real > 0 and X2[i].imag < 0:
        X2_pre_digital_point[2*i] = 1
        X2_pre_digital_point[2*i+1] = 0
    elif (X2[i].real < 0 and X2[i].imag > 0):
        X2_pre_digital_point[2*i] = 0
        X2_pre_digital_point[2*i+1] = 1
    elif (X2[i].real < 0 and X2[i].imag < 0):
        X2_pre_digital_point[2*i] = 1
        X2_pre_digital_point[2*i+1] = 1    
#print("X_pre-point",X2_pre_digital_point)  #此为第二个sample原始输入对应星座图还原的数字信号序列
X2_re_digital_point = np.zeros(40000)
for i in range(20000):
    if (X[i].real > 0 and X[i].imag > 0):
        X2_re_digital_point[2*i] = 0
        X2_re_digital_point[2*i+1] = 0
    elif (X[i].real > 0 and X[i].imag < 0):
        X2_re_digital_point[2*i] = 1
        X2_re_digital_point[2*i+1] = 0
    elif (X[i].real < 0 and X[i].imag > 0):
        X2_re_digital_point[2*i] = 0
        X2_re_digital_point[2*i+1] = 1
    elif (X[i].real < 0 and X[i].imag < 0):
        X2_re_digital_point[2*i] = 1
        X2_re_digital_point[2*i+1] = 1    
#print("X_re-point",X2_re_digital_point)    #此为第二个sample信道估计均衡后还原的数字信号位
corrected = 0
for j in range(20000):
    if X2_pre_digital_point[2*j] == X2_re_digital_point[2*j] and  X2_pre_digital_point[2*j+1] == X2_re_digital_point[2*j+1]:
        corrected = corrected+1
print("corrected:",corrected)        
    

plt.scatter(X.real[0:20000],X.imag[0:20000])
plt.xticks(fontsize = 8.5)
plt.yticks(fontsize = 8.5)
#plt.title(u"准确率变化对比图")
plt.xlabel(u"同相分量",fontsize = 13)
plt.ylabel(u"正交分量",fontsize = 13)
plt.savefig('GEH_final.png')
plt.close()
plt.scatter(X2.real[0:20000],X2.imag[0:20000])
plt.xticks(fontsize = 8.5)
plt.yticks(fontsize = 8.5)
#plt.title(u"准确率变化对比图")
plt.xlabel(u"同相分量",fontsize = 13)
plt.ylabel(u"正交分量",fontsize = 13)
plt.savefig('GX2_final.png')
plt.close()
#