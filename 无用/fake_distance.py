##本文件画出非法假样本信道估计与信道均衡后对初始信号点的影响##

from math import pi
import copy
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

##Y1##
fake_pre_data = np.load("fDATA_rand_point.npy")  # GAN使用假数据的QPSK信号点与合法接收机判别器D使用的假数据信号点一致
#print("fake_pre_data",fake_pre_data.size)
X1 = fake_pre_data[0:400]
X2 = fake_pre_data[400:800]
print("X1",X1)
#print("X2",X2.size)

#print("shape:",fake_pre_data.shape)
fake_data = data_process(fake_pre_data,fake_Tphase,fake_Rphase,H_ME,1*pi/40) #划分为实数

Y1 = fake_data[0:400]
Y2 = fake_data[400:800]    
#print("Y1",Y1)
#print("Y2",Y2.size)
'''
X1_pre_digital_point = np.zeros(800)
for i in range(400):
    if X1[i].real > 0 and X1[i].imag > 0:
        X1_pre_digital_point[2*i] = 0
        X1_pre_digital_point[2*i+1] = 0
    elif X1[i].real > 0 and X1[i].imag < 0:
        X1_pre_digital_point[2*i] = 1
        X1_pre_digital_point[2*i+1] = 0
    elif (X1[i].real < 0 and X1[i].imag > 0):
        X1_pre_digital_point[2*i] = 0
        X1_pre_digital_point[2*i+1] = 1
    elif (X1[i].real < 0 and X1[i].imag < 0):
        X1_pre_digital_point[2*i] = 1
        X1_pre_digital_point[2*i+1] = 1    
print("X1_pre-point",X1_pre_digital_point)
Y1_re_digital_point = np.zeros(800)
#print("X1",X1.size)
for i in range(400):
    if (Y1[i].real > 0 and Y1[i].imag > 0):
        Y1_re_digital_point[2*i] = 0
        Y1_re_digital_point[2*i+1] = 0
    elif (Y1[i].real > 0 and Y1[i].imag < 0):
        Y1_re_digital_point[2*i] = 1
        Y1_re_digital_point[2*i+1] = 0
    elif (Y1[i].real < 0 and Y1[i].imag > 0):
        Y1_re_digital_point[2*i] = 0
        Y1_re_digital_point[2*i+1] = 1
    elif (Y1[i].real < 0 and Y1[i].imag < 0):
        Y1_re_digital_point[2*i] = 1
        Y1_re_digital_point[2*i+1] = 1    
print("Y1_re-point",Y1_re_digital_point)
corrected1 = 0
for j in range(400):    
    if X1_pre_digital_point[2*j] == Y1_re_digital_point[2*j] and  X1_pre_digital_point[2*j+1] == Y1_re_digital_point[2*j+1]:
        corrected1 = corrected1+1
print("corrected1:",corrected1)    

H1 = Y1/X1
#print("H1",H1)
X = Y2/H1
print("X",X)
print("X2",X2)
'''
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

H1 = Y1_2/X1  #这个H包含了发射机接收机前端的相偏、信道的相偏及噪声

for k in range(Y2.size//100):  #将40万个点分为4000组100个点
    data_phase2 = np.zeros(100)
    data_phase2 = Y2[k*100:(k+1)*100]
    #print("sample_phase",sample_phase)
    #在200000个样本点中每100个样本点加一次0到99π/50的循环 
    Y2[k*100:(k+1)*100] = data_phase2 / (np.cos(sample_phase)+1j * np.sin(sample_phase))
  
Y2_1 = Y2 / (np.cos(fake_Tphase)+1j * np.sin(fake_Tphase))
Y2_2 = Y2_1 / (np.cos(fake_Rphase)+1j * np.sin(fake_Rphase))  
 
X = Y2_2/H1  #此为第二个sample信道均衡后还原的信号
print("X",X)
print("X2",X2)
'''
H1 = Y1/X1
#print("H1",H1)
X = Y2/H1


X2_pre_digital_point = np.zeros(800)
for i in range(400):
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
print("X_pre-point",X2_pre_digital_point)
X2_re_digital_point = np.zeros(800)
#print("X1",X1.size)
for i in range(400):
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
print("X_re-point",X2_re_digital_point)
corrected = 0
for j in range(400):    
    if X2_pre_digital_point[2*j] == X2_re_digital_point[2*j] and  X2_pre_digital_point[2*j+1] == X2_re_digital_point[2*j+1]:
        corrected = corrected+1
print("corrected:",corrected)    

#=================================================================================================#
#画Y和Y'星座图
plt.scatter(X.real[0:400],X.imag[0:400])
plt.savefig('fX.png')
plt.close()
plt.scatter(X2.real[0:400],X2.imag[0:400])
plt.savefig('fX2.png')
plt.close()    


    