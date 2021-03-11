# data_process


import numpy as np
import numpy.matlib
from math import pi
import math
import cmath
import copy




def data_process(data,ft,fr,IQtran_phase,Talpha,IQrec_phase,Ralpha):
    data = (1 + Talpha) * data.real * np.exp(1j * IQtran_phase) + 1j * data.imag * np.exp(-1j * IQtran_phase)
    #信道影响
    ah = np.random.randn(1)  # 信道的实部
    bh = np.random.randn(1)  # 信道的虚部
    H = (1 / cmath.sqrt(2)) * (ah + 1j * bh)
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
    sample_phase = np.zeros(450)
    for i in range(450):  # 产生与每组内100个点做循环加相偏的100个相位
        sample_phase[i] = (2 * i * pi) * (ft-fr)  # 样本点相位偏移0到99π/f
    for k in range(data.size // 450):  # 将40万个点分为4000组100个点
        data_phaseI = I[k * 450:(k + 1) * 450]
        data_phaseQ = Q[k * 450:(k + 1) * 450]
        # 在200000个样本点中每100个样本点加一次0到99π/50的循环
        data[k * 450:(k + 1) * 450] = data_phaseI * np.exp(1j * sample_phase) + data_phaseQ * np.exp(-1j * sample_phase)
    ##同步误差
    data = data * np.exp(1j * (pi * np.random.randint(-10, 10) / 180))


    return data


def data_union_D(DATA): #复数矩阵转化为实数矩阵
    DATA_real = DATA.real
    DATA_imag = DATA.imag
    size = 2 * DATA[0].size
    DATA_input = np.zeros((DATA.ndim,2*DATA[0].size))
    for i in range(DATA[0].size):
        DATA_input[:, i*2] = DATA_real[:, i]
        DATA_input[:, i*2+1] = DATA_imag[:, i]
    return DATA_input


def data_union(DATA): #一行的
    DATA_real = DATA.real
    DATA_imag = DATA.imag
    size = 2*DATA.size  #给定DATA大小，将大小变为两倍，将实数和虚数按序放好。
    DATA_input = np.zeros(size)
    for i in range(DATA.size):
        DATA_input[i*2] = DATA_real[i]
        DATA_input[i*2 + 1] = DATA_imag[i]
    return DATA_input

def data_to_complex(DATA): #将实数矩阵转化为复数矩阵
    size = int(DATA[0].size/2)
    DATA_complex = np.zeros((int(DATA.ndim), int(size)),dtype=complex)
    for i in range(size):
        DATA_complex[:, i] = DATA[:, 2*i] + 1j*DATA[:, 2*i+1]
    return DATA_complex

def data_to_complex_line(DATA): #将一行的实数转化为复数
    size = int(DATA.size/2)
    DATA_complex = np.zeros(int(size),dtype=complex)
    for i in range(size):
        DATA_complex[i] = DATA[2*i] + 1j*DATA[2*i+1]
    return DATA_complex

if __name__ == '__main__': 
    real_pre_data = np.load("rDATA_rand_point.npy")  # 10000个sample
    IQreal_Tphase = 1 * pi / 12  # 合法发射机
    IQfake_Tphase = 1 * pi / 6  # 非法发射机
    IQreal_Rphase = -1 * pi / 12  # 合法接收机
    IQfake_Rphase = -1 * pi / 6  # 非法接收机
    real_Talpha = 0.5
    fake_Talpha = -0.5
    real_Ralpha = 0.05
    fake_Ralpha = -0.05
    rDATA_GAN = []
    #H_BE = np.load("H_BE.npy")
    for i in range(100):  #将40万个点分为4000组100个点
        r = real_pre_data[i*45000:(i+1)*45000]
        r = data_process(r, 1/30, 1/40,IQreal_Tphase,real_Talpha,IQfake_Rphase,fake_Ralpha)  #合法发射机  H_BA
        rDATA_GAN = np.hstack((rDATA_GAN, r))

    print(rDATA_GAN.shape)
    np.save("rDATA_GAN", rDATA_GAN)   #还是复数，未经过拆分。这个数据集是到达真实数据到达攻击协作者的。



