import numpy as np
import numpy.matlib
from math import pi
import math
import cmath
import copy
import GAN_data_maker as process
import matplotlib.pyplot as plt
from GAN import NetG
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 只能单GPU运行
print("是否使用GPU加速:")
print(torch.cuda.is_available())
H = np.load('../H_ME.npy')
def random_point():
    #定义采样点长度
    size = 45000    #100个sample（为同一发射机发射的样本）
    #生成0-3离散均匀随机序列&定义相位偏移
    a = np.random.randint(0, 4, size)   #生成4000000个采样样本点，0,1,2,3
    #相位选择向量，00对应π/4；01对应3π/4；11对应5π/4；10对应7π/4
    phase_map = ([1*pi/4,3*pi/4,5*pi/4,7*pi/4])
    phase = np.zeros(size)
    #data = np.zeros(size)
    #对应相位与样本点
    for i in range(size):
        t = a[i]
        phase[i] = phase_map[t]
    data = np.cos(phase) + 1j*np.sin(phase)
    return data

def data_process(data,ft,fr,IQtran_phase,Talpha,IQrec_phase,Ralpha,SNR):
    data = (1 + Talpha) * data.real * np.exp(1j * IQtran_phase) + 1j * data.imag * np.exp(-1j * IQtran_phase)
    #信道影响
    # ah = np.random.randn(1)  # 信道的实部
    # bh = np.random.randn(1)  # 信道的虚部
    # H = (1 / cmath.sqrt(2)) * (ah + 1j * bh)
    # H = -0.13176909 - 0.94339763j
    data = data * H
    ##（四）高斯白噪声N ##
    # 信噪比为30dB
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
def data_process_noR(data,ft,IQtran_phase,Talpha,SNR):
    data = (1 + Talpha) * data.real * np.exp(1j * IQtran_phase) + 1j * data.imag * np.exp(-1j * IQtran_phase)
    #信道影响
    ah = np.random.randn(1)  # 信道的实部
    bh = np.random.randn(1)  # 信道的虚部
    H = (1 / cmath.sqrt(2)) * (ah + 1j * bh)
    # H = -0.13176909 - 0.94339763j
    data = data * H
    ##（四）高斯白噪声N ##
    # 信噪比为30dB
    snr = 10 ** (SNR / 10.0)  # 比例SNR与dB为单位的snr之间的转换
    an = np.random.randn(data.size) / cmath.sqrt(2 * snr)
    bn = np.random.randn(data.size) / cmath.sqrt(2 * snr)
    N = an + 1j * bn
    data = data + N  # 每个样本点上都加上不一样的高斯白噪声
    # print(data)

    # 接收端
    # K1 = ((1 + Ralpha) * np.exp(-1j * IQrec_phase) + np.exp(1j * IQrec_phase)) / 2
    # K2 = ((1 + Ralpha) * np.exp(1j * IQrec_phase) - np.exp(-1j * IQrec_phase)) / 2
    # I = K1 * data
    # Q = K2 * (data.conjugate())
    sample_phase = np.zeros(450)
    for i in range(450):  # 产生与每组内100个点做循环加相偏的100个相位
        sample_phase[i] = (2 * i * pi) * (ft)  # 样本点相位偏移0到99π/f
    for k in range(data.size // 450):  # 将40万个点分为4000组100个点
        # data_phaseI = I[k * 450:(k + 1) * 450]
        # data_phaseQ = Q[k * 450:(k + 1) * 450]
        # 在200000个样本点中每100个样本点加一次0到99π/50的循环
        data_phase = data[k * 450:(k + 1) * 450]
        data[k * 450:(k + 1) * 450] = data_phase * np.exp(1j * sample_phase)
    ##同步误差
    data = data * np.exp(1j * (pi * np.random.randint(-10, 10) / 180))


    return data
def data_process_noR_noTimingError(data,ft,IQtran_phase,Talpha,SNR):
    data = (1 + Talpha) * data.real * np.exp(1j * IQtran_phase) + 1j * data.imag * np.exp(-1j * IQtran_phase)
    #信道影响
    ah = np.random.randn(1)  # 信道的实部
    bh = np.random.randn(1)  # 信道的虚部
    H = (1 / cmath.sqrt(2)) * (ah + 1j * bh)
    # H = -0.13176909 - 0.94339763j
    data = data * H
    ##（四）高斯白噪声N ##
    # 信噪比为30dB
    snr = 10 ** (SNR / 10.0)  # 比例SNR与dB为单位的snr之间的转换
    an = np.random.randn(data.size) / cmath.sqrt(2 * snr)
    bn = np.random.randn(data.size) / cmath.sqrt(2 * snr)
    N = an + 1j * bn
    data = data + N  # 每个样本点上都加上不一样的高斯白噪声
    # print(data)

    # 接收端
    # K1 = ((1 + Ralpha) * np.exp(-1j * IQrec_phase) + np.exp(1j * IQrec_phase)) / 2
    # K2 = ((1 + Ralpha) * np.exp(1j * IQrec_phase) - np.exp(-1j * IQrec_phase)) / 2
    # I = K1 * data
    # Q = K2 * (data.conjugate())
    sample_phase = np.zeros(450)
    for i in range(450):  # 产生与每组内100个点做循环加相偏的100个相位
        sample_phase[i] = (2 * i * pi) * (ft)  # 样本点相位偏移0到99π/f
    for k in range(data.size // 450):        # 将40万个点分为4000组100个点
        # data_phaseI = I[k * 450:(k + 1) * 450]
        # data_phaseQ = Q[k * 450:(k + 1) * 450]
        # 在200000个样本点中每100个样本点加一次0到99π/50的循环
        data_phase = data[k * 450:(k + 1) * 450]
        data[k * 450:(k + 1) * 450] = data_phase * np.exp(1j * sample_phase)
    return data
def data_process_noR_noH(data,ft,IQtran_phase,Talpha,SNR):
    data = (1 + Talpha) * data.real * np.exp(1j * IQtran_phase) + 1j * data.imag * np.exp(-1j * IQtran_phase)

    sample_phase = np.zeros(450)
    for i in range(450):  # 产生与每组内100个点做循环加相偏的100个相位
        sample_phase[i] = (2 * i * pi) * (ft)  # 样本点相位偏移0到99π/f
    for k in range(data.size // 450):  # 将40万个点分为4000组100个点
        # data_phaseI = I[k * 450:(k + 1) * 450]
        # data_phaseQ = Q[k * 450:(k + 1) * 450]
        # 在200000个样本点中每100个样本点加一次0到99π/50的循环
        data_phase = data[k * 450:(k + 1) * 450]
        data[k * 450:(k + 1) * 450] = data_phase * np.exp(1j * sample_phase)
    return data
def data_process_noR_AWGN(data,ft,IQtran_phase,Talpha,SNR):
    data = (1 + Talpha) * data.real * np.exp(1j * IQtran_phase) + 1j * data.imag * np.exp(-1j * IQtran_phase)
    #信道影响

    ##（四）高斯白噪声N ##
    # 信噪比为30dB
    snr = 10 ** (SNR / 10.0)  # 比例SNR与dB为单位的snr之间的转换
    an = np.random.randn(data.size) / cmath.sqrt(2 * snr)
    bn = np.random.randn(data.size) / cmath.sqrt(2 * snr)
    N = an + 1j * bn
    data = data + N  # 每个样本点上都加上不一样的高斯白噪声

    sample_phase = np.zeros(450)
    for i in range(450):  # 产生与每组内100个点做循环加相偏的100个相位
        sample_phase[i] = (2 * i * pi) * (ft)  # 样本点相位偏移0到99π/f
    for k in range(data.size // 450):  # 将40万个点分为4000组100个点
        # data_phaseI = I[k * 450:(k + 1) * 450]
        # data_phaseQ = Q[k * 450:(k + 1) * 450]
        # 在200000个样本点中每100个样本点加一次0到99π/50的循环
        data_phase = data[k * 450:(k + 1) * 450]
        data[k * 450:(k + 1) * 450] = data_phase * np.exp(1j * sample_phase)
    return data

def mapping(dataR,dataAR,dataGR,name,SNR):
    R = plt.scatter(dataR.real,dataR.imag,c = 'b',marker = 'x',label="R")
    AR =  plt.scatter(dataAR.real, dataAR.imag,c = 'g',marker = 'v',label="AR")
    G_AR = plt.scatter(dataGR.real, dataGR.imag,c = 'r',marker = 'o',label="G_AR")
    # font1 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 23,
    #          }
    # myfont = mpl.font_manager.FontProperties(fname='./times.ttf')
    # plt.legend(loc='upper right',handles=[R,AR,G_AR], prop=myfont)
    plt.xticks(fontsize = 8.5)
    plt.yticks(fontsize = 8.5)
    # plt.xlim(-1.6, 1.6)
    # plt.ylim(-1.6, 1.6)
    #plt.title(u"准确率变化对比图")

    # font2 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 10,
    #          }
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.legend()
    plt.savefig(name+SNR+'.png')
    plt.show()


def Normalization1(data):
    modulus1 = np.zeros(data.size)
    for k in range(data.size):
        modulus1[k] = abs(data[k]) ** 2
    pw1 = modulus1.mean()
    data = data / cmath.sqrt(pw1)
    return data


IQreal_Tphase = 1 * pi / 12  # 合法发射机
IQfake_Tphase = 1 * pi / 6  # 非法发射机
IQreal_Rphase = -1 * pi / 12  # 合法接收机
IQfake_Rphase = -1 * pi / 6  # 非法接收机
real_Talpha = 0.1
fake_Talpha = -0.1
real_Ralpha = 0.05
fake_Ralpha = -0.05


net_G = NetG.Net_G().to(device)
net_G.load_state_dict(torch.load("../GAN/GAN_G.pkl", map_location=device))
##GR##
fake_pre_data = np.load("fDATA_rand_point.npy")  # GAN使用假数据的QPSK信号点与合法接收机判别器D使用的假数据信号点一致
print("fake_pre_data", fake_pre_data)
X1 = fake_pre_data[0:450]
fake_pre_data = process.data_union(fake_pre_data)
fake_pre_data = fake_pre_data.reshape(10000,900)
input = torch.FloatTensor(fake_pre_data[0:100, :]).reshape(100, 1, -1, 900).to(device)  # 只过第一个batch的sample
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
X_bar1 = Normalization1(out_data_later)

# 1.生成真样本和假样本未进行数据处理时的采样点
######下面都是无块衰落的
  # 合法发射机样本随机点，未经过射频前端
  # 非法发射机样本随机点，未经过射频前端
rDATA_rand_point = Normalization1(random_point())
fDATA_rand_point = Normalization1(random_point())
#
mapping(rDATA_rand_point,fDATA_rand_point,X_bar1, '真假原始信号和经过G输出信号','0dB')
r = Normalization1(data_process(rDATA_rand_point,1/30,1/50,IQreal_Tphase,real_Talpha, IQreal_Rphase,real_Ralpha,30))
f = Normalization1(data_process(fDATA_rand_point,1/20,1/50,IQfake_Tphase,fake_Talpha, IQreal_Rphase,real_Ralpha,30))
G = Normalization1(data_process(X_bar1,1/20,1/50,IQfake_Tphase,fake_Talpha, IQreal_Rphase,real_Ralpha,30))
# Y1 = data_process(X_bar1,1/20,IQfake_Tphase,fake_Talpha)
mapping(r,f,G,'全影响','30dB' )


#####
#无接收机指纹
r = Normalization1(data_process_noR(rDATA_rand_point,1/30,IQreal_Tphase,real_Talpha,30))
f = Normalization1(data_process_noR(fDATA_rand_point,1/20,IQfake_Tphase,fake_Talpha,30))
G = Normalization1(data_process_noR(X_bar1,1/20,IQfake_Tphase,fake_Talpha,30))
mapping(r,f,G,'无接收机指纹', '30dB')


#无接收机指纹,无定时误差
r = Normalization1(data_process_noR_noTimingError(rDATA_rand_point,1/30,IQreal_Tphase,real_Talpha,30))
f = Normalization1(data_process_noR_noTimingError(fDATA_rand_point,1/20,IQfake_Tphase,fake_Talpha,30))
G = Normalization1(data_process_noR_noTimingError(X_bar1,1/20,IQfake_Tphase,fake_Talpha,30))
mapping(r,f,G,'无接收机指纹无定时误差', '30dB')
# mapping(f,'无接收机指纹无定时误差AR-H', '30dB')

#无接收机指纹,无信道影响
r = Normalization1(data_process_noR_noH(rDATA_rand_point,1/30,IQreal_Tphase,real_Talpha,30))
f = Normalization1(data_process_noR_noH(fDATA_rand_point,1/20,IQfake_Tphase,fake_Talpha,30))
G = Normalization1(data_process_noR_noH(X_bar1,1/20,IQfake_Tphase,fake_Talpha,30))
mapping(r,f,G,'无接收机指纹无信道影响', '30dB')
# mapping(f,'无接收机指纹无信道影响AR', '30dB')

#无接收机指纹,AWGN信道
r = Normalization1(data_process_noR_AWGN(rDATA_rand_point,1/30,IQreal_Tphase,real_Talpha,30))
f = Normalization1(data_process_noR_AWGN(fDATA_rand_point,1/20,IQfake_Tphase,fake_Talpha,30))
G = Normalization1(data_process_noR_AWGN(X_bar1,1/20,IQfake_Tphase,fake_Talpha,30))
mapping(r,f,G,'无接收机指纹AWGN信道', '30dB')
# mapping(f,'无接收机指纹AWGN-AR', '30dB')