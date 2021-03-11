## 本文件存储合法接收机需要的数据集。 ##
import numpy as np 
from math import pi
import cmath

from torch.utils.data import Dataset,DataLoader
from scipy.fftpack import fft,ifft
import copy

def random_point():
    #定义采样点长度
    size = 4500000    #10000个sample（为同一发射机发射的样本）
                     #分别划分为训练集16000sample（8000真8000假）
                     #验证集2000sample（1000真1000假）
                     #测试集2000sample（1000真1000假）
                     #（8000:1000:1000）
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

def data_process(data,ft,fr,IQtran_phase,Talpha,IQrec_phase,Ralpha, SNR):
    data = (1 + Talpha) * data.real * np.exp(1j * IQtran_phase) + 1j * data.imag * np.exp(-1j * IQtran_phase)
    #信道影响
    ah = np.random.randn(1)  # 信道的实部
    bh = np.random.randn(1)  # 信道的虚部
    H = (1 / cmath.sqrt(2)) * (ah + 1j * bh)
    # H = -0.13176909 - 0.94339763j
    data = data * H
    ##（四）高斯白噪声N ##

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


##  放入合法判别器时元数据是实部虚部交叉放置的  ##   
def data_union_D(DATA):
    #提取实部虚部，并组合成(1,2*data.size)的numpy
    DATA_real = DATA.real
    DATA_imag = DATA.imag
    size = 2*DATA.size
    DATA_input = np.zeros(size)
    for i in range(DATA.size):
        DATA_input[i*2] = DATA_real[i]
        DATA_input[i*2+1] = DATA_imag[i]
    #DATA_input = np.vstack((DATA_real,DATA_imag))  #将分开的实部虚部合并为一个numpy矩阵
    #print(DATA_input)  #（实部,虚部,实部,虚部,...,实部,虚部）
    return DATA_input


if __name__ == '__main__':
    IQreal_Tphase = 1*pi/12  #合法发射机
    IQfake_Tphase = 1*pi/6  #非法发射机
    IQreal_Rphase = -1*pi/12  #合法接收机
    IQfake_Rphase = -1*pi/6  #非法接收机
    real_Talpha = 0.1
    fake_Talpha = -0.1
    real_Ralpha = 0.05
    fake_Ralpha = -0.05
    #H_BA = np.load("H_BA.npy")
    #H_MA = np.load("H_MA.npy")

    #1.生成真样本和假样本未进行数据处理时的采样点
    
    rDATA_rand_point = random_point()  #合法发射机样本随机点，未经过射频前端
    fDATA_rand_point = random_point()  #非法发射机样本随机点，未经过射频前端
    np.save("rDATA_rand_point",rDATA_rand_point)  #10000sample真样本
    np.save("fDATA_rand_point",fDATA_rand_point)  #10000sample假样本

    SNR = 30
    rDATA = []
    fDATA = []
    #2.将采样点进行数据处理，产生分类器的真假样本输入
    for i in range(100):
        r = rDATA_rand_point[i*45000:(i+1)*45000]
        f = fDATA_rand_point[i*45000:(i+1)*45000]
        r = data_process(r,1/30,1/50,IQreal_Tphase,real_Talpha, IQreal_Rphase,real_Ralpha,SNR)  #合法发射机 --合法接收机
        f = data_process(f,1/20,1/50,IQfake_Tphase,fake_Talpha, IQreal_Rphase,real_Ralpha,SNR)  #非法发射机 --合法接收机   为了训练合法接收机的识别能力
        rDATA = np.hstack((rDATA,r))
        fDATA = np.hstack((fDATA,f))
    
    
    np.save("rDATA",rDATA)  #Y,未切分
    np.save("fDATA",fDATA)  #Y’,未切分
    rDATA_input = data_union_D(rDATA)
    print(rDATA_input.shape)
    rDATA_input = rDATA_input.reshape(10000,900)
    fDATA_input = data_union_D(fDATA)
    fDATA_input = fDATA_input.reshape(10000,900)
    real_lable = np.ones((10000,1))
    feak_lable = np.zeros((10000, 1))
    rDATA_input = np.hstack([real_lable, rDATA_input])
    fDATA_input = np.hstack([feak_lable, fDATA_input])
    #  将合法非法数据按列合并为一个输入，前面为合法， 后面为非法
    Leg_DATA_input = np.vstack([rDATA_input,fDATA_input])
    print(Leg_DATA_input.shape)
    train_data = np.concatenate((Leg_DATA_input[0:8000, :], Leg_DATA_input[10000:18000, :]), axis=0)
    test_data = np.concatenate((Leg_DATA_input[8000:10000, :], Leg_DATA_input[18000:20000,:]), axis=0)
    print(train_data.shape)
    print(test_data.shape)
    np.save("train_data", train_data)
    np.save("test_data", test_data)
    np.save("30_Leg_DATA_input",Leg_DATA_input)     #这套数据集是为了训练合法接收机的网络，里面是真数据和假数据经过独立的信道到达接收机



