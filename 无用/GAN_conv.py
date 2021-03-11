from math import pi
import numpy as np
import numpy.matlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from time import time
import GAN_data_maker as process
from complexLayers import ComplexConv2d
import cmath
from NetG import Net_G
from recogize2 import Net, Net_D
import complexLayers as cl
import complexFunctions as cf
start = time()
IQreal_Tphase = 1 * pi / 12  # 合法发射机
IQfake_Tphase = 1 * pi / 6  # 非法发射机
IQreal_Rphase = -1 * pi / 12  # 合法接收机
IQfake_Rphase = -1 * pi / 6  # 非法接收机
real_Talpha = 0.5
fake_Talpha = -0.5
real_Ralpha = 0.05
fake_Ralpha = -0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 只能单GPU运行
print("是否使用GPU加速:")
print(torch.cuda.is_available())

rDATA_GAN = np.load("rDATA_GAN.npy")
rDATA_GAN = process.data_union(rDATA_GAN)  # 因为rDATA_GAN维度是一，所以用data_union.
# 合法发射机数据，分成实数了
fake_pre_data = np.load("fDATA_rand_point.npy")  # GAN使用假数据的QPSK信号点与合法接收机判别器D使用的假数据信号点一致

# SNR = 30
# snr = 10 ** (SNR / 10.0)  # 比例SNR与dB为单位的snr之间的转换
# an = np.random.randn(fake_pre_data.size) / cmath.sqrt(2 * snr)
# bn = np.random.randn(fake_pre_data.size) / cmath.sqrt(2 * snr)
# N = an + 1j * bn



fake_pre_data = process.data_union(fake_pre_data)  # 划分为实数虚数排在一起


# print("r_DATA_GAN", rDATA_GAN)
# print("size，800万", rDATA_GAN.size)


'''
ah2 = np.random.randn(1)  # 信道的实部
bh2 = np.random.randn(1)  # 信道的虚部
H_ME = (1 / cmath.sqrt(2)) * (ah2 + 1j * bh2)  # 非法发射机与合法接收机瑞利信道
'''
#H_ME = np.load('H_ME.npy') # H_ME是频响函数
H=0

# 切片，将判别器中合法的输入和生成器的输入调整成适合网络输入的维度

# 划分成合适的维度


def data_splite(data, train_rate, test_rate, eval_rate, Select_fun='Train'):
    size = int(data.size)
    if (Select_fun == 'Train'):
        select_data = data[0:int(size * train_rate)]  # 切片，从开头取到size*train_rate
    elif (Select_fun == 'Test'):
        select_data = data[int(size * train_rate):int(size * (train_rate + test_rate))]
    elif (Select_fun == 'Eval'):
        select_data = data[int(size * (1 - eval_rate)):size]
    return select_data
# 按照我的理解是这样的，这个data是一个大列表，存有训练数据，测试数据，评估数据。而我之前已经按比例存进去了，自己是知道比例的。
# 设计哪部分就改变Select_fun


class MyDataset(Dataset):
    def __init__(self, label, data, transform=None):  # 传入信号的生成数据，标签自己生成
        # 存入的数据中标签部分为传入的data
        super(MyDataset, self).__init__()
        self.data = data
        self.label = label
        self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        signal = self.data[800 * index:800 * index + 800]
        target = self.label[index]
        return signal, target


train_r_data_input = data_splite(rDATA_GAN, 0.8, 0.1, 0.1, Select_fun='Train')
# train_r_data_input = train_data_real.reshape((int(10000*0.8), 800)) #真的输入的
eval_r_data_input = data_splite(rDATA_GAN, 0.8, 0.1, 0.1, Select_fun='Eval')
test_r_data_input = data_splite(rDATA_GAN, 0.8, 0.1, 0.1, Select_fun='Test')

train_data_fake = data_splite(fake_pre_data, 0.8, 0.1, 0.1, Select_fun='Train')
eval_data_fake = data_splite(fake_pre_data, 0.8, 0.1, 0.1, Select_fun='Eval')
test_data_fake = data_splite(fake_pre_data, 0.8, 0.1, 0.1, Select_fun='Test')

train_f_data_Ginput = train_data_fake.reshape((int(10000 * 0.8), 800))  # 假的训练集
# 和论文对上了，一共收集真样本10000条，每条400个信号采样点。化为实数虚数就是10000*800，训练验证测试8：1：1
# 维度为8000*800  总元素个数不变，就是将矩阵维度变了。但为什么变目前不知道
eval_f_data_Ginput = eval_data_fake.reshape((int(10000 * 0.1), 800))  # 假的验证集
test_f_data_Ginput = test_data_fake.reshape((int(10000 * 0.1), 800))  # 假的测试集

# real_label = np.ones(10000)
# dataset
train_real_dataset = MyDataset(data=train_r_data_input, label=np.zeros(8000), transform=transforms.ToTensor())  #修改
eval_real_dataset = MyDataset(data=eval_r_data_input, label=np.zeros(1000), transform=transforms.ToTensor())
test_real_dataset = MyDataset(data=test_r_data_input, label=np.zeros(1000), transform=transforms.ToTensor())
# 将真实数据label设为一。
# dataloader
train_real_dataloader = DataLoader(train_real_dataset, batch_size=100, shuffle=True, num_workers=8)
eval_real_dataloader = DataLoader(eval_real_dataset, batch_size=100, shuffle=True, num_workers=8)
test_real_dataloader = DataLoader(test_real_dataset, batch_size=100, shuffle=True, num_workers=8)
# 8线程输入，神经网络100行一捆处理，使数据更有独立性，将输入数据打乱，所以shuffle=True

net_G = Net_G().to(device)
net_D = Net_D().to(device)
#  初始化D和G  ##

# torch.nn.init.normal(net_`, mean=0, std=1)

lr_D = 1e-3  # 学习率
lr_G = 1e-7#学习率 1e-9可以达到平衡，但是效果不好
#5e-7欺骗率71%
#双侧标签平滑更容易稳定，但是欺骗效果不好。
#单侧标签差一些，但是效果好一些。
optimizer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)  # 生成器优化器  class torch.nn.Parameter()返回模型所有参数的迭代器
optimizer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)  # 判别器优化器

adversarial_loss = torch.nn.BCELoss()  # 二元交叉熵损失函数

epoch_num = 15
batch_num = 80  # 80 * 100 个 sample

# 初始化权重
# net_G._initialize_weights()
net_D.load_state_dict(torch.load("0.748Leg_receiver_net.pkl", map_location=device))  # 自己导入模型的结构信息，但是这个map_location不太明白



# net_D.load_state_dict(torch.load("0.9905Leg_receiver_net.pkl", map_location=device))
# 产生更新D使用的经过G之后的fake_data_later
# 开始训练
# D_loss= 1.5
# g_loss= 0.7

for epoch in range(epoch_num):  # 多少个epoch
    for t, (data,real_label) in enumerate(train_real_dataloader):  # t为batch_number数 //？
        # enumerate 遍历并给出下标和数据
        # 非法的标签
        #print(real_label.numpy().size)
        fake_label = Variable(torch.full((real_label.cpu().numpy().size, 1), 1.0, requires_grad=False)).to(device)
        # 为什么要false，不计算梯度，不优化？
        real_label = real_label.to(device)
        input = torch.FloatTensor(train_f_data_Ginput[t * 100:(t + 1) * 100, :]).reshape(100, 1, -1, 800).to(
            device)  # 第一维是sample，第二维是每个sample的800个点
        # 是对input维度进行调整，是一个100行，800列的。列数据代表的是一次样本点的数据，100行是因为批处理设定的是100
        # print("input",input.shape)
        outxr, outxi = net_G(input)
        # print("outxr",outxr.shape)
        # print("outxi",outxi.shape)
        outxr = outxr.detach().cpu().numpy()
        # ？？不确定，为什么要改，这个出来也是张量形式，改成numpy为了存储?
        # 修改数据类型  CUDA tensor在这里应该是指变量而不是张量格式的数据改成numpy
        outxi = outxi.detach().cpu().numpy()
        xr = outxr.reshape(40000)  # 变成一个一维 400*100
        xi = outxi.reshape(40000)
        out_oneline = np.zeros(80000)  # 100个sample，每个sample800个点
        for i in range(40000):
            out_oneline[2 * i] = xr[i]  # 按行out输入到一行的out_oneline中
            out_oneline[2 * i + 1] = xi[i]
        out_data_later = process.data_to_complex_line(out_oneline)  # 重新转化为复数
        out_data_later = process.data_process(out_data_later, 1/20)

        # pi/40 是非法接收机的载波偏移。
        out_data_later = process.data_union(out_data_later)
        out_data_later = out_data_later.reshape(100, 1, -1, 800)
        out_data_later = torch.FloatTensor(out_data_later).to(device)  # 假数据的输入到GAN里的D
        # 按格式存储数据？

        # 训练D
        # while(D_loss-g_loss > 0.7):
        for i in range(1):
            # for i in range(10):
            optimizer_D.zero_grad()
            # 评估D的判别能力
            data = data.reshape(100, 1, -1, 800).float().to(device)
            real_loss = adversarial_loss(net_D(data), real_label)
            # adversarial_loss = torch.nn.BCELoss() #二元交叉熵损失函数
            if (t+1) % 80 == 0:
                print("epoch : {}/{},real_loss : {} ".format(epoch + 1, epoch_num, real_loss))
                # GD训练完一轮了。
            real_loss.backward()
            # print(out_data_later.shape)
            # print(fake_label)
            # print(fake_label.shape)
            fake_loss = adversarial_loss(net_D(out_data_later), fake_label)  # 对于D来说，希望假数据的标签为0
            fake_loss.backward()
            if (t+1) % 80 == 0:
                print("epoch : {}/{},fake_loss : {}".format(epoch + 1, epoch_num, fake_loss))
            D_loss = real_loss + fake_loss
            if (t+1) % 80 == 0:
                print("epoch : {}/{},D_loss : {}".format(epoch + 1, epoch_num, D_loss))

            optimizer_D.step()

        # 开始更新G
        for i in range(3):
            optimizer_G.zero_grad()
            g_loss = adversarial_loss(net_D(out_data_later), real_label)  # 对于G来说，希望是假数据变成标签为1
            g_loss.backward()  # 反向传播
            optimizer_G.step()  # 优化器单次优化G
            if (t+1) % 80 == 0:
                print("epoch : {}/{},G_loss : {}".format(epoch + 1, epoch_num, g_loss))
    # print("t",t)
    # 验证集

# 测试集
test_correct = 0
test_correct_rate = 0
for t_test, (data, real_label) in enumerate(test_real_dataloader):
    fake_label = Variable(torch.full((real_label.numpy().size, 1), 1.0, requires_grad=False)).to(device)
    input = torch.FloatTensor(test_f_data_Ginput[t_test * 100:(t_test + 1) * 100, :]).reshape(100, 1, -1, 800).to(
        device)
    outxr, outxi = net_G(input)
    outxr = outxr.detach().cpu().numpy()
    outxi = outxi.detach().cpu().numpy()
    xr = outxr.reshape(40000)
    xi = outxi.reshape(40000)
    out_oneline = np.zeros(80000)  # 100个sample，每个sample800个点
    for i in range(40000):
        out_oneline[2 * i] = xr[i]  # 按行out输入到一行的out_oneline中
        out_oneline[2 * i + 1] = xi[i]
        # print("out_oneline.shape",out_oneline.shape)
    out_data_later = process.data_to_complex_line(out_oneline)
    out_data_later = process.data_process(out_data_later, 1/20)

    out_data_later = process.data_union(out_data_later)
    out_data_later = out_data_later.reshape(100 ,1, -1, 800)
    out_data_later = torch.FloatTensor(out_data_later).to(device)  # fDATA_rand_point测试集部分过训练好的G得到GAN里D的测试输入
    output = net_D(out_data_later)
    # if t%9 == 0:
    # print("output",output)
    pred = (output < 0.5).cpu().float()  # output＞0.5输出1，＜0.5输出0
    # if t%9 == 0:
    # print("pred",pred)
    test_correct += pred.eq(torch.ones((100, 1)).view_as(pred)).sum().item()  # 假数据输出1的数
    # eq()的作用是找出pred中和torch.ones相等的 使用sum() 统计相等的个数：
    test_correct_rate = test_correct / len(test_real_dataset)
# print("t_tset",t_test)
print("test_acc = {}".format(test_correct / len(test_real_dataset)))  # 骗过GAN里的D


# 引入合法接收机网络
net = Net().to(device)

net.load_state_dict(torch.load("0.997Leg_receiver_net.pkl", map_location=device))
Leg_test_correct = 0
Leg_test_correct_rate = 0
for t_Legtest, (data, real_label) in enumerate(test_real_dataloader):
    fake_label = Variable(torch.full((real_label.numpy().size, 1), 1.0, requires_grad=False)).to(device)
    input = torch.FloatTensor(test_f_data_Ginput[t_Legtest * 100:(t_Legtest + 1) * 100, :]).reshape(100, 1, -1, 800).to(
        device)
    outxr, outxi = net_G(input)
    outxr = outxr.detach().cpu().numpy()
    outxi = outxi.detach().cpu().numpy()
    xr = outxr.reshape(40000)
    xi = outxi.reshape(40000)
    out_oneline = np.zeros(80000)  # 100个sample，每个sample800个点
    for i in range(40000):
        out_oneline[2 * i] = xr[i]  # 按行out输入到一行的out_oneline中
        out_oneline[2 * i + 1] = xi[i]
        # print("out_oneline.shape",out_oneline.shape)
    out_data_later = process.data_to_complex_line(out_oneline)

    #H_MA = np.load("H_MA.npy")
    out_data_later = process.data_process(out_data_later, 1/20)
    out_data_later = process.data_union(out_data_later)
    out_data_later = out_data_later.reshape(100 ,1, -1, 800)
    out_data_later = torch.FloatTensor(out_data_later).to(device)  # 假的输入
    output = net(out_data_later)
    # if t_Legtest%9 == 0:
    # print("output",output)
    pred = (output < 0.5).cpu().float()
    # if t_Legtest%9 == 0:
    # print("pred",pred)
    Leg_test_correct += pred.eq(torch.ones((100, 1)).view_as(pred)).sum().item()
    Leg_test_correct_rate = Leg_test_correct / len(test_real_dataset)  # 骗过合法接收机的D
# print("t_Legtest",t_Legtest)
print("GAN_test_acc = {}".format(test_correct / len(test_real_dataset)))  # 骗过GAN里的D
print("Leg_test_acc = {}".format(Leg_test_correct / len(test_real_dataset)))  # 骗过合法接收机的D

print(time() - start)
if (Leg_test_correct_rate > 0):
    torch.save(net_G.state_dict(), '{}net_G_cov.pkl'.format(Leg_test_correct_rate))
    torch.save(net_D.state_dict(), '{}net_D_cov.pkl'.format(Leg_test_correct_rate))
    # np.save("H_ME",H_ME)
