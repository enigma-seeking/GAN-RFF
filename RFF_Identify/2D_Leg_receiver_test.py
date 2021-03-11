import torch
import torch.nn as nn
import torch.optim as opt
from RFF_Identify import TwoD_Net
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 只能单GPU运行
if(torch.cuda.is_available()):
    print("Use GPU")
else:
    print("Use CPU")
model = TwoD_Net.Net()
model.cuda()
print(model)
model.load_state_dict(torch.load('train_receiver_Net.pkl'))
# 定义损失函数
loss_fc = nn.BCELoss()
# 采用随机梯度下降SGD
# optimizer = opt.Adam(params=model.parameters(), lr=0.01)
# 记录每次的损失值
loss_list = []
# 记录训练次数
x = []
start_time = time.time()
train_data = pd.DataFrame(np.load("test_data.npy"))
test_correct = 0
for i in range(1):
    # 每次随机读取30条数据
    batch_data = train_data.sample(n=3500, replace=False)
    # 标签值
    batch_y = torch.from_numpy(batch_data.iloc[:, 0].values).float()
    batch_x = torch.from_numpy(batch_data.iloc[:, 1::].values).float() \
        .view(-1, 1, 30, 30)
    batch_y = batch_y.cuda()
    batch_x = batch_x.cuda()
    # 图片信息，一条数据784维将其转化为通道数为1，大小28*28的图片。
    # 1.前向传播c
    pred = (model(batch_x) > 0.5).float()
    test_correct += pred.eq(batch_y.view_as(pred)).sum().item()
    accuracy = test_correct/3500
    # print(prediction)
    # print(batch_y)
    print("第%d组测试集，准确率为%.3f" % (i, accuracy))

print(accuracy)
if (accuracy > 0.6 and accuracy < 0.79):
    torch.save(model.state_dict(), 'Net_D.pkl')
    print("此模型可以作为D结构")

if (accuracy > 0.97):
    torch.save(model.state_dict(), 'Leg_Net.pkl')
    print("此模型已保存为识别网络")