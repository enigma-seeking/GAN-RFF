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
# print(model)
# 定义损失函数
loss_fc = nn.BCELoss()
# 采用随机梯度下降SGD
optimizer = opt.Adam(params=model.parameters(), lr=0.001)
# 记录每次的损失值
loss_list = []
# 记录训练次数
x = []
start_time = time.time()
train_data = pd.DataFrame(np.load("train_data.npy"))

for i in range(150):
    # 每次随机读取30条数据
    batch_data = train_data.sample(n=100, replace=False)
    # 标签值
    batch_y = torch.from_numpy(batch_data.iloc[:, 0].values).float()
    batch_x = torch.from_numpy(batch_data.iloc[:, 1::].values).float() \
        .view(-1, 1, 30, 30)
    batch_y = batch_y.cuda()
    batch_x = batch_x.cuda()
    # 图片信息，一条数据784维将其转化为通道数为1，大小28*28的图片。
    # 1.前向传播c
    prediction = model.forward(batch_x)
    # print('Prediction value is :', prediction)
    # print('Y value is :', batch_y)
    # 2.计算损失值
    loss = loss_fc(prediction, batch_y)
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新权重
    optimizer.step()
    print("第%d次训练，loss为%.3f" % (i, loss))
    loss_list.append(loss)
    x.append(i)
end_time = time.time()
print('Time cost', end_time - start_time, 's')
# 保存模型参数
torch.save(model.state_dict(), 'train_receiver_Net.pkl')
print("已保存模型")
plt.figure('图1')
# 可以将损失值进行绘制
plt.plot(x, loss_list, "r-")
plt.savefig('train_Loss.png')
plt.show()
print("结束")