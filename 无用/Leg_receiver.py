import numpy as np 
from math import pi  
import math
import cmath
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils

## GPU加速 ##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 只能单GPU运行
if(torch.cuda.is_available()):
    print("Use GPU")
else:
    print("Use CPU")
 
## （一）数据载入  ##
data = np.load("30_Leg_DATA_input.npy") 

## （二）数据自动切割对齐标签  ##
b = np.ones(10000)     #真样本标签10000
c = np.zeros(10000)    #假样本标签10000
label = np.concatenate((b,c))  #设置标签20000
#print(label.size)

#切分训练集、验证集、测试集，比例为8:1:1  #
def data_splite(data,label,train_rate,test_rate,eval_rate,Select_fun = 'Train'):
    size = int(data.size/2)
    label_size = int(label.size/2)
    if (Select_fun == 'Train'):
        select_data = np.concatenate((data[0:int(size*train_rate)],data[size:int(size*(1+train_rate))]))
        select_label = np.concatenate((label[0:int(label_size*train_rate)],label[label_size:int(label_size*(1+train_rate))]))
    elif (Select_fun == 'Test'):
        select_data = np.concatenate((data[int(size*train_rate):int(size*(train_rate+test_rate))],data[int(size*(1+train_rate)):int(size*(1+train_rate+test_rate))]))
        select_label = np.concatenate((label[int(label_size*train_rate):int(label_size*(train_rate+test_rate))],label[int(label_size*(1+train_rate)):int(label_size*(1+train_rate+test_rate))]))
    elif (Select_fun == 'Eval'):
        select_data = np.concatenate((data[int(size*(1 - eval_rate)):size],data[int(size*(2 - eval_rate)):2*size]))
        select_label = np.concatenate((label[int(label_size*(1 - eval_rate)):label_size],label[int(label_size*(2 - eval_rate)):2*label_size]))
    return select_data,select_label
    
class MyDataset(Dataset):
    def __init__(self,data,label,transform = None): #传入信号的生成数据，标签自己生成
        #存入的数据中标签部分为传入的data
        super(MyDataset,self).__init__()
        self.data = data
        self.label = label
        self.transform = transform
        #self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        signal = self.data[800*index:800*index+800]
        target = self.label[index]
        return signal,target

train_data,train_label = data_splite(data,label,train_rate = 0.8,eval_rate = 0.1,test_rate = 0.1,Select_fun = 'Train')
test_data,test_label = data_splite(data,label,train_rate = 0.8,eval_rate = 0.1,test_rate = 0.1,Select_fun = 'Test')
eval_data,eval_label = data_splite(data,label,train_rate = 0.8,eval_rate = 0.1,test_rate = 0.1,Select_fun = 'Eval')
#dataset
train_dataset = MyDataset(data = train_data,label = train_label,transform = transforms.Compose([transforms.Resize(4,100),transforms.ToTensor()]))
test_dataset = MyDataset(data = test_data,label = test_label,transform = transforms.Compose([transforms.Resize(4,100),transforms.ToTensor()]))
eval_dataset = MyDataset(data = eval_data,label = eval_label,transform = transforms.Compose([transforms.Resize(4,100),transforms.ToTensor()]))
#dataloader
batch_size = 100
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True,num_workers = 8)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=True,num_workers = 8)
eval_loader = DataLoader(dataset = eval_dataset, batch_size = batch_size, shuffle=True,num_workers = 8)

#testloader_size = int(test_label.size)
#print("test_loader_size:",int(test_label.size))


## （三）合法接收机分类器的网络架构 ##
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(800, 200)
        self.fc2 = nn.Linear(200, 100)
        self.dropout = torch.nn.Dropout(p=0.6)
        # self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # x = F.relu(self.fc3(x))
        #x = F.softmax(self.fc4(x),dim=0)
        x = F.sigmoid(self.fc4(x))
        return x
net = Net().to(device)

## （四） 训练网络  ##
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

#net.load_state_dict(torch.load("0.953Leg_receiver_net.pkl", map_location=device))

num_epoch = 15
loss = 0
criterion = nn.BCELoss()  #交叉熵函数
for epoch in range(num_epoch):
    batch_num = 0
    net.train()
    for signal,target in train_loader:
        batch_num = batch_num + 1
        #input = torch.tensor(signal, dtype=torch.float32).to(device)
        input = signal.float().to(device)
        target = target.to(device)
        out = net(input)
        loss = criterion(out,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(batch_num)%160 == 0:
            print("Epoch[{}/{}],loss:{:.6f}".format(epoch+1,num_epoch,loss.data))
    #print("batch_num:",batch_num)
    #查看验证集的acc变换情况
    net.eval()
    eval_correct = 0
    for signal,target in eval_loader:
        input = signal.float().to(device)
        target = torch.FloatTensor(target).to(device)
        out = net(input)
        pred = (out>0.5).float()
        eval_correct += pred.eq(target.view_as(pred)).sum().item()
    print("Epoch[{}/{}],eval_acc = {:.4f}".format(epoch+1,num_epoch,eval_correct/len(eval_loader.dataset)))
test_correct = 0
for signal,target in test_loader:
    input = signal.float().to(device)
    target = torch.FloatTensor(target).to(device)
    out = net(input)
    pred = (out>0.5).float()
    test_correct += pred.eq(target.view_as(pred)).sum().item()
print("test_acc ={:.4f}".format(test_correct/len(test_loader.dataset)))   

corrected = test_correct/len(test_loader.dataset)
if corrected > 0.9 or corrected <0.85:
    torch.save(net.state_dict(), '{}Leg_receiver_net.pkl'.format(corrected))

