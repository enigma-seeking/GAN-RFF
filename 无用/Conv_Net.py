import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import complexLayers as cl
# import complexFunctions as cf
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),  #
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1), # 28*28*3
            nn.BatchNorm2d(3),
            nn.ReLU(True),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),#24*24*6
            nn.BatchNorm2d(6),
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),  #12*12*6
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=9, stride=1),  # 4*4*16
            # nn.BatchNorm2d(16),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 16, out_features=128)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=32)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=32, out_features=1)
        )

        # self.dropout = nn.Dropout(p=0.09)


######
#残差块
        conv2_output = self.conv2(Intermediate_data2)
        input = conv2_output
        conv3_output = self.conv3(input)
        data = conv2_output + conv3_output
        Intermediate_data = self.RELU(data)
        conv4_output = self.conv4(Intermediate_data)
        Intermediate_data2 = self.mp(conv4_output)

        conv6_output = Intermediate_data2.view(-1, 4 * 4 * 16)