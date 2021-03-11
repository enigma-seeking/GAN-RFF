import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_D(nn.Module):

    def __init__(self):
        super(Net_D, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1),
                                   nn.BatchNorm2d(6),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=2, stride=1),
                                   nn.BatchNorm2d(12),
                                   )
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=1, stride=1)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1),
                                   nn.BatchNorm2d(12),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=2, stride=1),
                                   nn.BatchNorm2d(12),
                                   )
        # self.conv4 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=7, stride=1)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=9, stride=1)
        self.BN1 = nn.BatchNorm2d(3)
        self.RELU = nn.ReLU()
        self.BN2 = nn.BatchNorm2d(6)
        self.mp = nn.MaxPool2d(kernel_size=2)
        # self.fc1 = nn.Linear(in_features=4 * 4 * 16, out_features=128)
        self.fc1 = nn.Linear(in_features=7 * 7 * 16, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

        # self.dropout = nn.Dropout(p=0.09)

    def forward(self, input):
        conv0_output = self.conv0(input)
        input = conv0_output
        conv1_output = self.conv1(input)
        data = conv1_output + conv0_output
        Intermediate_data = self.RELU(data)

        Intermediate_data2 = self.mp(Intermediate_data)

        conv4_output = self.conv4(Intermediate_data2)
        conv6_output = conv4_output.view(-1, 7 * 7 * 16)
        fc1_output = self.RELU(self.fc1(conv6_output))
        # fc1_output = self.dropout(fc1_output)
        fc2_output = self.RELU(self.fc2(fc1_output))
        # fc2_output = self.dropout(fc2_output)
        fc3_output = F.sigmoid(self.fc3(fc2_output))
        return fc3_output