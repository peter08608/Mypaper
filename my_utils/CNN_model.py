import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        '''
        (in_channels輸入卷積層的圖片通道數,
        out_channels輸出的通道數,
        kernel_size輸出的通道數卷積核的大小，長寬相等5*5,
        stride滑動步長爲1,
        padding在輸入張量周圍補的邊)
        '''
        #公式:((原圖的寬或高-kernelSize+2*padding)/stride+1)/poolKernelSize
        '''
        self.conv1 = nn.Conv2d(3,128,kernel_size=3,stride=4,padding=0)
        self.conv2 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0)
        self.conv3 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0)
        self.fc1 = nn.Linear(131072, 4096)
        self.fc2 = nn.Linear(4096, 4070)
        '''
        self.conv1 = torch.nn.Sequential( #(300,300)
            torch.nn.Conv2d(3, 32, (5,5), 1, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(5)
        )
        self.conv2 = torch.nn.Sequential( #(60,60)
            torch.nn.Conv2d(32, 64, (4,4), 1, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(4)
        )
        self.conv3 = torch.nn.Sequential( #(15,15)
            torch.nn.Conv2d(64, 128, (3,3), 1, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3)
        )
        self.conv4 = torch.nn.Sequential( #(5,5)
            torch.nn.Conv2d(128, 64, (3,3), 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        #(9,16)
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 2 * 2, 2000)
        )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(2000, 1)
        )


    def forward(self, x):
        '''
        angle_out = F.max_pool2d(self.conv1(x), 3, stride=2)
        angle_out = F.max_pool2d(self.conv2(angle_out), 2, stride =1)
        angle_out = self.conv3(angle_out)
        angle_out = angle_out.view(angle_out.size(0), -1)
        angle_out = F.relu(self.fc1(angle_out))
        angle_out = F.relu(self.fc2(angle_out))
        angle_out = angle_out.view(-1, 1, 55, 74)

        distance_out = F.max_pool2d(self.conv1(x), 3, stride=2)
        distance_out = F.max_pool2d(self.conv2(distance_out), 2, stride =1)
        distance_out = self.conv3(distance_out)
        distance_out = distance_out.view(distance_out.size(0), -1)
        distance_out = F.relu(self.fc1(distance_out))
        distance_out = F.relu(self.fc2(distance_out))
        distance_out = distance_out.view(-1, 1, 55, 74)
        '''
        #####CNN two head#####
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        
        res = conv4_out.view(conv4_out.size(0), -1)
        
        angle_out = self.dense(res)
        angle_out = self.dense1(angle_out)

        distance_out = self.dense(res)
        distance_out = self.dense1(distance_out)
        
        #print(conv4_out.size())
        #print("-->{}".format(res.size()))

        return angle_out, distance_out