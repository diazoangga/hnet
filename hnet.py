import torch
import torch.nn as nn

class HNet(nn.Module):
    def __init__(self, in_channel):
        super(HNet, self).__init__()
        self.conv1 = conv_build([in_channel, 16])
        self.pool1 = conv_build('M')
        self.conv2 = conv_build([16, 32])
        self.pool2 = conv_build('M')
        self.conv3 = conv_build([32, 64])
        self.pool3 = conv_build('M')
        self.fc1 = conv_build(['FCN', 128, 1024])
        self.fcn = nn.Linear(1024, 6)
    
    def forward(self, input):
        c1 = self.conv1(input)
        print('Layer 1: {}'.format(c1.shape))
        c2 = self.pool1(c1)
        print('Layer 2: {}'.format(c2.shape))
        c3 = self.conv2(c2)
        print('Layer 3: {}'.format(c3.shape))
        c4 = self.pool2(c3)
        print('Layer 4: {}'.format(c4.shape))
        c5 = self.conv3(c4)
        print('Layer 5: {}'.format(c5.shape))
        c6 = self.pool3(c5)
        c6 = torch.flatten(c6, start_dim=2)
        print('Layer 6: {}'.format(c6.shape))
        c7 = self.fc1(c6)
        print('Layer 7: {}'.format(c7.shape))
        c8 = self.fcn(c7)
        print('Layer 8: {}'.format(c8.shape))

        return c8
    
class conv_build(nn.Module):
    def __init__(self, arch):
        super(conv_build, self).__init__()
        if arch == 'M':
            self.conv = nn.MaxPool2d(2, stride=2)
        elif arch[0] == 'FCN':
            self.conv = nn.Sequential(
                nn.Linear(arch[1], arch[2], bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(arch[0], arch[1], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(arch[1]),
                nn.ReLU(),
                nn.Conv2d(arch[1], arch[1], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(arch[1]),
                nn.ReLU()
            )

    def forward(self, input):
        return(self.conv(input))

a = HNet(1)
A = torch.zeros(1, 1, 128, 64)
a.forward(A)