import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5_2(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5_2, self).__init__()
        # act = nn.Tanh
        act = nn.ReLU
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), stride=(1, 1), padding=1),
            act(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            act(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=60, kernel_size=(3, 3), stride=(1, 1), padding=1),
            act(),
            nn.MaxPool2d(kernel_size=2),

        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=480, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=n_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        # act = nn.Tanh
        # act = nn.ReLU
        act = nn.LeakyReLU
        ks = 3
        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(ks, ks), stride=(1, 1), padding=1),
        #     nn.BatchNorm2d(6),
        #     act(inplace=True),
        #     # nn.AvgPool2d(kernel_size=2),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(ks, ks), stride=(1, 1), padding=1),
        #     nn.BatchNorm2d(16),
        #     act(inplace=True),
        #     # nn.AvgPool2d(kernel_size=2),
        #     # nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(ks, ks), stride=(1, 1), padding=1),
        #     nn.BatchNorm2d(32),
        #     act(inplace=True),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(ks, ks), stride=(1, 1), padding=1),
        #     nn.BatchNorm2d(64),
        #     act(inplace=True)
        # )
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=18, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            nn.BatchNorm2d(18),
            act(inplace=True),
            # nn.AvgPool2d(kernel_size=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=18, out_channels=32, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            act(inplace=True),
            # nn.AvgPool2d(kernel_size=2),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            act(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            act(inplace=True)
        )

        self.classifier = nn.Sequential(
            # nn.Linear(in_features=1280, out_features=512),
            # nn.Linear(in_features=192, out_features=64),
            # nn.Tanh(),
            # act(inplace=True),
            # nn.Linear(in_features=84, out_features=n_classes),
            nn.Linear(in_features=384, out_features=n_classes),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


class MLP2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 32, bias=True),
            nn.ReLU(inplace=True)
        )

        # torch.nn.init.xavier_uniform_(self.layer1[1].weight)
        # torch.nn.init.zeros_(self.layer1[1].bias)

        self.layer2 = nn.Sequential(
            nn.Linear(32, output_dim, bias=True),
            # nn.ReLU(inplace=True)
        )
        # torch.nn.init.xavier_uniform_(self.layer2[0].weight)
        # torch.nn.init.zeros_(self.layer2[0].bias)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 20, 5)
        self.fc1 = nn.Linear(20 * 1 * 5, num_classes)

        # print(self.conv1.weight.shape)
        # print(self.fc1)

        # torch.nn.init.xavier_uniform_(self.conv1.weight)
        # torch.nn.init.xavier_uniform_(self.conv2.weight)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # torch.nn.init.zeros_(self.fc1.bias)

        # torch.nn.init.uniform_(self.conv1.weight, b=1)
        # torch.nn.init.uniform_(self.conv2.weight, b=1)
        # torch.nn.init.uniform_(self.fc1.weight, b=1)

        torch.nn.init.normal_(self.conv1.weight, std=1.)
        torch.nn.init.normal_(self.conv2.weight, std=1.)
        torch.nn.init.normal_(self.fc1.weight, std=1.)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 1 * 5)
        x = self.fc1(x)
        x = F.relu(x)
        return x


class LeNetCIFAR100(nn.Module):
    def __init__(self):
        super(LeNetCIFAR100, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=(5, 5), padding=5 // 2, stride=(2, 2)),
            act(),
            nn.Conv2d(12, 12, kernel_size=(5, 5), padding=5 // 2, stride=(2, 2)),
            act(),
            nn.Conv2d(12, 12, kernel_size=(5, 5), padding=5 // 2, stride=(1, 1)),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 100)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


class LeNetCIFAR1(nn.Module):
    def __init__(self, classes=2):
        super(LeNetCIFAR1, self).__init__()
        # act = nn.Sigmoid
        act = nn.LeakyReLU
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=(5, 5), padding=5 // 2, stride=(2, 2)),
            act(),  # 16 * 8 * 12
            nn.Conv2d(12, 12, kernel_size=(5, 5), padding=5 // 2, stride=(2, 2)),
            act(),  # 8 * 4 * 12
            nn.Conv2d(12, 12, kernel_size=(5, 5), padding=5 // 2, stride=(1, 1)),
            act(),  # 8 * 4 * 12
        )
        self.fc = nn.Sequential(
            nn.Linear(384, classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


class LeNetCIFAR2(nn.Module):
    def __init__(self, classes=2):
        super(LeNetCIFAR2, self).__init__()
        act = nn.ReLU
        padding_1 = 1
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=18, kernel_size=(5, 5), padding=padding_1, stride=(1, 1)),
            act(),  # 128 * 64 * 12
            nn.Conv2d(in_channels=18, out_channels=36, kernel_size=(5, 5), stride=(1, 1)),
            act(),  # 64 * 32 * 12
            nn.Conv2d(in_channels=36, out_channels=64, kernel_size=(5, 5), stride=(1, 1)),
            act(),  # 64 * 32 * 12
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1)),
            act(),  # 64 * 32 * 12
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc = nn.Sequential(
            nn.Linear(1152, classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class LeNetCIFAR3(nn.Module):
    def __init__(self, classes=2):
        super(LeNetCIFAR3, self).__init__()
        # act = nn.Sigmoid
        # act = nn.LeakyReLU
        act = nn.ReLU
        padding_1 = 1
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=18, kernel_size=(3, 3), padding=padding_1, stride=(1, 1)),
            act(),  # 128 * 64 * 12
            nn.Conv2d(in_channels=18, out_channels=36, kernel_size=(3, 3), stride=(1, 1)),
            act(),  # 64 * 32 * 12
            nn.Conv2d(in_channels=36, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            act(),  # 64 * 32 * 12
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            act(),  # 64 * 32 * 12

        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc = nn.Sequential(
            # nn.Linear(64 * 16 * 16, classes)
            nn.Linear(8320, 1000),
            # nn.Linear(10752, 1000),
            nn.Linear(1000, classes)

        )

    def forward(self, x):
        out = self.body(x)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        # print("out: ", out.size())
        out = self.fc(out)
        return out


class LeNetMNIST(nn.Module):
    def __init__(self, classes=2):
        super(LeNetMNIST, self).__init__()
        # act = nn.Sigmoid
        # act = nn.LeakyReLU
        act = nn.ReLU
        padding_2 = 5 // 2
        # padding_1 = 1
        # print("padding:", padding)
        self.body = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=(5, 5), padding=padding_2, stride=(2, 2)),
            act(),
            nn.Conv2d(12, 12, kernel_size=(5, 5), padding=padding_2, stride=(2, 2)),
            act(),
            nn.Conv2d(12, 12, kernel_size=(5, 5), padding=padding_2, stride=(1, 1)),
            act()
            # nn.Conv2d(12, 12, kernel_size=3, padding=padding_1, stride=1),
            # act()
        )
        self.fc = nn.Sequential(
            nn.Linear(336, classes)
            # nn.Linear(288, classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# 这是残差网络中的basicblock，实现的功能如下方解释：
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):  # inplanes代表输入通道数，planes代表输出通道数。
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4  # 输出通道数的倍乘

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=(3, 3), bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = self.dropout(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=1000):  # layers=参数列表 block选择不同的类
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))  # 每个blocks的第一个residual结构保存在layers列表中。
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))  # 该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)  # 将输出结果展成一行
#         x = self.fc(x)
#
#         return x


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class MyConvolutionalNetwork(nn.Module):
    def __init__(self, classes=2):
        super(MyConvolutionalNetwork, self).__init__()

        # 32 * 32 * 3 -> 32 * 32 * 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop2 = nn.Dropout(0.3)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop3 = nn.Dropout(0.4)

        # Size of the output of the last convolution:
        # self.flattened_size = 4 * 4 * 128
        self.flattened_size = 1024

        self.fc1 = nn.Linear(self.flattened_size, classes)

    def forward(self, x):
        """
        Forward pass,
        x shape is (batch_size, 3, 32, 32)
        (color channel first)
        in the comments, we omit the batch_size in the shape
        """
        # shape : 3x32x32 -> 32x32x32
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=0.001)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.001)
        x = self.bn2(x)

        x = self.maxpool1(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = F.leaky_relu(x, negative_slope=0.001)
        x = self.bn3(x)
        x = self.conv4(x)
        x = F.leaky_relu(x, negative_slope=0.001)
        x = self.bn4(x)

        x = self.maxpool2(x)
        x = self.drop2(x)

        x = self.conv5(x)
        x = F.leaky_relu(x, negative_slope=0.001)
        x = self.bn5(x)
        x = self.conv6(x)
        x = F.leaky_relu(x, negative_slope=0.001)
        x = self.bn6(x)

        x = self.maxpool3(x)
        x = self.drop3(x)

        # Check the output size
        output_size = np.prod(x.size()[1:])
        assert output_size == self.flattened_size, \
            "self.flattened_size is invalid {} != {}".format(output_size, self.flattened_size)

        # 128x4x4 -> 2048
        x = x.view(-1, self.flattened_size)
        # 2048 -> 10
        x = F.leaky_relu(self.fc1(x), negative_slope=0.001)

        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim),
            # nn.BatchNorm1d(100),
            # nn.ReLU(),
            # nn.Linear(in_features=100, out_features=100),
            # nn.BatchNorm1d(100),
            # nn.ReLU(),
            # nn.Linear(in_features=100, out_features=10)
        )

    def forward(self, x):
        return self.classifier(x)
