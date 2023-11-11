'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        relu_input = out.clone().detach()
        out = F.relu(self.bn1(out))
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out, relu_input


class ResNetUpdate(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNetUpdate, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1_1 = block(self.in_planes, 64, 1)
        self.in_planes = 64 * block.expansion
        self.layer1_2 = block(self.in_planes, 64, 1)
        self.in_planes = 64 * block.expansion
        self.layer2_1 = block(self.in_planes, 128, 2)
        self.in_planes = 128 * block.expansion
        self.layer2_2 = block(self.in_planes, 128, 1)
        self.in_planes = 128 * block.expansion
        self.layer3_1 = block(self.in_planes, 256, 2)
        self.in_planes = 256 * block.expansion
        self.layer3_2 = block(self.in_planes, 256, 1)
        self.in_planes = 256 * block.expansion
        self.layer4_1 = block(self.in_planes, 512, 2)
        self.in_planes = 512 * block.expansion
        self.layer4_2 = block(self.in_planes, 512, 1)
        self.in_planes = 512 * block.expansion

        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        print("--------")
        print(len(layers))
        # print(type(*layers))
        # print(type(nn.Sequential(*layers)))
        return nn.Sequential(*layers)

    def forward(self, x):
        self.relu_input_list = []
        out = self.conv1(x)
        self.relu_input_list.append(out.detach().clone())
        out = F.relu(self.bn1(out))
        out, relu_input  = self.layer1_1(out)
        self.relu_input_list.append(relu_input.detach().clone())
        out, relu_input = self.layer1_2(out)
        self.relu_input_list.append(relu_input.detach().clone())
        out, relu_input = self.layer2_1(out)
        self.relu_input_list.append(relu_input.detach().clone())
        out, relu_input = self.layer2_2(out)
        self.relu_input_list.append(relu_input.detach().clone())
        out, relu_input = self.layer3_1(out)
        self.relu_input_list.append(relu_input.detach().clone())
        out, relu_input = self.layer3_2(out)
        self.relu_input_list.append(relu_input.detach().clone())
        out, relu_input = self.layer4_1(out)
        self.relu_input_list.append(relu_input.detach().clone())
        out, relu_input = self.layer4_2(out)
        self.relu_input_list.append(relu_input.detach().clone())

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18Update(num_classes = 10):
    return ResNetUpdate(BasicBlock, num_classes)


def test():
    seed_ = 70
    torch.manual_seed(seed_)
    torch.backends.cudnn.deterministic = True
    net = ResNet18Update()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    print(y)
    print(len(net.relu_input_list))


# test()
