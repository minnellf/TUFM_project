
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.append('../depth_adder/')
import adder
sys.path.append('../depth_adder/')
import depth_adder
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantAvgPool2d
from brevitas.nn import QuantAdd2d , QuantDepthAdd2d



from brevitas.quant import IntBias

from common import CommonIntActQuant, CommonUintActQuant
from common import CommonIntWeightPerChannelQuant, CommonIntWeightPerTensorQuant

FIRST_LAYER_BIT_WIDTH = 8

def conv1x1(in_planes, out_planes, stride=1):
    " 1x1 convolution "
    wbit=16
    abit=32
#a
    return QuantAdd2d(in_planes, out_planes, kernel_size=1,stride=stride, padding=0, bias=None)

def conv3x3_nodw(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    wbit=16
    abit=32
#a
    return QuantAdd2d(in_planes, out_planes, kernel_size=3,stride=stride, padding=1, bias=None)

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    wbit=16
    abit=32
    return QuantDepthAdd2d(in_planes, in_planes, kernel_size=3,stride=stride,padding=1, bias=None)

def convdw(in_planes, out_planes, stride=1):
            return nn.Sequential(
                conv3x3(in_planes,1),
 
                nn.BatchNorm2d(in_planes),
                QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=8,
            return_quant_tensor=True),
                conv1x1(in_planes, out_planes),

                #nn.BatchNorm2d(out_planes),
                #nn.ReLU6(inplace=True),
            )
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if stride==1:
          self.conv1 = convdw(inplanes, planes)
        else:
          self.conv1 = conv3x3_nodw(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=8,
            return_quant_tensor=True)
        self.conv2 = convdw(planes, planes)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.wbit=16
        self.abit=32
        #self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = QuantConv2d(3, 16, kernel_size=3, stride=1, padding=1,output_quant=Int8ActPerTensorFloat, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=8,
            return_quant_tensor=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = QuantAvgPool2d(kernel_size=8, stride=1)   
        self.fc = QuantConv2d(64 * block.expansion, num_classes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
#a
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QuantAdd2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=None),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.fc(x)

        x = self.bn2(x)

        return x.view(x.size(0), -1)


def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)

print('Summary\n')
from torchinfo import summary

model = resnet20().cuda()

summary(model,input_size=(1,3,32,32))    
