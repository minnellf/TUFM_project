
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantAvgPool2d
from brevitas.nn import QuantAdd2d



from brevitas.quant import IntBias

from common import CommonIntActQuant, CommonUintActQuant
from common import CommonIntWeightPerChannelQuant, CommonIntWeightPerTensorQuant

from brevitas.nn import QuantConv2d
from brevitas.nn import QuantAdd2d
from brevitas.nn import QuantReLU
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantAvgPool2d
from brevitas.nn import QuantMaxPool2d
from brevitas.inject.defaults import Int8ActPerTensorFloat

def conv3x3(in_planes, out_planes, stride=1, quantize=False, weight_bits=8, sparsity=0, quantize_v='sbm'):
    " 3x3 convolution with padding "

    return QuantAdd2d(
        in_planes, 
        out_planes, 
        kernel_size=(3,3), 
        stride=stride, 
        padding=1,
        #input_quant=Int8ActPerTensorFloat,
        #output_quant=Int8ActPerTensorFloat,
        bias=None,)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, quantize=False, weight_bits=8, sparsity=0, quantize_v='sbm'):
        #print('stride ',stride)
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity, quantize_v=quantize_v)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = QuantReLU(
            inplace=True, bit_width=8, return_quant_tensor=True
        )
        self.conv2 = conv3x3(planes, planes, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity, quantize_v=quantize_v)
        self.bn2 = nn.BatchNorm2d(planes)
        self.id2 = QuantIdentity(bit_width=8)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # MOD: Adding identity layer to have quantization in after second batchnorm
        out = self.id2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            #print(out.shape)
            #print(residual.shape)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, quantize=False, weight_bits=8, sparsity=0, quantize_v='sbm'):
        super(ResNet, self).__init__()
        self.quantize = quantize
        self.weight_bits = weight_bits
        self.sparsity = sparsity
        self.quantize_v = quantize_v
        self.inplanes = 16
        self.conv1 = QuantConv2d(
            3, 
            16, 
            kernel_size=(3, 3), 
            stride=1, 
            padding=1, 
            #input_quant=Int8ActPerTensorFloat,
            #output_quant=Int8ActPerTensorFloat,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = QuantReLU(inplace=True, bit_width=8, return_quant_tensor=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.id = QuantIdentity(bit_width=8,return_quant_tensor = True)
        self.avgpool = QuantMaxPool2d(8, stride=1)
        # use conv as fc layer (addernet)
        self.fc = QuantConv2d(
            64 * block.expansion, 
            num_classes,
            kernel_size=(1, 1), 
            #input_quant=Int8ActPerTensorFloat,
            #output_quant=Int8ActPerTensorFloat,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_classes)


        # init (new add)
        for m in self.modules():
            # if isinstance(m, adder.Adder2D):
            #     nn.init.kaiming_normal_(m.adder, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
#A
        if stride != 1 or self.inplanes != planes * block.expansion:
            #print('ciao')
            downsample = nn.Sequential(
                QuantAdd2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=(1, 1), 
                    stride=stride, 
                    #input_quant=Int8ActPerTensorFloat,
                    #output_quant=Int8ActPerTensorFloat,
                    bias=None
                ), # adder.Adder2D
                nn.BatchNorm2d(planes * block.expansion),
                QuantIdentity(bit_width=8,return_quant_tensor = True)
            )

        layers = []
        layers.append(block(inplanes = self.inplanes, planes = planes, stride = stride, downsample = downsample,
                            quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity, quantize_v=self.quantize_v))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes = self.inplanes, planes = planes, quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity, quantize_v=self.quantize_v))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print('in 1  ',x.shape)
        x = self.layer1(x)
        #print('in 2  ',x.shape)
        x = self.layer2(x)
        #print('in 3  ',x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.id(x)
        x = self.avgpool(x)
        #print(x.shape)
        x = self.fc(x)
        #x = self.bn2(x)
        return x.view(x.size(0), -1)



def resnet20_add(num_classes=10, quantize=False, weight_bits=8, sparsity=0, quantize_v='sbm', **kwargs):
    print(quantize, sparsity)
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, quantize=quantize,
                    weight_bits=weight_bits, sparsity=sparsity, quantize_v=quantize_v)
