
from typing import Union, Tuple, Type, Optional
import math

import torch
from torch import Tensor
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.functional import conv2d

from brevitas.function.ops import max_int
from brevitas.function.ops_ste import ceil_ste
from brevitas.quant_tensor import QuantTensor
from brevitas.inject.defaults import Int8WeightPerTensorFloat
from .quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from .quant_layer import WeightQuantType, BiasQuantType, ActQuantType
import sys
sys.path.append('depth_adder/')
import depth_adder, adder

__all__ = ['QuantDepthAdd2d']



class QuantDepthAdd2d(QuantWBIOL, depth_adder.DepthAdder2D):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = None,
            padding_type: str = 'standard',
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs) -> None:
        depth_adder.DepthAdder2D.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        QuantWBIOL.__init__(
            self,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        #assert self.padding_mode == 'zeros'
        assert not (padding_type == 'same' and padding != 0)
        self.padding_type = padding_type

    @property
    def per_elem_ops(self):
        flat_kernel_size = self.kernel_size[0] * self.kernel_size[1]
        return 2 * flat_kernel_size * (self.input_channels // self.groups)

    @property
    def output_channel_dim(self):
        if self.transposed:
            raise RuntimeError("Transposed kernels not supported")
        return 0

    @property
    def channelwise_separable(self) -> bool:
        return self.groups == self.out_channels

    def DAdd2D_zeros_pad(self, x: Tensor, weight: Tensor, bias: Tensor):
        out = depth_adder.DepthAdder2DFunction.apply(x, weight, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.weight_bits, self.quantize_v)
        #print('-------------------DAdd2d_zeros_pad----------------')
        #print(out)
        return out

    def DAdd2D_same_zeros_pad(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]):
        ih, iw = x.size()[-2:]
        kh, kw = weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        out = depth_adder.DepthAdder2DFunction.apply(x, weight, bias, self.stride, 0, self.dilation, self.groups,self.eta,self.quantize, self.weight_bits, self.quantize_v)
        return out

    def forward(self, input: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        return self.forward_impl(input)

    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        if self.padding_type == 'standard':
            return self.DAdd2D_zeros_pad(x, quant_weight, quant_bias)
        elif self.padding_type == 'same':
            return self.DAdd2D_same_zeros_pad(x, quant_weight, quant_bias)
        else:
            raise RuntimeError(f"Padding type {self.padding_type} not supported.")

    def max_acc_bit_width(self, input_bit_width: Tensor, weight_bit_width: Tensor):
        max_uint_input = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
        max_kernel_val = self.weight_quant.max_uint_value(weight_bit_width)
        group_size = self.out_channels // self.groups
        kernel_size = self.kernel_size * self.kernel_size
        max_uint_output = max_uint_input * max_kernel_val * kernel_size * group_size
        max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
        return max_output_bit_width
