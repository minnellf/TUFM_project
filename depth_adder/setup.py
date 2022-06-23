from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
from torch.utils.cpp_extension import CppExtension
import os

if "CUDA_HOME" in os.environ:
    setup(
        name='depth_adder',
        ext_modules=[
            CUDAExtension('depth_adder_cuda', [
                'depth_adder_cuda.cpp',
                'depth_adder_cuda_kernel.cu',
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        })

else:
    setup(
        name='depth_adder',
        ext_modules=[
            CppExtension('depth_adder_cpp', [
                'depth_adder.cpp',
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
