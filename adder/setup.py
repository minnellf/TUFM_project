from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
from torch.utils.cpp_extension import CppExtension
import os

if "CUDA_HOME" in os.environ:
    setup(
        name='adder',
        ext_modules=[
            CUDAExtension('adder_cuda', [
                'adder_cuda.cpp',
                'adder_cuda_kernel.cu',
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        })

else:
    setup(
        name='adder',
        ext_modules=[
            CppExtension('adder_cpp', [
                'adder.cpp',
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
