from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

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
