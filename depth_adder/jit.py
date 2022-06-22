from torch.utils.cpp_extension import load

conv_cuda = load(
    'depth_adder_cuda', ['depth_adder_cuda.cpp', 'depth_adder_cuda_kernel.cu'], verbose=True)
help(depth_adder_cuda)
