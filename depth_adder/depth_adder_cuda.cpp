#include <torch/torch.h>
#include <vector>

int depth_adder_cuda_forward(
    const at::Tensor &input,
    const at::Tensor &weight,
    // const at::Tensor &bias,
    at::Tensor &output,
    int KW, int KH,
    int SW, int SH,
    int PW, int PH);

int depth_adder_cuda_backward_grad_in(
    at::Tensor &grad_out,
    at::Tensor &input,
    at::Tensor &weight,
    at::Tensor &grad_in,
    int KW, int KH,
    int SW, int SH,
    int PW, int PH);

int depth_adder_cuda_backward_grad_weight(
    at::Tensor &grad_out,
    at::Tensor &input,
    at::Tensor &weight,
    at::Tensor &grad_weight,
    int KW, int KH,
    int SW, int SH,
    int PW, int PH);

#define CHECK_CUDA(x) AT_ASSERT((x).type().is_cuda(), #x "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT((x).type().is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA((x));   \
    CHECK_CONTIGUOUS((x))

int depth_adder_forward(
    const at::Tensor &input,
    const at::Tensor &weight,
    // const at::Tensor &bias,
    at::Tensor &output,
    int KW, int KH,
    int SW, int SH,
    int PW, int PH)
{
    // TODO: add checks checks
    return depth_adder_cuda_forward(
        input,
        weight,
        // bias,
        output,
        KW, KH,
        SW, SH,
        PW, PH);
}

int depth_adder_backward_input(
    at::Tensor &grad_out,
    at::Tensor &input,
    at::Tensor &weight,
    at::Tensor &grad_in,
    int KW, int KH,
    int SW, int SH,
    int PW, int PH)
{
    // TODO: add checks checks
    return depth_adder_cuda_backward_grad_in(
        grad_out,
        input,
        weight,
        grad_in,
        KW, KH,
        SW, SH,
        PW, PH);
}

int depth_adder_backward_weight(
    at::Tensor &grad_out,
    at::Tensor &input,
    at::Tensor &weight,
    at::Tensor &grad_weight,
    int KW, int KH,
    int SW, int SH,
    int PW, int PH)
{
    // TODO: add checks checks
    return depth_adder_cuda_backward_grad_weight(
        grad_out,
        input,
        weight,
        grad_weight,
        KW, KH,
        SW, SH,
        PW, PH);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &depth_adder_forward, "depth_adder forward (CUDA)");
    m.def("backward_input", &depth_adder_backward_input, "depth_adder backward input (CUDA)");
    m.def("backward_weight", &depth_adder_backward_weight, "depth_adder backward weight (CUDA)");
}
