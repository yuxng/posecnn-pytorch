#include <torch/torch.h>

#include <vector>

// CUDA forward declarations
// hard_label
std::vector<at::Tensor> hard_label_cuda_forward(
    float threshold,
    at::Tensor bottom_prob,
    at::Tensor bottom_label);

std::vector<at::Tensor> hard_label_cuda_backward(
    at::Tensor top_diff);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> hard_label_forward(
    float threshold,
    at::Tensor bottom_prob,
    at::Tensor bottom_label)
{
  CHECK_INPUT(bottom_prob);
  CHECK_INPUT(bottom_label);

  return hard_label_cuda_forward(threshold, bottom_prob, bottom_label);
}

std::vector<at::Tensor> hard_label_backward(
    at::Tensor top_diff) {
  CHECK_INPUT(top_diff);

  return hard_label_cuda_backward(top_diff);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hard_label_forward", &hard_label_forward, "hard_label forward (CUDA)");
  m.def("hard_label_backward", &hard_label_backward, "hard_label backward (CUDA)");
}
