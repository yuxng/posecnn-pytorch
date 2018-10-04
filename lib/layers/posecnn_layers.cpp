#include <torch/torch.h>

#include <vector>

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/************************************************************
 hard_label layer
*************************************************************/
std::vector<at::Tensor> hard_label_cuda_forward(
    float threshold,
    at::Tensor bottom_prob,
    at::Tensor bottom_label);

std::vector<at::Tensor> hard_label_cuda_backward(
    at::Tensor top_diff);

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


/************************************************************
 hough voting layer
*************************************************************/
std::vector<at::Tensor> hough_voting_cuda_forward(
    at::Tensor bottom_label,
    at::Tensor bottom_verex,
    at::Tensor meta_data,
    at::Tensor extents,
    int is_train,
    int skip_pixels,
    int labelThreshold,
    float inlierThreshold,
    float votingThreshold,
    float perThreshold);

std::vector<at::Tensor> hough_voting_forward(
    at::Tensor bottom_label,
    at::Tensor bottom_vertex,
    at::Tensor meta_data,
    at::Tensor extents,
    int is_train,
    int skip_pixels,
    int labelThreshold,
    float inlierThreshold,
    float votingThreshold,
    float perThreshold)
{
  CHECK_INPUT(bottom_label);
  CHECK_INPUT(bottom_vertex);
  CHECK_INPUT(extents);
  CHECK_INPUT(meta_data);

  return hough_voting_cuda_forward(bottom_label, bottom_vertex, meta_data, extents, 
    is_train, skip_pixels, labelThreshold, inlierThreshold, votingThreshold, perThreshold);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hard_label_forward", &hard_label_forward, "hard_label forward (CUDA)");
  m.def("hard_label_backward", &hard_label_backward, "hard_label backward (CUDA)");
  m.def("hough_voting_forward", &hough_voting_forward, "hough_voting forward (CUDA)");
}
