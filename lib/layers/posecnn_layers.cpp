#include <torch/torch.h>

#include <vector>

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/************************************************************
 hard label layer
*************************************************************/
std::vector<at::Tensor> hard_label_cuda_forward(
    float threshold,
    float sample_percentage,
    at::Tensor bottom_prob,
    at::Tensor bottom_label,
    at::Tensor bottom_rand);

std::vector<at::Tensor> hard_label_cuda_backward(
    at::Tensor top_diff);

std::vector<at::Tensor> hard_label_forward(
    float threshold,
    float sample_percentage,
    at::Tensor bottom_prob,
    at::Tensor bottom_label,
    at::Tensor bottom_rand)
{
  CHECK_INPUT(bottom_prob);
  CHECK_INPUT(bottom_label);
  CHECK_INPUT(bottom_rand);

  return hard_label_cuda_forward(threshold, sample_percentage, bottom_prob, bottom_label, bottom_rand);
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


/************************************************************
 roi pool layer
*************************************************************/
std::vector<at::Tensor> roi_pool_cuda_forward(
    int pooled_height,
    int pooled_width,
    float spatial_scale,
    at::Tensor bottom_features,
    at::Tensor bottom_rois);

std::vector<at::Tensor> roi_pool_cuda_backward(
    int batch_size,
    int height,
    int width,
    float spatial_scale,
    at::Tensor top_diff,
    at::Tensor bottom_rois,
    at::Tensor argmax_data);

std::vector<at::Tensor> roi_pool_forward(
    int pooled_height,
    int pooled_width,
    float spatial_scale,
    at::Tensor bottom_features,
    at::Tensor bottom_rois)
{
  CHECK_INPUT(bottom_features);
  CHECK_INPUT(bottom_rois);

  return roi_pool_cuda_forward(pooled_height, pooled_width, spatial_scale, bottom_features, bottom_rois);
}

std::vector<at::Tensor> roi_pool_backward(
    int batch_size,
    int height,
    int width,
    float spatial_scale,
    at::Tensor top_diff,
    at::Tensor bottom_rois,
    at::Tensor argmax_data) 
{
  CHECK_INPUT(top_diff);
  CHECK_INPUT(bottom_rois);
  CHECK_INPUT(argmax_data);

  return roi_pool_cuda_backward(batch_size, height, width, spatial_scale, top_diff, bottom_rois, argmax_data);
}

/************************************************************
 point matching loss layer
*************************************************************/

std::vector<at::Tensor> pml_cuda_forward(
    at::Tensor bottom_prediction,
    at::Tensor bottom_target,
    at::Tensor bottom_weight,
    at::Tensor points,
    at::Tensor symmetry,
    float hard_angle);

std::vector<at::Tensor> pml_cuda_backward(
    at::Tensor grad_loss,
    at::Tensor bottom_diff);

std::vector<at::Tensor> pml_forward(
    at::Tensor bottom_prediction,
    at::Tensor bottom_target,
    at::Tensor bottom_weight,
    at::Tensor points,
    at::Tensor symmetry,
    float hard_angle)
{
  CHECK_INPUT(bottom_prediction);
  CHECK_INPUT(bottom_target);
  CHECK_INPUT(bottom_weight);
  CHECK_INPUT(points);
  CHECK_INPUT(symmetry);

  return pml_cuda_forward(bottom_prediction, bottom_target, bottom_weight, points, symmetry, hard_angle);
}

std::vector<at::Tensor> pml_backward(
    at::Tensor grad_loss,
    at::Tensor bottom_diff) 
{
  CHECK_INPUT(grad_loss);
  CHECK_INPUT(bottom_diff);

  return pml_cuda_backward(grad_loss, bottom_diff);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hard_label_forward", &hard_label_forward, "hard_label forward (CUDA)");
  m.def("hard_label_backward", &hard_label_backward, "hard_label backward (CUDA)");
  m.def("hough_voting_forward", &hough_voting_forward, "hough_voting forward (CUDA)");
  m.def("roi_pool_forward", &roi_pool_forward, "roi_pool forward (CUDA)");
  m.def("roi_pool_backward", &roi_pool_backward, "roi_pool backward (CUDA)");
  m.def("pml_forward", &pml_forward, "pml forward (CUDA)");
  m.def("pml_backward", &pml_backward, "pml backward (CUDA)");
}
