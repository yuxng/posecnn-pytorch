#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/************************************************************
 backproject depth to 3D points
*************************************************************/

std::vector<at::Tensor> backproject_cuda_forward(
    float fx, float fy, float px, float py,
    at::Tensor depth);

std::vector<at::Tensor> backproject_forward(
    float fx, float fy, float px, float py,
    at::Tensor depth)
{
  CHECK_INPUT(depth);

  return backproject_cuda_forward(fx, fy, px, py, depth);
}


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
 roi align layer
*************************************************************/

at::Tensor ROIAlign_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int sampling_ratio);

at::Tensor ROIAlign_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width,
                                  const int batch_size,
                                  const int channels,
                                  const int height,
                                  const int width,
                                  const int sampling_ratio);

// Interface for Python
at::Tensor ROIAlign_forward(const at::Tensor& input,
                            const at::Tensor& rois,
                            const float spatial_scale,
                            const int pooled_height,
                            const int pooled_width,
                            const int sampling_ratio) 
{
  return ROIAlign_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
}

at::Tensor ROIAlign_backward(const at::Tensor& grad,
                             const at::Tensor& rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             const int batch_size,
                             const int channels,
                             const int height,
                             const int width,
                             const int sampling_ratio)
{
  return ROIAlign_backward_cuda(grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio);
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


/************************************************************
 sdf matching loss layer
*************************************************************/

std::vector<at::Tensor> sdf_loss_cuda_forward(
    at::Tensor pose_delta,
    at::Tensor pose_init,
    at::Tensor sdf_grids,
    at::Tensor sdf_limits,
    at::Tensor points,
    at::Tensor regularization);

std::vector<at::Tensor> sdf_loss_cuda_backward(
    at::Tensor grad_loss,
    at::Tensor bottom_diff);

std::vector<at::Tensor> sdf_loss_forward(
    at::Tensor pose_delta,
    at::Tensor pose_init,
    at::Tensor sdf_grids,
    at::Tensor sdf_limits,
    at::Tensor points,
    at::Tensor regularization)
{
  CHECK_INPUT(pose_delta);
  CHECK_INPUT(pose_init);
  CHECK_INPUT(sdf_grids);
  CHECK_INPUT(sdf_limits);
  CHECK_INPUT(points);
  CHECK_INPUT(regularization);

  return sdf_loss_cuda_forward(pose_delta, pose_init, sdf_grids, sdf_limits, points, regularization);
}

std::vector<at::Tensor> sdf_loss_backward(
    at::Tensor grad_loss,
    at::Tensor bottom_diff) 
{
  CHECK_INPUT(grad_loss);
  CHECK_INPUT(bottom_diff);

  return sdf_loss_cuda_backward(grad_loss, bottom_diff);
}


/************************************************************
 correlation layer
*************************************************************/

int correlation_forward_cuda_kernel(at::Tensor& output,
    int ob,
    int oc,
    int oh,
    int ow,
    int osb,
    int osc,
    int osh,
    int osw,

    at::Tensor& input1,
    int ic,
    int ih,
    int iw,
    int isb,
    int isc,
    int ish,
    int isw,

    at::Tensor& input2,
    int gc,
    int gsb,
    int gsc,
    int gsh,
    int gsw,

    at::Tensor& rInput1,
    at::Tensor& rInput2,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2,
    int corr_type_multiply,
    cudaStream_t stream);


int correlation_backward_cuda_kernel(   
    at::Tensor& gradOutput,
    int gob,
    int goc,
    int goh,
    int gow,
    int gosb,
    int gosc,
    int gosh,
    int gosw,

    at::Tensor& input1,
    int ic,
    int ih,
    int iw,
    int isb,
    int isc,
    int ish,
    int isw,

    at::Tensor& input2,
    int gsb,
    int gsc,
    int gsh,
    int gsw,

    at::Tensor& gradInput1, 
    int gisb,
    int gisc,
    int gish,
    int gisw,

    at::Tensor& gradInput2,
    int ggc,
    int ggsb,
    int ggsc,
    int ggsh,
    int ggsw,

    at::Tensor& rInput1,
    at::Tensor& rInput2,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2,
    int corr_type_multiply,
    cudaStream_t stream);

int correlation_forward_cuda(at::Tensor& input1, at::Tensor& input2, at::Tensor& rInput1, at::Tensor& rInput2, at::Tensor& output,
                       int pad_size,
                       int kernel_size,
                       int max_displacement,
                       int stride1,
                       int stride2,
                       int corr_type_multiply)
{
  int batchSize = input1.size(0);

  int nInputChannels = input1.size(1);
  int inputHeight = input1.size(2);
  int inputWidth = input1.size(3);

  int kernel_radius = (kernel_size - 1) / 2;
  int border_radius = kernel_radius + max_displacement;

  int paddedInputHeight = inputHeight + 2 * pad_size;
  int paddedInputWidth = inputWidth + 2 * pad_size;

  int nOutputChannels = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1);

  int outputHeight = ceil(static_cast<float>(paddedInputHeight - 2 * border_radius) / static_cast<float>(stride1));
  int outputwidth = ceil(static_cast<float>(paddedInputWidth - 2 * border_radius) / static_cast<float>(stride1));

  rInput1.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels});
  rInput2.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels});
  output.resize_({batchSize, nOutputChannels, outputHeight, outputwidth});

  rInput1.fill_(0);
  rInput2.fill_(0);
  output.fill_(0);

  int success = correlation_forward_cuda_kernel(
    output,
    output.size(0), 
    output.size(1),
    output.size(2),
    output.size(3),
    output.stride(0),
    output.stride(1),
    output.stride(2),
    output.stride(3),
    input1,
    input1.size(1),
    input1.size(2),
    input1.size(3),
    input1.stride(0),
    input1.stride(1),
    input1.stride(2),
    input1.stride(3),
    input2,
    input2.size(1),
    input2.stride(0),
    input2.stride(1),
    input2.stride(2),
    input2.stride(3),
    rInput1,
    rInput2,
    pad_size,     
    kernel_size,
    max_displacement,
    stride1,
    stride2,
    corr_type_multiply,
    at::cuda::getCurrentCUDAStream()
  );

  //check for errors
  if (!success) {
    AT_ERROR("CUDA call failed");
  }

  return 1;

}

int correlation_backward_cuda(at::Tensor& input1, at::Tensor& input2, at::Tensor& rInput1, at::Tensor& rInput2, at::Tensor& gradOutput, 
                       at::Tensor& gradInput1, at::Tensor& gradInput2,
                       int pad_size,
                       int kernel_size,
                       int max_displacement,
                       int stride1,
                       int stride2,
                       int corr_type_multiply)
{

  int batchSize = input1.size(0);
  int nInputChannels = input1.size(1);
  int paddedInputHeight = input1.size(2)+ 2 * pad_size;
  int paddedInputWidth = input1.size(3)+ 2 * pad_size;

  int height = input1.size(2);
  int width = input1.size(3);

  rInput1.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels});
  rInput2.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels});
  gradInput1.resize_({batchSize, nInputChannels, height, width});
  gradInput2.resize_({batchSize, nInputChannels, height, width});

  rInput1.fill_(0);
  rInput2.fill_(0);
  gradInput1.fill_(0);
  gradInput2.fill_(0);

  int success = correlation_backward_cuda_kernel(gradOutput,
                                                gradOutput.size(0),
                                                gradOutput.size(1),
                                                gradOutput.size(2),
                                                gradOutput.size(3),
                                                gradOutput.stride(0),
                                                gradOutput.stride(1),
                                                gradOutput.stride(2),
                                                gradOutput.stride(3),
                                                input1,
                                                input1.size(1),
                                                input1.size(2),
                                                input1.size(3),
                                                input1.stride(0),
                                                input1.stride(1),
                                                input1.stride(2),
                                                input1.stride(3),
                                                input2,  
                                                input2.stride(0),
                                                input2.stride(1),
                                                input2.stride(2),
                                                input2.stride(3),
                                                gradInput1,
                                                gradInput1.stride(0),
                                                gradInput1.stride(1),
                                                gradInput1.stride(2),
                                                gradInput1.stride(3),
                                                gradInput2,
                                                gradInput2.size(1),
                                                gradInput2.stride(0),
                                                gradInput2.stride(1),
                                                gradInput2.stride(2),
                                                gradInput2.stride(3),
                                                rInput1,
                                                rInput2,
                                                pad_size,
                                                kernel_size,
                                                max_displacement,
                                                stride1, 
                                                stride2,
                                                corr_type_multiply,
                                                at::cuda::getCurrentCUDAStream()
                                               );

  if (!success) {
    AT_ERROR("CUDA call failed");
  }

  return 1;
}

/********* python interface ***********/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("correlation_forward", &correlation_forward_cuda, "Correlation forward (CUDA)");
  m.def("correlation_backward", &correlation_backward_cuda, "Correlation backward (CUDA)");
  m.def("backproject_forward", &backproject_forward, "backproject forward (CUDA)");
  m.def("hard_label_forward", &hard_label_forward, "hard_label forward (CUDA)");
  m.def("hard_label_backward", &hard_label_backward, "hard_label backward (CUDA)");
  m.def("hough_voting_forward", &hough_voting_forward, "hough_voting forward (CUDA)");
  m.def("roi_pool_forward", &roi_pool_forward, "roi_pool forward (CUDA)");
  m.def("roi_pool_backward", &roi_pool_backward, "roi_pool backward (CUDA)");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("pml_forward", &pml_forward, "pml forward (CUDA)");
  m.def("pml_backward", &pml_backward, "pml backward (CUDA)");
  m.def("sdf_loss_forward", &sdf_loss_forward, "SDF loss forward (CUDA)");
  m.def("sdf_loss_backward", &sdf_loss_backward, "SDF loss backward (CUDA)");
}
