#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void ROIAlignForward(const int nthreads, const float* bottom_data, const float spatial_scale, const int height, const int width,
                                const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, c, ph, pw) is an element in the aligned output
    // int n = index;
    // int pw = n % aligned_width;
    // n /= aligned_width;
    // int ph = n % aligned_height;
    // n /= aligned_height;
    // int c = n % channels;
    // n /= channels;

    int pw = index % aligned_width;
    int ph = (index / aligned_width) % aligned_height;
    int c  = (index / aligned_width / aligned_height) % channels;
    int n  = index / aligned_width / aligned_height / channels;

    float roi_batch_ind = bottom_rois[n * 7 + 0];
    float roi_start_w = bottom_rois[n * 7 + 2] * spatial_scale;
    float roi_start_h = bottom_rois[n * 7 + 3] * spatial_scale;
    float roi_end_w = bottom_rois[n * 7 + 4] * spatial_scale;
    float roi_end_h = bottom_rois[n * 7 + 5] * spatial_scale;

    // Force malformed ROIs to be 1x1
    float roi_width = fmaxf(roi_end_w - roi_start_w + 1.0, 0.0);
    float roi_height = fmaxf(roi_end_h - roi_start_h + 1.0, 0.0);
    float bin_size_h = roi_height / (aligned_height - 1.0);
    float bin_size_w = roi_width / (aligned_width - 1.0);

    float h = (float)(ph) * bin_size_h + roi_start_h;
    float w = (float)(pw) * bin_size_w + roi_start_w;

    int hstart = fminf(floor(h), height - 2);
    int wstart = fminf(floor(w), width - 2);

    int img_start = roi_batch_ind * channels * height * width;

    // bilinear interpolation
    if (h < 0 || h >= height || w < 0 || w >= width) 
    {
      top_data[index] = 0.;
    }
    else
    {
      float h_ratio = h - (float)(hstart);
      float w_ratio = w - (float)(wstart);
      int upleft = img_start + (c * height + hstart) * width + wstart;
      int upright = upleft + 1;
      int downleft = upleft + width;
      int downright = downleft + 1;

      top_data[index] = bottom_data[upleft] * (1. - h_ratio) * (1. - w_ratio)
                      + bottom_data[upright] * (1. - h_ratio) * w_ratio
                      + bottom_data[downleft] * h_ratio * (1. - w_ratio)
                      + bottom_data[downright] * h_ratio * w_ratio;
    }
  }
}


std::vector<at::Tensor> roi_align_cuda_forward(
    int aligned_height,
    int aligned_width,
    float spatial_scale,
    at::Tensor bottom_features,
    at::Tensor bottom_rois) 
{
  // run kernels
  const int kThreadsPerBlock = 1024;
  cudaError_t err;

  const int batch_size = bottom_features.size(0);
  const int num_channels = bottom_features.size(1);
  const int height = bottom_features.size(2);
  const int width = bottom_features.size(3);
  const int num_rois = bottom_rois.size(0);

  auto top_data = at::zeros({num_rois, num_channels, aligned_height, aligned_width}, bottom_features.options());
  const int output_size = num_rois * num_channels * aligned_height * aligned_width;

  ROIAlignForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
    output_size, bottom_features.data<float>(), spatial_scale, height, width, num_channels, 
    aligned_height, aligned_width, bottom_rois.data<float>(), top_data.data<float>());

  err = cudaGetLastError();
  if(cudaSuccess != err) 
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return {top_data};
}


__global__ void ROIAlignBackward(const int nthreads, const float* top_diff, const float spatial_scale, const int height, const int width,
                                 const int channels, const int aligned_height, const int aligned_width, float* bottom_diff, const float* bottom_rois) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, c, ph, pw) is an element in the aligned output
    int pw = index % aligned_width;
    int ph = (index / aligned_width) % aligned_height;
    int c  = (index / aligned_width / aligned_height) % channels;
    int n  = index / aligned_width / aligned_height / channels;

    float roi_batch_ind = bottom_rois[n * 7 + 0];
    float roi_start_w = bottom_rois[n * 7 + 2] * spatial_scale;
    float roi_start_h = bottom_rois[n * 7 + 3] * spatial_scale;
    float roi_end_w = bottom_rois[n * 7 + 4] * spatial_scale;
    float roi_end_h = bottom_rois[n * 7 + 5] * spatial_scale;

    // Force malformed ROIs to be 1x1
    float roi_width = fmaxf(roi_end_w - roi_start_w + 1.0, 0.0);
    float roi_height = fmaxf(roi_end_h - roi_start_h + 1.0, 0.0);
    float bin_size_h = roi_height / (aligned_height - 1.0);
    float bin_size_w = roi_width / (aligned_width - 1.0);

    float h = (float)(ph) * bin_size_h + roi_start_h;
    float w = (float)(pw) * bin_size_w + roi_start_w;

    int hstart = fminf(floor(h), height - 2);
    int wstart = fminf(floor(w), width - 2);
    int img_start = roi_batch_ind * channels * height * width;

    // bilinear interpolation
    if (!(h < 0 || h >= height || w < 0 || w >= width)) 
    {
      float h_ratio = h - (float)(hstart);
      float w_ratio = w - (float)(wstart);
      int upleft = img_start + (c * height + hstart) * width + wstart;
      int upright = upleft + 1;
      int downleft = upleft + width;
      int downright = downleft + 1;

      atomicAdd(bottom_diff + upleft, top_diff[index] * (1.0 - h_ratio) * (1.0 - w_ratio));
      atomicAdd(bottom_diff + upright, top_diff[index] * (1.0 - h_ratio) * w_ratio);
      atomicAdd(bottom_diff + downleft, top_diff[index] * h_ratio * (1.0 - w_ratio));
      atomicAdd(bottom_diff + downright, top_diff[index] * h_ratio * w_ratio);
    }
  }
}


std::vector<at::Tensor> roi_align_cuda_backward(
    int batch_size,
    int height,
    int width,
    float spatial_scale,
    at::Tensor top_diff,
    at::Tensor bottom_rois)
{
  const int kThreadsPerBlock = 1024;
  cudaError_t err;

  const int num_rois = top_diff.size(0);
  const int num_channels = top_diff.size(1);
  const int aligned_height = top_diff.size(2);
  const int aligned_width = top_diff.size(3);

  auto bottom_diff = at::zeros({batch_size, num_channels, height, width}, top_diff.options());
  const int output_size = num_rois * num_channels * aligned_height * aligned_width;

  ROIAlignBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
    output_size, top_diff.data<float>(), spatial_scale, height, width, num_channels,
    aligned_height, aligned_width, bottom_diff.data<float>(), bottom_rois.data<float>());

  err = cudaGetLastError();
  if(cudaSuccess != err) 
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit(-1);
  }

  return {bottom_diff};
}
