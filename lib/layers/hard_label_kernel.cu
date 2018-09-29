#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename Dtype>
__global__ void HardLabelForward(const int nthreads, const float threshold,
    const Dtype* bottom_prob, const Dtype* bottom_label, Dtype* top_data)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    if (bottom_label[index] > 0 && bottom_prob[index] < threshold)
      top_data[index] = 1.0;    
  }
}


std::vector<at::Tensor> hard_label_cuda_forward(
    float threshold,
    at::Tensor bottom_prob,
    at::Tensor bottom_label) 
{
  // run kernels
  const int kThreadsPerBlock = 1024;
  int output_size;

  const int batch_size = bottom_prob.size(0);
  const int num_channels = bottom_prob.size(1);
  const int height = bottom_prob.size(2);
  const int width = bottom_prob.size(3);

  auto top_data = at::zeros({batch_size, num_channels, height, width}, bottom_prob.options());

  AT_DISPATCH_FLOATING_TYPES(bottom_prob.type(), "hard_label_forward_cuda", ([&] {

    // compute the losses and gradients
    output_size = batch_size * num_channels * height * width;
    HardLabelForward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        output_size,
        threshold,
        bottom_prob.data<scalar_t>(),
        bottom_label.data<scalar_t>(),
        top_data.data<scalar_t>());

  }));

  return {top_data};
}


std::vector<at::Tensor> hard_label_cuda_backward(
    at::Tensor top_diff)
{
  const int batch_size = top_diff.size(0);
  const int num_channels = top_diff.size(1);
  const int height = top_diff.size(2);
  const int width = top_diff.size(3);

  auto grad_prob = at::zeros({batch_size, num_channels, height, width}, top_diff.options());
  auto grad_label = at::zeros({batch_size, num_channels, height, width}, top_diff.options());

  return {grad_prob, grad_label};
}
