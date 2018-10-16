from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='posecnn',
    ext_modules=[
        CUDAExtension(
            name='posecnn_cuda', 
            sources = [
            'posecnn_layers.cpp',
            'hard_label_kernel.cu',
            'hough_voting_kernel.cu',
            'roi_pooling_kernel.cu',
            'point_matching_loss_kernel.cu'],
            include_dirs = ['/usr/local/include/eigen3'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
