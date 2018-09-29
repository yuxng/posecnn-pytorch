from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='posecnn',
    ext_modules=[
        CUDAExtension('posecnn_cuda', [
            'hard_label.cpp',
            'hard_label_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
