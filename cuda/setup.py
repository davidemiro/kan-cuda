from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='cuda_layers',
      ext_modules=[CUDAExtension('cuda_kan', ['cuda_kan.cu'])],
      cmdclass={'build_ext': BuildExtension})