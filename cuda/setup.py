from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='cuda_kan',
      ext_modules=[CUDAExtension('cuda_kan', ['kan.cu'])],
      cmdclass={'build_ext': BuildExtension})