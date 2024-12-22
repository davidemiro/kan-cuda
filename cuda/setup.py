from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='cuda_kan',
      ext_modules=[CUDAExtension('cuda_kan', ['cuda_kan.cu'], extra_compile_args={
            'nvcc': ['-rdc=true']  # Enable relocatable device code
      })],
      cmdclass={'build_ext': BuildExtension})