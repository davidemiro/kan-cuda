from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='cuda_kan',
      ext_modules=[CUDAExtension('cuda_kan',
                        ['cuda_kan.cu', 'spline.cu'],
                        dlink=True,
                        dlink_libraries=["dlink_lib"],
                                 extra_compile_args={
                                       'nvcc': ['-rdc=true']  # Enable relocatable device code
      })],
      cmdclass={'build_ext': BuildExtension})