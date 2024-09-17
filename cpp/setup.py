from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='cpp_layers',
      ext_modules=[cpp_extension.CppExtension('cpp_spline', ['cpp_spline.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})