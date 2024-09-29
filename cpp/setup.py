from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='cpp_kan',
      ext_modules=[cpp_extension.CppExtension('cpp_kan', ['cpp_kan.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})