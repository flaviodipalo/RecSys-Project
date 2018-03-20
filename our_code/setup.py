from distutils.core import setup
import numpy
from Cython.Build import cythonize
# python setup.py build_ext --inplace

setup(
  name = 'Hello world app',
  ext_modules = cythonize("cython_main.pyx"),
  include_dirs=[numpy.get_include()]
)