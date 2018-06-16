from distutils.core import setup
import numpy
from Cython.Build import cythonize
# python setup.py build_ext --inplace

setup(
  name='Hello world app',
  ext_modules=cythonize("SLIM_RMSE_Cython_Epoch_Normalized.pyx"),
  include_dirs=[numpy.get_include()]
)

setup(
  name='Hello',
  ext_modules=cythonize('SLIM_RMSE_Cython_Epoch_Normal.pyx'),
  include_dirs=[numpy.get_include()]
)