from distutils.core import setup
import numpy
from Cython.Build import cythonize
# python setup_mf.py build_ext --inplace

setup(
  name='Hello',
  ext_modules=cythonize('Cython/MF_RMSE.pyx'),
  include_dirs=[numpy.get_include()]
)
