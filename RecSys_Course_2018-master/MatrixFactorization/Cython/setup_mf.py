from distutils.core import setup
import numpy
from Cython.Build import cythonize
# python setup_mf.py build_ext --inplace

setup(
  name='Hello',
  ext_modules=cythonize('MatrixFactorization_Cython_Epoch.pyx'),
  include_dirs=[numpy.get_include()]
)
