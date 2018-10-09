from distutils.core import setup
import numpy
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

#TODO: fallo entrare nella stessa cartella dove dovrebbe essere.
# python setup_mf.py build_ext --inplace


setup(
  include_dirs=[numpy.get_include()],
  name='Hello',
  ext_modules=cythonize('MatrixFactorization_Cython_Epoch.pyx')
)
