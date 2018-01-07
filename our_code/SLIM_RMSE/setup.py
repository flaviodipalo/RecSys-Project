from distutils.core import setup
from Cython.Build import cythonize
# python setup.py build_ext --inplace

setup(
  name = 'Hello world app',
  ext_modules = cythonize("SLIM_RMSE.pyx"),
)