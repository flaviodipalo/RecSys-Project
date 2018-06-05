from distutils.core import setup
import numpy
from distutils.extension import Extension
from Cython.Build import cythonize
# python setup.py build_ext --inplace

ext_modules = [
    Extension(
        "hello",
        ["SLIM_RMSE_Cython_Epoch.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
  name = 'Hello world app',
  ext_modules = cythonize(ext_modules),
  include_dirs=[numpy.get_include()]
)