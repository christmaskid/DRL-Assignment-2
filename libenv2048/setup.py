from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

sourceFiles = ["env2048.py"]

extensions = cythonize(Extension(
             name="env2048compiled",
             sources = sourceFiles
      ))

kwargs = {
       "name": "env2048compiled",
       "packages": find_packages(),
       "ext_modules": extensions
      }

setup(**kwargs)