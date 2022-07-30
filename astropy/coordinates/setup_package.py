# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Copied from astropy/time/setup_package.py

import os
from setuptools import Extension

import numpy

C_COORDINATES_PKGDIR = os.path.relpath(os.path.dirname(__file__))

SRC_FILES = [
    'optimizations.c',
]
SRC_FILES = [
    os.path.join(C_COORDINATES_PKGDIR, 'src', filename)
    for filename in SRC_FILES
]

print("SOURCE_FILES", SRC_FILES)


def get_extensions():
    # Add '-Rpass-missed=.*' to ``extra_compile_args`` when compiling with clang
    # to report missed optimizations
    _coord_ext = Extension(
        name='astropy.coordinates._optimizations',
        sources=SRC_FILES,
        include_dirs=[numpy.get_include()],
        language='c',
    )

    return [_coord_ext]
