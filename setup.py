#!/usr/bin/env python

import os
import setuptools


module_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(module_dir, 'readme.rst'), "r") as f:
    long_description = f.read()

install_requires = [
    "ase",
    "matplotlib",
    "ndsplines",
    "numba",
    "numpy<=1.26.4",
    "pandas",
    "plotly",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "tables",
    "tqdm",
]

test_requires = ['pytest']

if __name__ == "__main__":
    setuptools.setup(
        name='uf3',
        version='0.4.0',
        description='Ultra-Fast Force Fields for molecular dynamics',
        long_description=long_description,
        url='https://github.com/uf3/uf3',
        author='Stephen R. Xie, Matthias Rupp',
        author_email='sxiexie@ufl.edu',
        license='Apache 2.0',
        packages=setuptools.find_packages(exclude=["tests"]),
        install_requires=install_requires,
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering'
        ],
        python_requires='>=3.9, <3.13',
        tests_require=test_requires,
    )
