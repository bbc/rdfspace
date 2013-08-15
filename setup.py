#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='rdfspace',
    version='0.1',
    description='Modelling RDF data as a vector space',
    author='Yves Raimond',
    author_email='yves.raimond@bbc.co.uk',
    packages=['rdfspace'],
    install_requires=[
        'scipy',
        'sparsesvd',
        'cython',
    ],
)
