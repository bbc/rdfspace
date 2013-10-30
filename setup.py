#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='rdfspace',
    version='0.0.2',
    description="""RDFSpace constructs a vector space 
                 from any RDF dataset which can be used for 
                 computing similarities between resources 
                 in that dataset.""",
    author='Yves Raimond',
    author_email='yves.raimond@bbc.co.uk',
    packages=['rdfspace'],
    install_requires=[
        'scipy',
        'sparsesvd',
        'cython',
    ],
)
