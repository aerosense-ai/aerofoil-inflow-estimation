#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup  # , find_packages


setup(
    name='aiem',
    version='0.1.0',
    description='Airfoil Inflow Estimation Models',
    author='Aerosense Team',
    author_email='julien.deparday@ost.ch',
    package_dir={'': '.'},
    packages=['aiem'],
    license='All rights reserved to Aerosense Team',
    zip_safe=False
)
