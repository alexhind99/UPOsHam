#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 06:31:40 2019

@author: Wenyang Lyu, Shibabrat Naik
"""

from setuptools import find_packages
from distutils.core import setup

import os
on_rtd = os.environ.get('READTHEDOCS') == 'True'

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

package_name = 'uposham'
version_num = '0.1.0'

def main():
    install_requires = ['m2r'] if on_rtd else []
    setup(
        name = package_name,

        version = version_num,
        
        description = 'Python package for computing unstable periodic orbits',
        
        long_description = readme,
        
        author = 'Wenyang Lyu, Shibabrat Naik',
        
        author_email = 'wl16298@bristol.ac.uk, shiba@vt.edu.',
        
        url = 'https://github.com/WyLyu/UPOsHam',
        
        license = license,
        
        # packages=find_packages(exclude=('test','data', 'docs')),
        packages=['uposham'],
        package_dir={'uposham': 'src'},

        install_requires = requirements
    )


if __name__ == '__main__':
    main()
