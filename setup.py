#!/usr/bin/env python

import os

from setuptools import setup, find_packages

__version__ = "0.9"

setup(name='eventdetector',
      version=__version__,
      description='Deep Learning models for automatic gait event annotation',
      author='Lukasz Kidzinski',
      author_email='lukasz.kidzinski@gmail.com',
      url='http://github.com/kidzik/event-detector',
      downlnoad_url='https://github.com/kidzik/event-detector/archive/v0.9.tar.gz',
      license='Apache 2.0',
      packages=find_packages(),
      package_data={},
      include_package_data=True,
      install_requires=['h5py','numpy>=1.13','tensorflow','keras>=2.0.6',''],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          ],
      scripts=['scripts/event-detector']
)
