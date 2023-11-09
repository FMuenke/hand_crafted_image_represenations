#!/usr/bin/env python
# run: python setup.py bdist_wheel

from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
      "opencv-contrib-python",
      "scikit-learn",
      "imagesize",
      "tqdm",
      "matplotlib",
      "scikit-image",
      "numpy",
      "joblib",
      "seaborn"
]

setup(name='handcrafted_image_representations',
      version='1.0.1',
      description='This package is used to classify images using hand-crafted features',
      author='Friedrich Muenke',
      author_email='friedrich.muenke@me.com',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      install_requires=REQUIRED_PACKAGES,)
