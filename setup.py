#!/usr/bin/env python
# run: python setup.py bdist_wheel

from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
      "opencv-contrib-python",
      "opencv-python==4.2.0.32",
      "scikit-learn",
      "xgboost",
      "tqdm",
      "matplotlib",
      "scikit-image",
      "numpy",
      "joblib",
      "imblearn",
]

setup(name='classic_image_classification',
      version='1.0.0',
      description='This package is used to classify images using hand-crafted features',
      author='Friedrich Muenke',
      author_email='friedrich.muenke@me.com',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      install_requires=REQUIRED_PACKAGES,)
