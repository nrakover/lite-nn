# lite-nn

[![CircleCI](https://circleci.com/gh/nrakover/lite-nn.svg?style=shield)](https://circleci.com/gh/nrakover/lite-nn)

Lightweight neural networks library written entirely in Python3 and numpy

## Overview
This project is based on my experience with the *deeplearning.ai* [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) offered on Coursera, partly as a way to cement what I learned throughout the courses.

The library is meant to be an ultra-portable, minimal-dependency tool for testing out and iterating on ideas. It is designed to be easy to use, with familiar APIs and sensible defaults, while also offering a useful level of configurability and extensibility.

The implementation makes extensive use of vectorization via numpy. All operations that scale with the number of examples and/or with the number of features/hidden-units are vectorized.

## Features
- fully-connected neural nets of configurable depth and height
- regularization:
  - L2 / weight-decay
  - dropout (per layer)
- weight initializers:
  - sample from zero-mean normal
  - Xavier et al.
  - He et al.
- activation functions:
  - ReLU
  - sigmoid
  - softmax [COMING SOON]
  - tanh [COMING SOON]
- cost functions:
  - sigmoid cross-entropy
  - softmax cross-entropy [COMING SOON]
- mini-batch size
- gradient optimizers:
  - vanilla batch GD
  - momentum [COMING SOON]
  - RMSProp [COMING SOON]
  - Adam
- batch normalization [COMING SOON]

## Dependencies
- python3
- numpy
- unittest
