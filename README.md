# lite-nn
Lightweight neural networks library written entirely in Python3 and numpy

## Overview
This project is based on my experience with the *deeplearning.ai* [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) offered on Coursera, partly as a way to cement what I learned throughout the courses.

The library is meant to be an ultra-portable, minimal-dependency tool for testing out and iterating on ideas. It is designed to be easy to use, with familiar APIs and sensible defaults, while also offering a useful level of configurability and extensibility.

The implementation is heavily vectorized.

## Features
- fully-connected neural nets of configurable depth and height
- regularization:
  - L2 / weight-decay
  - dropout (per layer)
- configurable options:
  - weight initializers
  - activation functions
  - cost functions
  - mini-batch size [COMING SOON]
  - gradient optimizers [COMING SOON]

## Dependencies
- python3
- numpy
- unittest
