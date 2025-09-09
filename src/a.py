print('hi lavi')

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide info & warnings

import tensorflow as tf