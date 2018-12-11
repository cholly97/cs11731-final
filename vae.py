import tensorflow as tf
import numpy as np

mnist_moving = np.load(open('../data/mnist_test_seq.npy', mode = 'rb')) # (depth: 20, batch_size: 10000, height: 64, width: 64)