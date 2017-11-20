# create a top-level flow to use the underlying NN
import sys
sys.path.append('./src')

import network
import mnist_loader
import timeit
import profile
import pstats
import numpy as np

do_profile=0

prof_file='Learn1.prof'

# load the data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper(fname='data/mnist.pkl.gz')

# setup the network
net = network.Network([784,60,30,10])

train_size_ratio = 1
test_size_ratio  = 1

num_train   = int(len(training_data)/train_size_ratio)
num_test    = int(len(test_data)/test_size_ratio)


net.SGD(training_data[:num_train], epochs=10, mini_batch_size=32, eta=3.0, test_data=test_data[:num_test])

# run profiler
# else:
#     cmd='net.SGD(training_data, epochs=100, mini_batch_size=10, eta=3.0, test_data=test_data)'
#     profile.run(cmd,prof_file)
#
#     # read the stats file
#     stats = pstats.Stats(prof_file)
#
#     # Clean up filenames for the report
#     stats.strip_dirs()
#     stats.sort_stats('tottime')
#
#     # print the stats
#     stats.print_stats()
