# create a top-level flow to use the underlying NN
import sys
sys.path.append('./src')

import network
import mnist_loader
import timeit
import profile
import pstats
import numpy as np
import argparse

# --- command line arguments --- #
parser = argparse.ArgumentParser()
# training
parser.add_argument('--train_scale_ratio', help="scale down training  size", type=int, default=1)
parser.add_argument('--test_scale_ratio',  help="scale down test-data size", type=int, default=1)
parser.add_argument('--epochs',            help="number of epochs to train",        type=int, default=30)

# network configuration
parser.add_argument('--hidden_layers',     help="hidden layers, space seperated",   type=str, default='30')

# hyper parameters
parser.add_argument('--learn_rate',        help="learning-rate",                    type=float, default=3.0)
parser.add_argument('--batch_size',        help="mini batch size",                  type=int,   default=32)

# optimizations
parser.add_argument('--norm_weights',      help="apply simple weight norm",         type=int, default=1)

# Miscellaneous
parser.add_argument('--plot_act_hist',     help="plot the activation hist/epoch",   type=int, default=0)

# parse args
args = parser.parse_args()

# form the hidden layer(s)
hidden_layer = []
for layer in args.hidden_layers.split():
    hidden_layer.append(int(layer))

# load the data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper(fname='data/mnist.pkl.gz')

# setup the network, can't change the input & final layer
all_layer = [784] + hidden_layer + [10]

net = network.Network(all_layer,norm_weights=args.norm_weights,do_hist=args.plot_act_hist)

train_size_ratio = args.train_scale_ratio
test_size_ratio  = args.test_scale_ratio

num_train   = int(len(training_data)/train_size_ratio)
num_test    = int(len(test_data)/test_size_ratio)
net.SGD(training_data[:num_train], epochs=args.epochs, mini_batch_size=args.batch_size, eta=args.learn_rate, test_data=test_data[:num_test])

print "INFO: training done.. press any key to continue"
raw_input()

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
