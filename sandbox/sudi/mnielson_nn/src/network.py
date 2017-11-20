"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import visualization as vsl

class Network(object):
# SUDI: looks ok
    def __init__(self, sizes,debug=0):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.debug = debug
        self.cost  = 0
        print ("Network: layers=%d sizes=%s\n" %(self.num_layers,self.sizes));

# SUDI: these are actually lists containing arrays
        self.biases  = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

# SUDI: going over
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        # for plotting
        accuracy_X = list()
        accuracy_Y = list()
        epoch_X    = list()
        cost_Y     = list()

        print ("SGD: train_size=%d epochs=%d mini_batch_size=%d eta=%2.3f" \
        %(len(training_data), epochs, mini_batch_size, eta));

        accuracy   = 0

        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            epoch_cost = 0

            # shuffle order in training_data
            random.shuffle(training_data)

            # list of lists each of size=mini_batch_size
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

            # this loop is called (n/mini_batch_size) times
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                epoch_cost  += self.cost

            # cost related
            epoch_X.append(j)
            cost_Y.append(epoch_cost)
            prev_cost = cost_Y[j-1]
            cost_diff = 0 if (j==0) else (((epoch_cost-prev_cost)/prev_cost) * 100.0)
            print ('\n(train)Epoch %02d: Cost:%03.2f(%03.2f)' %(j,epoch_cost,cost_diff))

            # even if test_data is provided only check every 4 epochs
            if test_data and (j%1==0):
                num_correct = self.evaluate(test_data)
                temp        = accuracy
                accuracy = (float(num_correct) / float(n_test)) * 100.0
                change   = 0 if (temp==0) else ((accuracy-temp)/temp)   * 100.0
                accuracy_X.append(j)
                accuracy_Y.append(accuracy)

                print ('(test) Epoch %02d: %03.2f (%02.2f)' %(j, accuracy,change))
            else:
                print ('(test) Epoch %02d: completed' %(j))
        # plot
        vsl.scatter_plot(accuracy_X,'epoch',accuracy_Y,'accuracy','accuracy')
        vsl.scatter_plot(epoch_X,   'epoch',cost_Y,    'cost',   'cost v/s epoch')

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # accumulate the delta for each sample of the batch
        for x, y in mini_batch:
            # backprop across the samples in the mini_batch
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # update the bias/weights
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # index of the final layer
        final_layer = -1

        # compute the cost & store it
        self.cost = self.compute_cost(activations[final_layer], y)

        # backward pass
        # Layer: output layer, i.e. L
        delta = self.cost_derivative(activations[final_layer], y) * \
            sigmoid_prime(zs[final_layer])
        nabla_b[final_layer] = delta
        nabla_w[final_layer] = np.dot(delta, activations[final_layer-1].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.

        # Layer: L-1 to 1
        for l in xrange(2, self.num_layers):
            cur_layer= -l
            z = zs[cur_layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[cur_layer+1].transpose(), delta) * sp
            nabla_b[cur_layer] = delta
            nabla_w[cur_layer] = np.dot(delta, activations[cur_layer-1].transpose())

        # done computing, return values
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        # SUDI: how exactly does argmax work?
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def compute_cost(self, output_activations, y):
        """Return the MSE cost"""
        delta = (output_activations - y)
        cost  = (1/(2*float(self.sizes[-1]))) * (float(np.dot(delta.transpose(),delta)))
        return (cost)

# SUDI: TODO this will change if the classifier changes, e.g. SVM, softmax
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
# SUDI: TODO this assumes the non-linerity is sigmoid, will change if trying ReLU etc
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

# SUDI: TODO this assumes the non-linerity is sigmoid, will change if trying ReLU etc
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
