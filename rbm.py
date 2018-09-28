'''
Python implementation of a restricted Boltzmann machine (RBM) with arbitrary number of visible and hidden units.
Uses a gradient-based contrastive-divergence algorithm.
Daniel J. Sharpe 02/2018
'''

import numpy as np
from copy import deepcopy
from itertools import product

class Rbm(object):

    def __init__(self, n_visible, n_hidden, trainfile, n_epochs=5000, eta=0.1, theta=1.0E-04, zeta=0.5,  mean=0.0, \
                 stddev=0.1, thresh=1.0E-04, batch_size=5, n_reconstruct=1, vis_out="logit", randseed=17):
        self.n_visible = n_visible # no. of visible units
        self.n_hidden = n_hidden # no. of hidden units
        self.n_epochs = n_epochs # max. no. of epochs in training procedure
        self.eta = eta # learning rate
        self.theta = theta # weight decay coeff
        self.zeta = zeta # momentum coeff
        self.thresh = thresh # error threshold during training
        self.batch_size = batch_size # size of batches used as set of training instances in a single epoch
        self.n_reconstruct = n_reconstruct # number of reconstructions of the visible units before updating hidden units
        self.vis_out = vis_out # option whether output of visible layers is logistic sigmoid "logit" OR "softmax"
        np.random.seed(randseed)
        # initialise weights with uniform distribution between +/-(sqrt(6/(nhidden+nvis))) with given mean and std dev
        self.traindata = self.readdata(trainfile)
        np.set_printoptions(precision=4)
        print "Training data:\n", self.traindata
        self.weights = np.asarray(np.random.uniform(
                low=(-stddev*np.sqrt(6. / (self.n_visible + self.n_hidden))+mean),
                high=(stddev*np.sqrt(6. / (self.n_visible + self.n_hidden))+mean),
                size = (self.n_visible, self.n_hidden)))
        self.weights = np.insert(self.weights, 0, 0., axis = 0) # initial biases for visible units
        self.weights = np.insert(self.weights, 0, 0., axis = 1) # initial biases for hidden units
        self.weights[1:,0] = self.get_bias_vis(self.traindata) # new initial biases for visible units
        self.weights_momentum = np.zeros((self.n_visible+1, self.n_hidden+1))
        self.Z = 0. # partition function

    # print results of training
    def print_soln(self, epochs, err, visible_probs, hidden_probs, energy, config_probs):
        print "After %i epochs of training...\nError: %f\nFinal weights:\n" % (epochs, err), self.weights, \
               "\nProbabilities for visible units:\n", visible_probs[:,1:], \
               "\nProbabilities for hidden units:\n", hidden_probs[:,1:], "\nEnergies of configurations:\n", energy, \
               "\nProbabilities of configurations:\n", config_probs

    # print results of testing
    def print_result(self, testdata, hidden_states, hidden_probs, energy, config_probs):
        print "Test data:\n", testdata, "\nHidden unit states:\n", hidden_states, "\nHidden unit probabilities\n", \
              hidden_probs, "\nEnergies of configurations:\n", energy, "\nProbabilities of configurations\n", config_probs

    # get optimal initial biases for visible units according to proportion of training instance vectors in which unit j is on
    def get_bias_vis(self, visible_states):
        unit_on_occur = np.zeros(self.n_visible, dtype=int) # count occurences of visible units being on over all training instances
        for i in range(np.shape(self.traindata)[0]):
            for j in range(self.n_visible):
                if visible_states[i,j] == 1: # visible unit j is on for training instance i
                    unit_on_occur[j] += 1
        p = np.zeros(self.n_visible, dtype=float)
        for j in range(self.n_visible):
            p[j] = float(unit_on_occur[j]) / float(np.shape(self.traindata)[0])
            p[j] = np.log(p[j]/(1.0-p[j]))
        return p

    # sigmoidal logistic function
    def logit(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    # softmax function for use when visible units of the RBM have multinomial output
    def softmax(self, x):
        exps = np.exp(x - np.amax(x))
        sum_exps = np.sum(exps)
        return exps / sum_exps

    # Calculate probability of configurations with given energy based on knowledge of partition function
    def calc_prob(self, energies):
        return (1./self.Z)*np.exp(-energies)

    # Evaluate the partition function by calculating the energies of all possible configurations
    def partition_func(self):
        all_hidden = np.array(list(product([0,1], repeat=self.n_hidden))) # all possible hidden states
        all_visible = np.array(list(product([0,1], repeat=self.n_visible))) # all possible visible states
        all_hidden = np.insert(all_hidden, 0, 1, axis=1) # bias units
        all_visible = np.insert(all_visible, 0, 1, axis=1) # bias units
        for i in range(2**self.n_hidden):
            for j in range(2**self.n_visible):
                try:
                    energies = np.append(energies, self.config_energy(all_hidden[i,:], all_visible[j,:]))
                except NameError: #first instance
                    energies = self.config_energy(all_hidden[i,:], all_visible[j,:])
        self.Z = np.sum(np.exp(-energies)) # partition function
        sum_config_probs = np.sum(self.calc_prob(energies))
        if round(sum_config_probs, 8) != 1.: print "Warning: Sum of probabilities is not equal to unity"

    # Return the energy of a configuration (pair of Boolean vectors v,h (visible,hidden units on/off))
    # (or corresponding probability vectors) given the weight matrix
    def config_energy(self, hidden_states, visible_states):
        config_energies = []
        if len(np.shape(hidden_states)) != 1: n_iter = np.shape(hidden_states)[0]
        else: n_iter = 1
        for i in range(n_iter): # loop through instances (each config (v,h))
            if len(np.shape(hidden_states)) != 1:
                config_energies.append(np.dot(visible_states[i,:], np.dot(self.weights, \
                        np.transpose(hidden_states[i,:]))))
            else:
                config_energies.append(np.dot(visible_states, np.dot(self.weights, np.transpose(hidden_states))))
        config_energies = np.array(config_energies)
        return config_energies

    # Read training / test data into numpy array
    def readdata(self, fname):
        with open(fname, "r") as f:
            for line in f: # training instances
                line = line.split()
                try:
                    data = np.append(data,[[float(point) for point in line]],axis=0)
                except NameError: # first entry
                    data = np.array([[float(point) for point in line]], dtype=float)
        return data

    # Test learned machine using sample data and converged edge weights
    def test(self, fname):
        print "Testing..."
        testdata = self.readdata(fname)
        hidden_states, hidden_probs = self.run_layer(testdata, "visible")
        testdata_wbias = deepcopy(testdata)
        testdata_wbias = np.insert(testdata_wbias, 0, 1, axis=1)
        hidden_states_wbias = deepcopy(hidden_states)
        hidden_states_wbias = np.insert(hidden_states_wbias, 0, 1, axis=1)
        energy = self.config_energy(hidden_states_wbias, testdata_wbias)
        config_probs = self.calc_prob(energy)
        self.print_result(testdata, hidden_states, hidden_probs[:,1:], energy, config_probs)

    # Train machine by contrastive-divergence algorithm implementing Gibbs sampling
    def train(self):
        print "Training..."
        n_instances = np.shape(self.traindata)[0]
        data = deepcopy(self.traindata)
        data = np.insert(data, 0, 1, axis=1) # visible bias unit (always on)
        epoch = 0
        err = float("inf")
        final_loop = False
        while epoch < self.n_epochs+1:
            if not final_loop: np.random.shuffle(data)
            # reality phase (positive gradient)
            pos_hidden_states, pos_hidden_probs = self.run_layer(data[:self.batch_size,1:], "visible")
            pos_assocs = np.dot(np.transpose(data[:self.batch_size,:]), pos_hidden_probs)
            # daydreaming phase (negative gradient)
            neg_hidden_probs = deepcopy(pos_hidden_states)
            neg_hidden_probs = np.insert(neg_hidden_probs, 0, 1, axis=1)
            for i in range(self.n_reconstruct):
                neg_visible_probs = self.run_layer(neg_hidden_probs[:,1:], "hidden")[1]
                neg_hidden_probs = self.run_layer(neg_visible_probs[:,1:], "visible")[1]
            neg_assocs = np.dot(np.transpose(neg_visible_probs), neg_hidden_probs)
            # update phase
            self.weights_momentum *= self.zeta
            self.weights += self.eta*((pos_assocs - neg_assocs) / n_instances)
            self.weights += self.eta*self.weights_momentum / n_instances
            self.weights -= self.weights*self.theta # L2 regularization
            self.weights_momentum += pos_assocs - neg_assocs
            err = np.sum((data[:self.batch_size,:] - neg_visible_probs)**2) / self.batch_size # avg err in a single set of vis units
            epoch += 1
            if final_loop: break
            if epoch == self.n_epochs-1 or err < self.thresh:
                final_loop = True
                self.batch_size = n_instances
                data = deepcopy(self.traindata)
                data = np.insert(data, 0, 1, axis=1)
        self.partition_func()
        print "Partition function: Z =", self.Z
        energy = self.config_energy(neg_hidden_probs, neg_visible_probs)
        config_probs = self.calc_prob(energy)
        self.print_soln(epoch, err, neg_visible_probs, neg_hidden_probs, energy, config_probs)
        return

    # Given (learned) weights (also used to initialise bias weights for visible units)
    # run the network on a set of visible units to get a sample of the hidden units
    # OR run the network on a set of hidden units to get a sample of the visible units
    # other_layer is a matrix, each row of which is a Boolean vector describing the state
    # of the layer NOT being computed
    def run_layer(self, other_layer, layer):
        n_samples = np.shape(other_layer)[0] # no. of samples (sets of other layer states)
        other_layer = np.insert(other_layer, 0, 1, axis=1) # bias units (always on)
        if layer == "visible": # run network on visible units (get hidden states)
            n_units = self.n_hidden+1
            activations = np.dot(other_layer, self.weights) # activations of hidden units
        elif layer == "hidden": # run network on hidden units (get visible states)
            n_units = self.n_visible+1
            activations = np.dot(other_layer, np.transpose(self.weights)) # activations of visible units
        states = np.ones((n_samples, n_units)) # matrix containing (Boolean) states of hidden or visible units (as appropriate)
                                               # (incl. bias unit) for each training instance
        if layer == "hidden" or (layer == "visible" and self.vis_out == "logit"): probs = self.logit(activations)
        elif layer == "visible" and self.vis_out == "softmax": probs = self.softmax(activations)
        # turn the hidden units on according to the above probabilities
        states[:,:] = probs > np.random.rand(n_samples, n_units)
        probs[:,0] = 1.
        return states[:,1:], probs # ignore biases in states but not in probs

    # Randomly initialise visible units and run Gibbs sampling steps alternately on visible and hidden layers.
    # Return a matrix "samples" where each row is a vector describing the visible unit states of a single "daydream"
    def daydream(self, n_dreams):
        print "Daydreaming..."
        hidden_states = np.ones((n_dreams, self.n_hidden+1))
        visible_states = np.ones((n_dreams, self.n_visible+1))
        visible_probs = np.ones((n_dreams, self.n_visible+1)) # probabilities of visible units being turned on
        for i in range(n_dreams):
            visible_probs[i,1:] = np.random.rand(self.n_visible) # initialise visible units as uniform distribution
        for k in range(self.n_reconstruct):
            hidden_activations = np.dot(visible_probs, self.weights)
            hidden_probs = self.logit(hidden_activations)
            hidden_states[:,:] = hidden_probs > np.random.rand(self.n_hidden+1)
            hidden_probs[:,0], hidden_states[:,0] = 1., 1. # bias unit always on
            # now recalculate probabilities that visible units are on
            visible_activations = np.dot(hidden_states, np.transpose(self.weights))
            if self.vis_out == "logit": visible_probs = self.logit(visible_activations)
            elif self.vis_out == "softmax": visible_probs = self.softmax(visible_activations)
            visible_states[:,:] = visible_probs > np.random.rand(self.n_visible+1)
            visible_probs[:,0], visible_states[:,0] = 1., 1.
        energy = self.config_energy(hidden_states, visible_states)
        config_probs = self.calc_prob(energy)
        self.print_result(visible_states[:,1:], hidden_states[:,1:], hidden_probs[:,1:], energy, config_probs)
        return hidden_states[:,1:] # ignore bias units

#Driver code
trainfile = "rbm_dataset.dat" # file containing training data
testfile = "rbm_testdata.dat" # file containing test data
rbm1 = Rbm(6, 2, trainfile, n_reconstruct=5)
rbm1.train()
rbm1.test(testfile)
dream_results = rbm1.daydream(5)
print rbm1.config_energy(np.array([1., 0., 1.]), np.array([1., 1., 1., 1., 0., 0., 0.])) # this should be a high-energy config...
