"""
Main trainer function

Modified by: William Blackie

Reason: Numpy error with None comparison was unpythonic and throwing errors

Reason: Added graphing to show progression of training vectors

Reason: pep8 formatting
"""
from theano import shared, function, sandbox
from theano.tensor import scalar, sqrt, grad, switch

import cPickle as pkl
from numpy import mod, float32, isinf, isnan, savez, mean
import copy

import os
import warnings
import sys
import time

import homogeneous_data

# from sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from Utils.graphUtils import create_graph

from utils import *
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer
from optim import adam
from model import init_params, build_model
from vocab import load_dictionary


# main trainer
def trainer(X,
            dim_word=620,  # word vector dimensionality
            dim=2400,  # the number of GRU units
            encoder='gru',
            decoder='gru',
            max_epochs=5,
            dispFreq=1,
            decay_c=0.,
            grad_clip=5.,
            n_words=20000,
            maxlen_w=30,
            optimizer='adam',
            batch_size=16,  # lowered from 64...
            saveto=r'D:\Projects\skip-thoughts\models\my_uni_skip.npz',
            dictionary=r'D:\Projects\skip-thoughts\models\corpus\new_dict.pk1',
            saveFreq=1000,
            reload_=True):

    # Model options
    model_options = {'dim_word': dim_word, 'dim': dim, 'encoder': encoder, 'decoder': decoder, 'max_epochs': max_epochs,
                     'dispFreq': dispFreq, 'decay_c': decay_c, 'grad_clip': grad_clip, 'n_words': n_words,
                     'maxlen_w': maxlen_w, 'optimizer': optimizer, 'batch_size': batch_size, 'saveto': saveto,
                     'dictionary': dictionary, 'saveFreq': saveFreq, 'reload_': reload_}

    print model_options

    x_graph = []
    y_graph = []
    temp_x_graph = []
    temp_y_graph = []

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'reloading...' + saveto
        with open('%s.pkl'%saveto, 'rb') as f:
            models_options = pkl.load(f)

    # load dictionary
    print 'Loading dictionary...'
    worddict = load_dictionary(dictionary)
    # Inverse dictionary
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, x, x_mask, y, y_mask, z, z_mask, \
          opt_ret, \
          cost = \
          build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask, z, z_mask]

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = function(inps, cost, profile=False)
    print 'Done'

    # weight decay, if applicable
    if decay_c > 0.:
        decay_c = shared(float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    f_cost = function(inps, cost, profile=False)
    print 'Done'

    print 'Done'
    print 'Building f_grad...',
    grads = grad(cost, wrt=itemlist(tparams))
    f_grad_norm = function(inps, [(g ** 2).sum() for g in grads], profile=False)
    f_weight_norm = function([], [(t ** 2).sum() for k, t in tparams.iteritems()], profile=False)

    if grad_clip > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(switch(g2 > (grad_clip ** 2),
                                    g / sqrt(g2) * grad_clip,
                                    g))
        grads = new_grads

    lr = scalar(name='lr')
    print 'Building optimizers...',
    # (compute gradients), (updates parameters)
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)

    print 'Optimization'

    # Each sentence in the minibatch have same length (for encoder)
    trainX = homogeneous_data.grouper(X)
    train_iter = homogeneous_data.HomogeneousData(trainX, batch_size, maxlen_w)

    uidx = 0
    lrate = 0.01
    for eidx in xrange(max_epochs):
        n_samples = 0

        print 'Epoch ', eidx

        for x, y, z in train_iter:
            n_samples += len(x)
            uidx += 1

            x, x_mask, y, y_mask, z, z_mask = homogeneous_data.prepare_data(x, y, z, worddict, maxlen_w, n_words)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen_w
                uidx -= 1
                continue

            ud_start = time.time()
            cost = f_grad_shared(x, x_mask, y, y_mask, z, z_mask)
            f_update(lrate)
            ud = time.time() - ud_start

            if isnan(cost) or isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            if mod(uidx, saveFreq) == 0:
                print 'Saving...',

                params = unzip(tparams)
                savez(saveto, history_errs=[], **params)
                pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                print 'Done'

            temp_x_graph.append(cost)  # Creating graphs for training visualisation
            temp_y_graph.append(uidx)

            if mod(uidx, 100) == 0:
                x_graph.append(mean(temp_x_graph))
                y_graph.append(max(temp_y_graph))
                print 'Saving graph values'

                #  Wipe older iterations
                temp_x_graph = []
                temp_y_graph = []

            if mod(uidx, 50000) == 0:
                create_graph(x_graph, y_graph, "Corpus: 49")
                print 'Generating Graph'

        print 'Seen %d samples' % n_samples


if __name__ == '__main__':
    pass


