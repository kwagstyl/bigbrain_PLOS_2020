#!/usr/bin/env python2
# coding: utf-8

#Inspire du fichier train_fcn8.py

import os


import argparse
import time
from getpass import getuser
from distutils.dir_util import copy_tree

import numpy as np
import random
import theano
# theano.config.dnn.conv.algo_fwd = 'time_once'
# theano.config.dnn.conv.algo_bwd_filter = 'time_once'
# theano.config.dnn.conv.algo_bwd_data = 'time_once'

import theano.tensor as T
from theano import config
import lasagne
from lasagne.regularization import regularize_network_params
from lasagne.objectives import categorical_crossentropy

import PIL.Image as Image

from fcn_1D_general import buildFCN_1D
from metrics import jaccard, accuracy, crossentropy
from data_loader.cortical_layers import Cortical4LayersDataset, Cortical6LayersDataset
from simple_model import build_simple_model


# In[2]:

_FLOATX = config.floatX

SAVEPATH = '/Tmp/larocste/cortical_layers'
LOADPATH = '/data/lisatmp4/larocste/cortical_layers'
WEIGHTS_PATH = LOADPATH


# ## Hyperparameters
#
# ### Model hyperparameters
# - n_filters : int, nb of filters for each convLayer
# - filter_size : list of odd int (to fit with the pad='same'), len(filter_size) = nb of convLayer in simple_model, each of these layer with the corresponding filter_size
# - depth : int, depth of the network (how many stacked convolution)
#
# ### Training loop parameters
# - weight_decay : not implemented yet
# - num_epochs : int, max number of epochs
# - max_patience : int, max nb of epochs without improvement in the jaccard accuracy (jacc_valid) on the validation set
# - learning rate : defined later as a theano shared variable
#
# ### Hyperparameters for the dataset loader
# - batch_size=[training_batch_size, valid_batch_size, test_batch_size]
# - smooth_or_raw : 'smooth' or 'raw', whether to use smooth OR raw data
# - shuffle_at_each_epoch : boolean (keep it to True)
# - minibatches_subset : int, if>0 : get only that number of minibatch instead of all training dataset.
#

# In[3]:

#Model hyperparameters
n_filters = 4
filter_size = [5] #[7,15,25,49]
depth  = 2
data_augmentation={} #{'horizontal_flip': True, 'fill_mode':'constant'}

#Training loop hyperparameters
weight_decay=0.001
num_epochs=500
max_patience=25
resume=False
learning_rate_value = 0.0005 #learning rate is defined below as a theano variable.
ratios=[0.80, 0.85, 0.90]


#Hyperparameters for the dataset loader
batch_size=[500,500,1]
smooth_or_raw = 'both'
shuffle_at_each_epoch = True
minibatches_subset = 0

n_layers = 6



print 'learning rate=' + str(learning_rate_value)
print 'n_filters=' + str(n_filters)
print 'filter_sizes=' + str(filter_size)
print 'depth=' + str(depth)
print 'batchsize=', batch_size
print 'smooth or raw? ' + smooth_or_raw
print 'weight_decay=' + str(weight_decay)
print 'patience=' + str(max_patience)

# In[4]:

#
# Prepare load/save directories
#

savepath=SAVEPATH
loadpath=LOADPATH

exp_name = 'simple_model'
exp_name += '_lrate=' + str(learning_rate_value)
exp_name += '_fil=' + str(n_filters)
exp_name += '_fsizes=' + str(filter_size)
exp_name += '_depth=' + str(depth)
exp_name += '_data=' + smooth_or_raw
exp_name += '_decay=' + str(weight_decay)
exp_name += '_pat=' + str(max_patience)
exp_name += 'nothreads'
exp_name += ('_noshuffle'+str(minibatches_subset)+'batch') if not shuffle_at_each_epoch else ''
#exp_name += 'test'

dataset = 'cortical_layers'
savepath = os.path.join(savepath, dataset, exp_name)
loadpath = os.path.join(loadpath, dataset, exp_name)
print 'Savepath : '
print savepath
print 'Loadpath : '
print loadpath

if not os.path.exists(savepath):
    os.makedirs(savepath)
else:
    print('\033[93m The following folder already exists {}. '
          'It will be overwritten in a few seconds...\033[0m'.format(
              savepath))

print('Saving directory : ' + savepath)
with open(os.path.join(savepath, "config.txt"), "w") as f:
    for key, value in locals().items():
        f.write('{} = {}\n'.format(key, value))


# In[ ]:




# In[5]:

#
# Define symbolic variables
#
input_var = T.tensor3('input_var') #n_example*nb_in_channels*ray_size
target_var = T.ivector('target_var') #n_example*ray_size

learn_step=  theano.shared(np.array(learning_rate_value, dtype=theano.config.floatX))


# In[6]:

#
# Build dataset iterator
#

if smooth_or_raw =='both':
    nb_in_channels = 2
    use_threads = False
else:
    nb_in_channels = 1
    use_threads = True
if n_layers ==4 :
    train_iter = Cortical4LayersDataset(
        which_set='train',
        smooth_or_raw = smooth_or_raw,
        batch_size=batch_size[0],
        data_augm_kwargs=data_augmentation,
        shuffle_at_each_epoch = shuffle_at_each_epoch,
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False)

    val_iter = Cortical4LayersDataset(
        which_set='valid',
        smooth_or_raw = smooth_or_raw,
        batch_size=batch_size[1],
        shuffle_at_each_epoch = shuffle_at_each_epoch,
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False)

    test_iter = None
elif n_layers ==6:
    train_iter = Cortical6LayersDataset(
        which_set='train',
        smooth_or_raw = smooth_or_raw,
        batch_size=batch_size[0],
        data_augm_kwargs=data_augmentation,
        shuffle_at_each_epoch = shuffle_at_each_epoch,
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False)

    val_iter = Cortical6LayersDataset(
        which_set='valid',
        smooth_or_raw = smooth_or_raw,
        batch_size=batch_size[1],
        shuffle_at_each_epoch = shuffle_at_each_epoch,
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False)

    test_iter = None




n_batches_train = train_iter.nbatches
n_batches_val = val_iter.nbatches
n_batches_test = test_iter.nbatches if test_iter is not None else 0
n_classes = train_iter.non_void_nclasses
void_labels = train_iter.void_labels
#nb_in_channels = train_iter.data_shape[0]


# In[7]:

#
# Build network
#
simple_net_output, net = build_simple_model(input_var,
                    filter_size = filter_size,
                    n_filters = n_filters,
                    depth = depth,
                    nb_in_channels = nb_in_channels,
                    n_classes = n_classes)

#simple_net_output = last layer of the simple_model net
#net = dictionary containing the names of every layer (used to visualize data)

# To print each layer, uncomment this:
# lays = lasagne.layers.get_all_layers(simple_net_output)
# for l in lays:
#     print l, l.output_shape
#     #print  simple_net_output[l], simple_net_output[l].output_shape, l
# print '---------------------------'
# print 'simple_net_output :', simple_net_output
# print '---------------------------'
# #print 'net :', net


# In[ ]:




# In[8]:

#
# Define and compile theano functions
#
print "Defining and compiling training functions"

prediction = lasagne.layers.get_output(simple_net_output[0])
loss = categorical_crossentropy(prediction, target_var)
loss = loss.mean()

if weight_decay > 0:
    weightsl2 = regularize_network_params(
        simple_net_output, lasagne.regularization.l2)
    loss += weight_decay * weightsl2

train_acc, train_sample_acc = accuracy(prediction, target_var, void_labels)

params = lasagne.layers.get_all_params(simple_net_output, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=learn_step)

train_fn = theano.function([input_var, target_var], [loss, train_acc, train_sample_acc],
                           updates=updates)#, profile=True)

print "Done"


# In[9]:

print "Defining and compiling valid functions"
valid_prediction = lasagne.layers.get_output(simple_net_output[0],
                                            deterministic=True)
valid_loss = categorical_crossentropy(valid_prediction, target_var)
valid_loss = valid_loss.mean()
#valid_loss = crossentropy(valid_prediction, target_var, void_labels)
valid_acc, valid_sample_acc = accuracy(valid_prediction, target_var, void_labels)
valid_jacc = jaccard(valid_prediction, target_var, n_classes)

valid_fn = theano.function([input_var, target_var],
                           [valid_loss, valid_acc, valid_sample_acc, valid_jacc])#,profile=True)
print "Done"


# Uncomment this if only 1 minibatch (smaller dataset)
# and comment 2 lines in training loop to avoid getting new minibatches

# X_train_batch, L_train_batch = train_iter.next()
# X_val_batch, L_val_batch = val_iter.next()
# minibatches_subset = 1



#
# Train loop
#
err_train = []
acc_train = []
sample_acc_train_tot = []

err_valid = []
acc_valid = []
jacc_valid = []
sample_acc_valid_tot = []
patience = 0

# Training main loop
print "Start training"

for epoch in range(num_epochs):
    #learn_step.set_value((learn_step.get_value()*0.99).astype(theano.config.floatX))

    # Single epoch training and validation
    start_time = time.time()
    #Cost train and acc train for this epoch
    cost_train_epoch = 0
    acc_train_epoch = 0
    sample_acc_train_epoch = np.array([0.0 for i in range(len(ratios))])

    # Train
    if minibatches_subset > 0:
        n_batches_val = minibatches_subset
        n_batches_train = minibatches_subset


    for i in range(n_batches_train):
        # Get minibatch (comment the next line if only 1 minibatch in training)
        X_train_batch, L_train_batch = train_iter.next()
        L_train_batch = np.reshape(L_train_batch, np.prod(L_train_batch.shape))

        # Training step
        cost_train_batch, acc_train_batch, sample_acc_train_batch = train_fn(X_train_batch, L_train_batch)
        sample_acc_train_batch_mean = [np.mean([(i>=ratio)
                                for i in sample_acc_train_batch]) for ratio in ratios]
        #print i, 'training batch cost : ', cost_train_batch, ' batch accuracy : ', acc_train_batch

        #Update epoch results
        cost_train_epoch += cost_train_batch
        acc_train_epoch += acc_train_batch
        sample_acc_train_epoch += sample_acc_train_batch_mean

    #Add epoch results
    err_train += [cost_train_epoch/n_batches_train]
    acc_train += [acc_train_epoch/n_batches_train]
    sample_acc_train_tot += [sample_acc_train_epoch/n_batches_train]


    # Validation
    cost_val_epoch = 0
    acc_val_epoch = 0
    sample_acc_valid_epoch = np.array([0.0 for i in range(len(ratios))])
    jacc_val_epoch = np.zeros((2, n_classes))


    for i in range(n_batches_val):
        # Get minibatch (comment the next line if only 1 minibatch in training)
        X_val_batch, L_val_batch = val_iter.next()
        L_val_batch = np.reshape(L_val_batch, np.prod(L_val_batch.shape))


        # Validation step
        cost_val_batch, acc_val_batch, sample_acc_valid_batch, jacc_val_batch = valid_fn(X_val_batch, L_val_batch)
        #print i, 'validation batch cost : ', cost_val_batch, ' batch accuracy : ', acc_val_batch
        sample_acc_valid_batch_mean = [np.mean([(i>=ratio)
                                for i in sample_acc_valid_batch]) for ratio in ratios]

        #Update epoch results
        cost_val_epoch += cost_val_batch
        acc_val_epoch += acc_val_batch
        sample_acc_valid_epoch += sample_acc_valid_batch_mean
        jacc_val_epoch += jacc_val_batch

    #Add epoch results
    err_valid += [cost_val_epoch/n_batches_val]
    acc_valid += [acc_val_epoch/n_batches_val]
    sample_acc_valid_tot += [sample_acc_valid_epoch/n_batches_val]
    jacc_perclass_valid = jacc_val_epoch[0, :] / jacc_val_epoch[1, :]
    jacc_valid += [np.mean(jacc_perclass_valid)]


    #Print results (once per epoch)
    print "----------------------------------------------------------"
    out_str = "EPOCH %i: Avg cost train %f, acc train %f"+        ", cost val %f, acc val %f, jacc val %f took %f s"
    out_str = out_str % (epoch, err_train[epoch],
                         acc_train[epoch],
                         err_valid[epoch],
                         acc_valid[epoch],
                         jacc_valid[epoch],
                         time.time()-start_time)
    out_str2 = 'Per sample train accuracy (ratios ' + str(ratios) + ') ' + str(sample_acc_train_tot[epoch])
    out_str3 = 'Per sample valid accuracy (ratios ' + str(ratios) + ') ' + str(sample_acc_valid_tot[epoch])
    print out_str
    print out_str2
    print out_str3



    # Early stopping and saving stuff

    with open(os.path.join(savepath, "fcn1D_output.log"), "a") as f:
        f.write(out_str + "\n")

    if epoch == 0:
        best_jacc_val = jacc_valid[epoch]
    elif epoch > 1 and jacc_valid[epoch] > best_jacc_val:
        print('saving best (and last) model')
        best_jacc_val = jacc_valid[epoch]
        patience = 0
        np.savez(os.path.join(savepath, 'new_fcn1D_model_best.npz'),
                 *lasagne.layers.get_all_param_values(simple_net_output))
        np.savez(os.path.join(savepath , "fcn1D_errors_best.npz"),
                 err_train=err_train, acc_train=acc_train,
                 err_valid=err_valid, acc_valid=acc_valid, jacc_valid=jacc_valid)
        np.savez(os.path.join(savepath, 'new_fcn1D_model_last.npz'),
                 *lasagne.layers.get_all_param_values(simple_net_output))
        np.savez(os.path.join(savepath , "fcn1D_errors_last.npz"),
                 err_train=err_train, acc_train=acc_train,
                 err_valid=err_valid, acc_valid=acc_valid, jacc_valid=jacc_valid)
    else:
        patience += 1
        print('saving last model')
        np.savez(os.path.join(savepath, 'new_fcn1D_model_last.npz'),
                 *lasagne.layers.get_all_param_values(simple_net_output))
        np.savez(os.path.join(savepath , "fcn1D_errors_last.npz"),
                 err_train=err_train, acc_train=acc_train,
                 err_valid=err_valid, acc_valid=acc_valid, jacc_valid=jacc_valid)
    # Finish training if patience has expired or max nber of epochs reached

    if patience == max_patience or epoch == num_epochs-1:
        if savepath != loadpath:
            print('Copying model and other training files to {}'.format(loadpath))
            copy_tree(savepath, loadpath)
        break
