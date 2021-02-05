
# coding: utf-8

# In[1]:

#!/usr/bin/env python2
#Inspire du fichier train_fcn8.py

import os


import argparse
import time
from getpass import getuser
from distutils.dir_util import copy_tree
import pickle

import numpy as np
import random
import theano
import theano.tensor as T
from theano import config
import lasagne
from lasagne.regularization import regularize_network_params
from lasagne.objectives import squared_error

import PIL.Image as Image
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import gridspec

#from fcn_1D_general import buildFCN_1D
from metrics import jaccard, accuracy, crossentropy, weighted_crossentropy

# from data_loader.cortical_layers import CorticalLayersDataset
from data_loader.cortical_layers_w_regions_kfold import CorticalLayersDataset

from offset_model_1path import build_offset_model
from profile_functions import profile2indices

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-nf','--n_filters', help='Number of filters', default=64, type=int)
parser.add_argument('-fs','--filter_size', help='Filter size', default=25, type=int)
parser.add_argument('-d','--depth', help='Number of layers', default=8, type=int)
parser.add_argument('-wd','--weight_decay', help='Weight decay', default=0.001, type=float)
parser.add_argument('-ne','--num_epochs', help='Number of epochs', default=500, type=int)
parser.add_argument('-p','--patience', help='Patience', default=50, type=int)
parser.add_argument('-lr','--learning_rate', help='Initial learning rate', default=0.0005, type=float)
parser.add_argument('-sr','--smooth_or_raw', help='Smooth or raw', default="both", type=str)
parser.add_argument('-nl','--number_layers', help='Number of layers of the dataset', default=6, type=int)
parser.add_argument('-kf','--k_fold', help='Number of folds', default=10, type=int)
parser.add_argument('-vf','--val_fold', help='Validation fold', default=0, type=int)
parser.add_argument('-tf','--test_fold', help='Test fold', default=1, type=int)
parser.add_argument('-bs','--batch_size', help='Batch size', default=1000, type=int)
parser.add_argument('-hs','--hidden_size', help='Hidden size', default=512, type=int)
parser.add_argument('-hl','--hidden_layers', help='Hidden layers', default=2, type=int)
parser.add_argument('-pool','--poolings', nargs='+', help='List of poolings', default=[0,2,0,2,0], type=int)
parser.add_argument('-dr','--dropout', help='Dropout rate for hidden layers', default=0.5, type=float)

args = parser.parse_args()
print(args)

def row2offset(row, n_transitions):
    '''Recieves a profile and converts it to offsets'''
    out = np.zeros((n_transitions), dtype=np.int32)

    # Basically we want to count the number of appearances of each layer, but
    # we need to remove the final padding of 0 because we don't want to count
    # them as 0

    # Find where it transitions from 6 to padding of 0
    res = (row[:-1]==6) & (row[1:]==0)
    slice_idx = np.nonzero(res)

    # remove the 0 padding at the end
    if len(slice_idx[0]):
        sliced = row[:slice_idx[0][0]+1]
    else:
        sliced = row

    # count
    for j in range(n_transitions):
        out[j] = (sliced == j).sum()

    return out

def segmentation_to_offsets(y_true, n_transitions):
    '''Converts segmentation labels to offsets'''

    # apply the function to each row (a row is a sample)
    out = np.zeros((y_true.shape[0], n_transitions), dtype=np.int32)
    for i in range(out.shape[0]):
        out[i, :] = row2offset(y_true[i,:], n_transitions)
    return out

def offsets_to_segmentation(offsets, profile_length):
    '''Converts the predicted outputs to a segmentation output'''
    out = np.zeros((offsets.shape[0], profile_length), dtype=np.int32)
    for i in range(offsets.shape[0]):
        seg = []
        pred_offsets = np.floor(np.maximum(offsets[i,:], 0)).astype(int)
        for j in range(offsets.shape[1]):
            seg += [j]*pred_offsets[j]

        # keep only 200 values if seg is longer
        seg = seg[:profile_length]
        # add 0 to the positions at the end
        seg += [0]*(profile_length-len(seg))
        out[i,:] = np.array(seg)

    return out

_FLOATX = config.floatX

SAVEPATH = '/Tmp/cucurulg/cortical_layers'
LOADPATH = '/data/lisatmp4/cucurulg/cortical_layers'
WEIGHTS_PATH = LOADPATH

#Model hyperparameters
n_filters = args.n_filters # 64
filter_size = [args.filter_size] # [25]#[7,15,25,49]
depth = args.depth #8
data_augmentation={} #{'horizontal_flip': True, 'fill_mode':'constant'}
block = 'bn_relu_conv'

# Define the structure. Each item represents a convolutional layer, and the
# number is the pooling stride to be applied in each layer. 0 means no pooling
poolings = args.poolings # [0,2,0,2,0]
n_hidden = args.hidden_layers # 2 # number of fully connected layers
hidden_size = args.hidden_size # 1024 # number of units in each fully connected layer
dropout_p = args.dropout

#Training loop hyperparameters
weight_decay= args.weight_decay # 0.001
num_epochs= args.num_epochs # 500
max_patience= args.patience # 50
resume=False
learning_rate_value = args.learning_rate # 0.0005 #learning rate is defined below as a theano variable.

#Hyperparameters for the dataset loader
batch_size=[args.batch_size,args.batch_size,1] # [1000, 1000, 1]
smooth_or_raw = args.smooth_or_raw # 'both'
shuffle_at_each_epoch = True
minibatches_subset = 0
n_layers = args.number_layers # 6
kfold = args.k_fold # 8
val_fold = args.val_fold # 0
test_fold = args.test_fold # 1

# save models every N epochs
save_every = 20

#
# Prepare load/save directories
#

savepath=SAVEPATH
loadpath=LOADPATH

exp_name = 'offset_model'
exp_name += '_lrate=' + str(learning_rate_value)
exp_name += '_fil=' + str(n_filters)
exp_name += '_fsizes=' + str(filter_size)
exp_name += '_hlayers=' + str(n_hidden)
exp_name += '_hsize=' + str(hidden_size)
exp_name += '_dr=' + str(dropout_p)
exp_name += '_pools=' + str(poolings)
exp_name += '_data=' + smooth_or_raw
exp_name += '_decay=' + str(weight_decay)
exp_name += '_pat=' + str(max_patience)
exp_name += '_kfold=' + str(kfold)
exp_name += '_val=' + str(val_fold)
exp_name += '_test=' + str(test_fold)
exp_name += ('_noshuffle'+str(minibatches_subset)+'batch') if not shuffle_at_each_epoch else ''
#exp_name += 'test'

dataset = str(n_layers)+'cortical_layers_all'
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

#
# Define symbolic variables
#
input_var = T.tensor3('input_var') #n_example*nb_in_channels*n_transitions
target_var = T.imatrix('target_var') #n_example*n_transitions
target_seg = T.ivector('target_var') #n_example*n_transitions
predicted_seg = T.ivector('target_var') #n_example*n_transitions

learn_step=  theano.shared(np.array(learning_rate_value, dtype=theano.config.floatX))

#
# Build dataset iterator
#

if smooth_or_raw =='both':
    nb_in_channels = 2
    use_threads = False
else:
    nb_in_channels = 1
    use_threads = True

train_iter = CorticalLayersDataset(
    which_set='train',
    smooth_or_raw = smooth_or_raw,
    batch_size=batch_size[0],
    data_augm_kwargs=data_augmentation,
    shuffle_at_each_epoch = shuffle_at_each_epoch,
    return_one_hot=False,
    return_01c=False,
    return_list=False,
    use_threads=use_threads,
    preload=True,
    n_layers=n_layers,
    kfold=kfold, # if None, kfold = number of regions (so there is one fold per region)
    val_fold=val_fold, # it will use the first fold for validation
    test_fold=test_fold) # this fold will not be used to train nor to validate

val_iter = CorticalLayersDataset(
    which_set='valid',
    smooth_or_raw = smooth_or_raw,
    batch_size=batch_size[1],
    shuffle_at_each_epoch = shuffle_at_each_epoch,
    return_one_hot=False,
    return_01c=False,
    return_list=False,
    use_threads=use_threads,
    preload=True,
    n_layers=n_layers,
    kfold=kfold, # if None, kfold = number of regions (so there is one fold per region)
    val_fold=val_fold, # it will use the first fold for validation
    test_fold=test_fold) # this fold will not be used to train nor to validate

test_iter = None

n_batches_train = train_iter.nbatches
n_batches_val = val_iter.nbatches
n_batches_test = test_iter.nbatches if test_iter is not None else 0
n_classes = train_iter.non_void_nclasses
void_labels = train_iter.void_labels


#
# Build network
#


# Define model
n_transitions = 7 # 7 transitions (0-1-2-3-4-5-6-0)
simple_net_output, net = build_offset_model(input_var,
                    filter_size = filter_size,
                    n_filters = n_filters,
                    poolings = poolings,
                    n_hidden = n_hidden,
                    hidden_size = hidden_size,
                    block= block,
                    nb_in_channels = nb_in_channels,
                    n_classes = n_transitions,
                    dropout_p=dropout_p)
#
# Define and compile theano functions
#

print "Defining and compiling training functions"

predictions = lasagne.layers.get_output(simple_net_output)
predictions = T.concatenate(predictions, axis=1)
loss = squared_error(predictions, target_var)
loss = loss.mean()

if weight_decay > 0:
    weightsl2 = regularize_network_params(
        simple_net_output, lasagne.regularization.l2)
    loss += weight_decay * weightsl2

# train_acc, train_sample_acc = accuracy(prediction, target_var, void_labels)

params = lasagne.layers.get_all_params(simple_net_output, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=learn_step)

train_fn = theano.function([input_var, target_var], [loss, predictions],
                           updates=updates)#, profile=True)

# compute jaccard with the predicted segmentation obtained by converting
# the offsets to a segmentation
jacc = jaccard(predicted_seg, target_seg, n_classes)
jacc_fn = theano.function([predicted_seg, target_seg], jacc)

print "Done"


print "Defining and compiling valid functions"
valid_predictions = lasagne.layers.get_output(simple_net_output,
                                            deterministic=True)
valid_predictions = T.concatenate(valid_predictions, axis=1)
valid_loss = squared_error(predictions, target_var)
valid_loss = valid_loss.mean()

valid_fn = theano.function([input_var, target_var],
                           [valid_loss, valid_predictions])#,profile=True)
print "Done"


#whether to plot labels prediction or not during training
#(1 random example of the last minibatch for each epoch)
plot_results_train = False #from the training set
plot_results_valid = False #from the validation set

# Train loop
#
err_train = []
acc_train = []
jacc_train = []
sample_acc_train_tot = []
worse_indices_train = []
already_seen_idx = []

err_valid = []
acc_valid = []
jacc_valid = []
sample_acc_valid_tot = []
patience = 0
worse_indices_valid =[]

# Training main loop
print "Start training"

for epoch in range(num_epochs):
    #learn_step.set_value((learn_step.get_value()*0.99).astype(theano.config.floatX))

    # Single epoch training and validation
    start_time = time.time()
    #Cost train and acc train for this epoch
    cost_train_epoch = 0
    acc_train_epoch = 0
    jacc_train_epoch = np.zeros((2, n_classes))
    # worse_indices_train_epoch = []

    for i in range(n_batches_train):
        # Get minibatch (comment the next line if only 1 minibatch in training)
        train_batch = train_iter.next()
        X_train_batch, L_train_batch = train_batch['data'], train_batch['labels']

        offsets = segmentation_to_offsets(L_train_batch, n_transitions)

        # L_train_batch = np.reshape(L_train_batch, np.prod(L_train_batch.shape))

        # Training step
        cost_train_batch, pred = train_fn(
                            X_train_batch, offsets)

        # compute accuracy and jaccard according to the
        pred_seg = offsets_to_segmentation(pred, 200)
        eqs = (L_train_batch == pred_seg)
        acc_train_batch = float(eqs.sum()) / np.prod(pred_seg.shape)

        jacc_train_batch = jacc_fn(np.reshape(pred_seg, np.prod(pred_seg.shape)), np.reshape(L_train_batch, np.prod(L_train_batch.shape)))

        # sample_acc_train_batch_mean = [np.mean([(i>=ratio)
        #                         for i in sample_acc_train_batch]) for ratio in ratios]

        #print i, 'training batch cost : ', cost_train_batch, ' batch accuracy : ', acc_train_batch

        #Update epoch results
        cost_train_epoch += cost_train_batch
        acc_train_epoch += acc_train_batch
        jacc_train_epoch += jacc_train_batch
        # sample_acc_train_epoch += sample_acc_train_batch_mean

    #Add epoch results
    err_train += [cost_train_epoch/n_batches_train]
    acc_train += [acc_train_epoch/n_batches_train]
    jacc_perclass_train = jacc_train_epoch[0, :] / jacc_train_epoch[1, :]
    jacc_train += [np.mean(jacc_perclass_train)]
    # sample_acc_train_tot += [sample_acc_train_epoch/n_batches_train]

    # Validation
    cost_val_epoch = 0
    acc_val_epoch = 0
    jacc_val_epoch = np.zeros((2, n_classes))
    # worse_indices_val_epoch = []

    for i in range(n_batches_val):

        # Get minibatch (comment the next line if only 1 minibatch in training)
        val_batch = val_iter.next()
        X_val_batch, L_val_batch = val_batch['data'], val_batch['labels']

        val_offsets = segmentation_to_offsets(L_val_batch, n_transitions)

        # L_val_batch = np.reshape(L_val_batch, np.prod(L_val_batch.shape))

        # Validation step
        cost_val_batch, val_pred = valid_fn(X_val_batch, val_offsets)
        #print i, 'validation batch cost : ', cost_val_batch, ' batch accuracy : ', acc_val_batch

        # compute accuracy and jaccard according to the
        pred_seg = offsets_to_segmentation(val_pred, 200)
        eqs = (L_val_batch == pred_seg)
        acc_val_batch = float(eqs.sum()) / np.prod(pred_seg.shape)

        jacc_val_batch = jacc_fn(np.reshape(pred_seg, np.prod(pred_seg.shape)), np.reshape(L_val_batch, np.prod(L_val_batch.shape)))

        # sample_acc_valid_batch_mean = [np.mean([(i>=ratio)
        #                         for i in sample_acc_valid_batch]) for ratio in ratios]

        #Update epoch results
        cost_val_epoch += cost_val_batch
        acc_val_epoch += acc_val_batch
        # sample_acc_valid_epoch += sample_acc_valid_batch_mean
        jacc_val_epoch += jacc_val_batch


    #Add epoch results
    err_valid += [cost_val_epoch/n_batches_val]
    acc_valid += [acc_val_epoch/n_batches_val]
    # sample_acc_valid_tot += [sample_acc_valid_epoch/n_batches_val]
    jacc_perclass_valid = jacc_val_epoch[0, :] / jacc_val_epoch[1, :]
    jacc_valid += [np.mean(jacc_perclass_valid)]


    #Print results (once per epoch)

    out_str = "EPOCH %i: Avg cost train %f, avg acc train %f, jacc train %f, cost val %f, avg acc val %f, jacc val %f took %f s"
    out_str = out_str % (epoch, err_train[epoch],
                         acc_train[epoch],
                         jacc_train[epoch],
                         err_valid[epoch],
                         acc_valid[epoch],
                         jacc_valid[epoch],
                         time.time()-start_time)
    # out_str2 = 'Per sample accuracy (ratios ' + str(ratios) + ') '
    # out_str2 += ' train ' +str(sample_acc_train_tot[epoch])
    # out_str2 += ' valid ' + str(sample_acc_valid_tot[epoch])
    print out_str
    # print out_str2



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
                 err_train=err_train, acc_train=acc_train, jacc_train=jacc_train,
                 err_valid=err_valid, acc_valid=acc_valid, jacc_valid=jacc_valid)
        np.savez(os.path.join(savepath, 'new_fcn1D_model_last.npz'),
                 *lasagne.layers.get_all_param_values(simple_net_output))
        np.savez(os.path.join(savepath , "fcn1D_errors_last.npz"),
                 err_train=err_train, acc_train=acc_train, jacc_train=jacc_train,
                 err_valid=err_valid, acc_valid=acc_valid, jacc_valid=jacc_valid)
    else:
        patience += 1
        if epoch % save_every == 0:
            print('saving last model')
            np.savez(os.path.join(savepath, 'new_fcn1D_model_last.npz'),
                     *lasagne.layers.get_all_param_values(simple_net_output))
            np.savez(os.path.join(savepath , "fcn1D_errors_last.npz"),
                     err_train=err_train, acc_train=acc_train, jacc_train=jacc_train,
                     err_valid=err_valid, acc_valid=acc_valid, jacc_valid=jacc_valid)
    # Finish training if patience has expired or max nber of epochs reached

    if patience == max_patience or epoch == num_epochs-1:
        if savepath != loadpath:
            print('Copying model and other training files to {}'.format(loadpath))
            copy_tree(savepath, loadpath)
        break
