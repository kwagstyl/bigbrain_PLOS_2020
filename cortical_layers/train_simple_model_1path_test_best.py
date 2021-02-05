
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
from lasagne.objectives import categorical_crossentropy

import PIL.Image as Image
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
from scipy import interpolate

#from fcn_1D_general import buildFCN_1D
from metrics import jaccard, accuracy, crossentropy, weighted_crossentropy

# from data_loader.cortical_layers import CorticalLayersDataset
from data_loader.cortical_layers_w_regions_kfold_val_train_test import CorticalLayersDataset

from simple_model_1path import build_simple_model
from profile_functions import profile2indices

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-nf','--n_filters', help='Number of filters', default=64, type=int)
parser.add_argument('-fs','--filter_size', help='Filter size', default=25, type=int)
parser.add_argument('-d','--depth', help='Number of layers', default=8, type=int)
parser.add_argument('-wd','--weight_decay', help='Weight decay', default=0.001, type=float)
parser.add_argument('-sp','--smooth_penalty', help='Smooth penalty', default=0.0, type=float)
parser.add_argument('-ne','--num_epochs', help='Number of epochs', default=500, type=int)
parser.add_argument('-p','--patience', help='Patience', default=50, type=int)
parser.add_argument('-lr','--learning_rate', help='Initial learning rate', default=0.0005, type=float)
parser.add_argument('-sr','--smooth_or_raw', help='Smooth or raw', default="both", type=str)
parser.add_argument('-nl','--number_layers', help='Number of layers of the dataset', default=6, type=int)
parser.add_argument('-kf','--k_fold', help='Number of folds', default=10, type=int)
parser.add_argument('-vf','--val_fold', help='Validation fold', default=0, type=int)
parser.add_argument('-tf','--test_fold', help='Test fold', default=1, type=int)
parser.add_argument('-bs','--batch_size', help='Batch size', default=1000, type=int)
parser.add_argument('-cl','--clipping', help='Maximum allowable data clipping', default=0, type=int)
parser.add_argument('-id', '--input_data', help='input data prefix eg training_20_', default='training_', type=str)

args = parser.parse_args()
print(args)

# In[2]:

_FLOATX = config.floatX
SAVEPATH = '/data1/users/kwagstyl/bigbrain/cortical_layers'
LOADPATH = '/data1/users/kwagstyl/bigbrain/cortical_layers'
WEIGHTS_PATH = LOADPATH


# In[22]:

#Model hyperparameters
n_filters = args.n_filters # 64
filter_size = [args.filter_size] # [25]#[7,15,25,49]
depth = args.depth #8
data_augmentation={} #{'horizontal_flip': True, 'fill_mode':'constant'}
block = 'bn_relu_conv'

#Training loop hyperparameters
weight_decay= args.weight_decay # 0.001
smooth_penalty = args.smooth_penalty #0.005
num_epochs= args.num_epochs # 500
max_patience= args.patience # 50
resume=False
learning_rate_value = args.learning_rate # 0.0005 #learning rate is defined below as a theano variable.
max_clipping = args.clipping
prefix=args.input_data

#Hyperparameters for the dataset loader
batch_size=[args.batch_size,args.batch_size,1] # [1000, 1000, 1]
smooth_or_raw = args.smooth_or_raw # 'both'
shuffle_at_each_epoch = True
minibatches_subset = 0
n_layers = args.number_layers # 6
kfold = args.k_fold # 8
val_fold = args.val_fold # 0
test_fold = args.test_fold # 1


# In[4]:

#
# Prepare load/save directories
#

savepath=SAVEPATH
loadpath=LOADPATH

exp_name = 'simple_model'
exp_name += '_dataset' + prefix
exp_name += '_lrate=' + str(learning_rate_value)
exp_name += '_fil=' + str(n_filters)
exp_name += '_fsizes=' + str(filter_size)
exp_name += '_depth=' + str(depth)
exp_name += '_data=' + smooth_or_raw
exp_name += '_decay=' + str(weight_decay)
exp_name += '_smooth=' + str(smooth_penalty)
exp_name += '_pat=' + str(max_patience)
exp_name += '_kfold=' + str(kfold)
exp_name += '_val=' + str(val_fold)
exp_name += '_test=' + str(test_fold)
exp_name += '_batch_size=' + str(batch_size[0])
exp_name += ('_noshuffle'+str(minibatches_subset)+'batch') if not shuffle_at_each_epoch else ''

#exp_name += 'test'

dataset = str(n_layers)+'cortical_layers'
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
weight_vector = T.fvector('weight_vector')

learn_step=  theano.shared(np.array(learning_rate_value, dtype=theano.config.floatX))


# In[21]:

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
    prefix=prefix,
    smooth_or_raw = smooth_or_raw,
    batch_size=batch_size[0],
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
    prefix=prefix,
    smooth_or_raw = smooth_or_raw,
    batch_size=batch_size[1],
    return_one_hot=False,
    return_01c=False,
    return_list=False,
    use_threads=use_threads,
    preload=True,
    n_layers=n_layers,
    kfold=kfold, # if None, kfold = number of regions (so there is one fold per region)
    val_fold=val_fold, # it will use the first fold for validation
    test_fold=test_fold) # this fold will not be used to train nor to validate

test_iter = CorticalLayersDataset(
                                  which_set='test',
                                  prefix=prefix,
                                  smooth_or_raw = smooth_or_raw,
                                  batch_size=batch_size[1],
                                  return_one_hot=False,
                                  return_01c=False,
                                  return_list=False,
                                  use_threads=use_threads,
                                  preload=True,
                                  n_layers=n_layers,
                                  kfold=kfold, # if None, kfold = number of regions (so there is one fold per region)
                                  val_fold=val_fold, # it will use the first fold for validation
                                  test_fold=test_fold) # this fold will not be used to train nor to validate


n_batches_train = train_iter.nbatches
n_batches_val = val_iter.nbatches
n_batches_test = test_iter.nbatches if test_iter is not None else 0
n_classes = train_iter.non_void_nclasses
void_labels = train_iter.void_labels


#
# Build network
#
simple_net_output, net = build_simple_model(input_var,
                    filter_size = filter_size,
                    n_filters = n_filters,
                    depth = depth,
                    block= block,
                    nb_in_channels = nb_in_channels,
                    n_classes = n_classes)

#simple_net_output = last layer of the simple_model net


#
# Define and compile theano functions
#
#get weights
#Class=np.loadtxt('6layers_segmentation/training_cls.txt')
Class=np.loadtxt(os.path.join(LOADPATH,'6layers_segmentation/training_cls.txt'))
def compute_class_weights(Class):
    #get unique labels and number of pixels of each class
    unique, counts = np.unique(Class,return_counts=True)
    #calculate freq(c) number of pixels per class divided by the total number of pixels in images where c is present
    freq=counts.astype(float)/Class.size
    return np.median(freq)/freq

weights=compute_class_weights(Class)

# penalty to enforce smoothness
def smooth_convolution(prediction, n_classes):
    from lasagne.layers import Conv1DLayer as ConvLayer
    from lasagne.layers import DimshuffleLayer, ReshapeLayer
    prediction = ReshapeLayer(prediction, (-1, 200, n_classes))
    # channels first
    prediction = DimshuffleLayer(prediction, (0,2,1))

    input_size = lasagne.layers.get_output(prediction).shape
    # reshape to put each channel in the batch dimensions, to filter each
    # channel independently
    prediction = ReshapeLayer(prediction, (T.prod(input_size[0:2]),1,input_size[2]))

    trans_filter = np.tile(np.array([0,-1.,1.]).astype('float32'), (1,1,1))
    convolved = ConvLayer(prediction,
                    num_filters = 1,
                    filter_size = 3,
                    stride=1,
                    b = None,
                    nonlinearity=None,
                    W = trans_filter,
                    pad='same')

    # reshape back
    convolved = ReshapeLayer(convolved, input_size)

    return convolved

def clipping(x_train, l_train, max_clipping = 10):
    """ Clip profiles at either end by random amounts. Augments data"""
    k=-1
    for profile, label in zip(x_train, l_train):
        k+=1
        cut_top=np.random.randint(max_clipping)
        cut_bottom=np.random.randint(max_clipping)
        n=np.delete(profile,range(np.random.randint(cut_top)),0)
        n=np.delete(n,len(n)-np.arange(np.random.randint(cut_bottom)),0)
        l=np.delete(label,range(np.random.randint(cut_top)),0)
        l=np.delete(l,len(n)-np.arange(np.random.randint(cut_bottom)),0)
            #interpolate back to full length.
        x = np.linspace(0, len(n), x_train.shape[0])
        p[:,0]=np.interp(x,range(len(n)),n[:,0])
        p[:,1]=np.interp(x,range(len(n)),n[:,1])
        new_label=np.round(np.interp(x,range(len(n)),l))
        x_train[k]=p
        l_train[k]=new_label
    return x_train, l_train


print "Defining and compiling training functions"

convolved = smooth_convolution(simple_net_output[0], n_classes)

prediction, convolved = lasagne.layers.get_output([simple_net_output[0], convolved])
#loss = categorical_crossentropy(prediction, target_var)
#loss = loss.mean()

loss = weighted_crossentropy(prediction, target_var, weight_vector)
loss = loss.mean()


if weight_decay > 0:
    weightsl2 = regularize_network_params(
        simple_net_output, lasagne.regularization.l2)
    loss += weight_decay * weightsl2

if smooth_penalty > 0:
    smooth_cost = T.sum(abs(convolved), axis=(1,2))
    loss += smooth_penalty * smooth_cost.mean()

train_acc, train_sample_acc = accuracy(prediction, target_var, void_labels)

params = lasagne.layers.get_all_params(simple_net_output, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=learn_step)

train_fn = theano.function([input_var, target_var, weight_vector], [loss, train_acc, train_sample_acc, prediction],
                           updates=updates)#, profile=True)

print "Done"


# In[11]:

print "Defining and compiling valid functions"
valid_prediction = lasagne.layers.get_output(simple_net_output[0],
                                            deterministic=True)
#valid_loss = categorical_crossentropy(valid_prediction, target_var)
#valid_loss = valid_loss.mean()
valid_loss = weighted_crossentropy(valid_prediction, target_var, weight_vector)
valid_loss = valid_loss.mean()

#valid_loss = crossentropy(valid_prediction, target_var, void_labels)
valid_acc, valid_sample_acc = accuracy(valid_prediction, target_var, void_labels)
valid_jacc = jaccard(valid_prediction, target_var, n_classes)

valid_fn = theano.function([input_var, target_var, weight_vector],
                           [valid_loss, valid_acc, valid_sample_acc, valid_jacc])#,profile=True)
print "Done"
print "Defining and compiling test functions"
test_prediction = lasagne.layers.get_output(simple_net_output[0],
                                            deterministic=True)
test_loss = weighted_crossentropy(valid_prediction, target_var, weight_vector)
test_loss = test_loss.mean()
test_acc, test_acc_per_sample = accuracy(test_prediction, target_var, void_labels)
test_jacc = jaccard(test_prediction, target_var, n_classes)

test_fn = theano.function([input_var, target_var, weight_vector], [test_loss, test_acc,
                                                    test_jacc, test_acc_per_sample])
print "Done"

#whether to plot labels prediction or not during training
#(1 random example of the last minibatch for each epoch)
plot_results_train = False #from the training set
plot_results_valid = False #from the validation set

treshold = 0.7 # for extracting the very incorrect labelled samples
ratios=[0.80,0.85, 0.90] #ratios for the per sample accuracy


# In[ ]:

# Train loop
#
err_train = []
acc_train = []
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
    sample_acc_train_epoch = np.array([0.0 for i in range(len(ratios))])
    # worse_indices_train_epoch = []




    for i in range(n_batches_train):
        # Get minibatch (comment the next line if only 1 minibatch in training)
        train_batch = train_iter.next()
        X_train_batch, L_train_batch = train_batch['data'], train_batch['labels']
        if max_clipping > 0:
            X_train_batch, L_train_batch = clipping(X_train_batch, L_train_batch, max_clipping)
        L_train_batch = np.reshape(L_train_batch, np.prod(L_train_batch.shape))

        # Training step
        cost_train_batch, acc_train_batch, sample_acc_train_batch, pred = train_fn(
                            X_train_batch, L_train_batch, weights[L_train_batch].astype('float32'))

        sample_acc_train_batch_mean = [np.mean([(i>=ratio)
                                for i in sample_acc_train_batch]) for ratio in ratios]

        # worse_indices_train_batch = index_worse_than(sample_acc_train_batch,
        #                                   idx_train_batch, treshold=treshold)


        #print i, 'training batch cost : ', cost_train_batch, ' batch accuracy : ', acc_train_batch

        #Update epoch results
        cost_train_epoch += cost_train_batch
        acc_train_epoch += acc_train_batch
        sample_acc_train_epoch += sample_acc_train_batch_mean
        # worse_indices_train_epoch = np.hstack((worse_indices_train_epoch,worse_indices_train_batch))

    #Add epoch results
    err_train += [cost_train_epoch/n_batches_train]
    acc_train += [acc_train_epoch/n_batches_train]
    sample_acc_train_tot += [sample_acc_train_epoch/n_batches_train]
    # worse_indices_train += [worse_indices_train_epoch]

    # Validation
    cost_val_epoch = 0
    acc_val_epoch = 0
    sample_acc_valid_epoch = np.array([0.0 for i in range(len(ratios))])
    jacc_val_epoch = np.zeros((2, n_classes))
    # worse_indices_val_epoch = []


    for i in range(n_batches_val):

        # Get minibatch (comment the next line if only 1 minibatch in training)
        val_batch = val_iter.next()
        X_val_batch, L_val_batch = val_batch['data'], val_batch['labels']

        L_val_batch = np.reshape(L_val_batch, np.prod(L_val_batch.shape))


        # Validation step
        cost_val_batch, acc_val_batch, sample_acc_valid_batch, jacc_val_batch = valid_fn(X_val_batch, L_val_batch, weights[L_val_batch].astype('float32'))
        #print i, 'validation batch cost : ', cost_val_batch, ' batch accuracy : ', acc_val_batch


        sample_acc_valid_batch_mean = [np.mean([(i>=ratio)
                                for i in sample_acc_valid_batch]) for ratio in ratios]



        #Update epoch results
        cost_val_epoch += cost_val_batch
        acc_val_epoch += acc_val_batch
        sample_acc_valid_epoch += sample_acc_valid_batch_mean
        jacc_val_epoch += jacc_val_batch
        # worse_indices_val_epoch = np.hstack((worse_indices_val_epoch, worse_indices_val_batch))
        #


    #Add epoch results
    err_valid += [cost_val_epoch/n_batches_val]
    acc_valid += [acc_val_epoch/n_batches_val]
    sample_acc_valid_tot += [sample_acc_valid_epoch/n_batches_val]
    jacc_perclass_valid = jacc_val_epoch[0, :] / jacc_val_epoch[1, :]
    jacc_valid += [np.mean(jacc_perclass_valid)]
    # worse_indices_valid += [worse_indices_val_epoch]


    #Print results (once per epoch)

    out_str = "EPOCH %i: Avg cost train %f, acc train %f"+        ", cost val %f, acc val %f, jacc val %f took %f s"
    out_str = out_str % (epoch, err_train[epoch],
                         acc_train[epoch],
                         err_valid[epoch],
                         acc_valid[epoch],
                         jacc_valid[epoch],
                         time.time()-start_time)
    out_str2 = 'Per sample accuracy (ratios ' + str(ratios) + ') '
    out_str2 += ' train ' +str(sample_acc_train_tot[epoch])
    out_str2 += ' valid ' + str(sample_acc_valid_tot[epoch])
    print out_str
    print out_str2



    # Early stopping and saving stuff

    with open(os.path.join(savepath, "fcn1D_output.log"), "a") as f:
        f.write(out_str + "\n")

    if epoch == 0:
        best_jacc_val = jacc_valid[epoch]
    elif epoch > 1 and jacc_valid[epoch] > best_jacc_val:
        print('saving and testing best (and last) model')
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
                 
                 #test the best iteration and save results for comparisons
        for i in range(n_batches_test):
            test_batch = test_iter.next()
            X_train_batch, L_test_batch = test_batch['data'], test_batch['labels']
        
            L_test_batch = np.reshape(L_test_batch, np.prod(L_test_batch.shape))
        
        # Testing step
            cost_test_batch, acc_test_batch, sample_acc_test_batch, jacc_test_batch = test_fn(
                                    X_test_batch, L_test_batch, weights[L_test_batch].astype('float32'))
            sample_acc_test_batch_mean = [np.mean([(i>=ratio)
                                    for i in sample_acc_test_batch]) for ratio in ratios]
                                    
            cost_test_epoch += cost_test_batch
            acc_test_epoch += acc_test_batch
            sample_acc_test_epoch += sample_acc_test_batch_mean
            jacc_test_epoch += jacc_test_batch
        err_test += [cost_test_epoch/n_batches_test]
        acc_test += [acc_test_epoch/n_batches_test]
        sample_acc_test_tot += [sample_acc_test_epoch/n_batches_test]
        jacc_perclass_test = jacc_test_epoch[0, :] / jacc_test_epoch[1, :]
        jacc_test += [np.mean(jacc_perclass_test)]

        np.savez(os.path.join(savepath , "fcn1D_test_errors_best.npz"),
             err_test=err_test, acc_test=acc_test, jacc_test=jacc_test)
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
