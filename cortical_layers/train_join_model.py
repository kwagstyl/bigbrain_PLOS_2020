#!/usr/bin/env python2

import os
import time
import numpy as np
import theano.tensor as T
from theano import shared, config
import theano
import lasagne
from lasagne.regularization import regularize_network_params
from distutils.dir_util import copy_tree
from data_loader.cortical_layers_w_regions import CorticalLayersRegionsDataset
from simple_model_1path import build_simple_model
from region_models import build_region_model, build_simple_join_model
from metrics import weighted_crossentropy, accuracy_regions, accuracy, jaccard

_FLOATX = config.floatX

# IMPORTANT PATHS
SAVEPATH = '/Tmp/cucurulg/cortical_layers'
LOADPATH = '/data/lisatmp4/cucurulg/cortical_layers'
WEIGHTS_PATH = LOADPATH

#Model hyperparameters
n_filters = 64
filter_size = [25]#[7,15,25,49]
depth  = 8
data_augmentation={} #{'horizontal_flip': True, 'fill_mode':'constant'}
block = 'bn_relu_conv'
# region network
n_filters_reg = 32
poolings = [0,2,0,2,0]
n_hidden = 2
hidden_size = 512
region_depth = 3

#Training loop hyperparameters
weight_decay=0.001
num_epochs=100
max_patience=50
resume=False
#learning_rate_value = 0.0005 #learning rate is defined below as a theano variable.
learning_rate_value = 0.00025 #learning rate is defined below as a theano variable.

#Hyperparameters for the dataset loader
batch_size=[1000,1000,1]
smooth_or_raw = 'both'
shuffle_at_each_epoch = True
minibatches_subset = 0
n_layers = 6

# load/save directories
savepath=SAVEPATH
loadpath=LOADPATH

# experiment name
exp_name = 'joined_model'
exp_name += '_lrate=' + str(learning_rate_value)
exp_name += '_fil=' + str(n_filters)
exp_name += '_fsizes=' + str(filter_size)
exp_name += '_fregsizes=' + str(n_filters_reg)
exp_name += '_depth=' + str(depth)
exp_name += '_classnetdepth=' + str(len(poolings))
exp_name += '_regdepth=' + str(region_depth)
exp_name += '_nhidden=' + str(n_hidden)
exp_name += '_hsize=' + str(hidden_size)
exp_name += '_data=' + smooth_or_raw
exp_name += '_decay=' + str(weight_decay)
exp_name += '_pat=' + str(max_patience)
exp_name += ('_noshuffle'+str(minibatches_subset)+'batch') if not shuffle_at_each_epoch else ''

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

# Input symbolic variables
input_var = T.tensor3('input_var') #n_example*nb_in_channels*ray_size
target_var_reg = T.ivector('target_var_reg') #n_example
target_var_seg = T.ivector('target_var_seg') #n_example*ray_size
weight_vector_reg = T.fvector('weight_vector_reg')
weight_vector_seg = T.fvector('weight_vector_seg')

learn_step = shared(np.array(learning_rate_value,
                            dtype=_FLOATX))

# dataset iterator
nb_in_channels = 2 if smooth_or_raw == 'both' else 1
use_threads = False if smooth_or_raw == 'both' else True

train_iter = CorticalLayersRegionsDataset(
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
    n_layers=n_layers)

val_iter = CorticalLayersRegionsDataset(
    which_set='valid',
    smooth_or_raw = smooth_or_raw,
    batch_size=batch_size[1],
    shuffle_at_each_epoch = shuffle_at_each_epoch,
    return_one_hot=False,
    return_01c=False,
    return_list=False,
    use_threads=use_threads,
    preload=True,
    n_layers=n_layers)

n_batches_train = train_iter.nbatches
n_batches_val = val_iter.nbatches
n_classes = train_iter.non_void_nclasses
n_regions = train_iter.n_regions
void_labels = train_iter.void_labels

# Define model
simple_net_output, net = build_simple_join_model(input_var,
                    filter_size = filter_size,
                    n_filters = n_filters,
                    n_filters_reg = n_filters_reg,
                    depth = depth,
                    region_depth = region_depth,
                    poolings = poolings,
                    n_hidden = n_hidden,
                    hidden_size = hidden_size,
                    block= block,
                    nb_in_channels = nb_in_channels,
                    n_classes = n_classes, n_regions=n_regions)

# load weights to balance training error
def compute_class_weights(Class):
    #get unique labels and number of pixels of each class
    unique, counts = np.unique(Class,return_counts=True)
    #calculate freq(c) number of pixels per class divided by the total number of pixels in images where c is present
    freq=counts.astype(float)/Class.size
    return np.median(freq)/freq

# weights for the region labels
Class=np.loadtxt('/Tmp/cucurulg/datasets/cortical_layers/6layers_segmentation/training_regions.txt')
weights_reg=compute_class_weights(Class)

# weights for the segmentation labels
Class=np.loadtxt('/Tmp/cucurulg/datasets/cortical_layers/6layers_segmentation/training_cls.txt')
weights_seg=compute_class_weights(Class)

# Define training functions

print "Defining and compiling training functions"

prediction_seg, prediction_reg = lasagne.layers.get_output([
                                                simple_net_output[0],
                                                simple_net_output[1]])

# Loss function
loss_reg = weighted_crossentropy(prediction_reg, target_var_reg, weight_vector_reg)
loss_reg = loss_reg.mean()

loss_seg = weighted_crossentropy(prediction_seg, target_var_seg, weight_vector_seg)
loss_seg = loss_seg.mean()

loss = loss_reg+loss_seg

# Add regularization
if weight_decay > 0:
    weightsl2 = regularize_network_params(
        simple_net_output, lasagne.regularization.l2)
    loss += weight_decay * weightsl2

# train accuracy
train_acc_reg = accuracy_regions(prediction_reg, target_var_reg)
train_acc_seg, train_sample_acc_seg = accuracy(prediction_seg, target_var_seg,
                                                void_labels)


# Define the update function
params = lasagne.layers.get_all_params(simple_net_output, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=learn_step)

# training function
train_fn = theano.function([input_var, target_var_reg, target_var_seg,
                            weight_vector_reg, weight_vector_seg],
                           [loss, train_acc_reg, train_acc_seg, train_sample_acc_seg],
                           updates=updates)
print "Done"

print "Defining and compiling valid functions"
valid_prediction_seg, valid_prediction_reg = lasagne.layers.get_output(
                                            [simple_net_output[0],
                                            simple_net_output[1]],
                                            deterministic=True)
valid_loss_reg = weighted_crossentropy(valid_prediction_reg, target_var_reg,
                                        weight_vector_reg)
valid_loss_reg = valid_loss_reg.mean()

# segmentation loss
valid_loss_seg = weighted_crossentropy(valid_prediction_seg, target_var_seg,
                                        weight_vector_seg)
valid_loss_seg = valid_loss_seg.mean()

valid_loss = valid_loss_seg+valid_loss_reg

valid_acc_reg = accuracy_regions(valid_prediction_reg, target_var_reg)
valid_acc_seg, valid_sample_acc_seg = accuracy(valid_prediction_seg, target_var_seg,
                                            void_labels)
valid_jacc = jaccard(valid_prediction_seg, target_var_seg, n_classes)

valid_fn = theano.function([input_var, target_var_reg, target_var_seg,
                            weight_vector_reg, weight_vector_seg],
                [valid_loss, valid_acc_reg, valid_acc_seg, valid_sample_acc_seg, valid_jacc])
print "Done"

# Training loop parameters
plot_results_train = False #from the training set
plot_results_valid = False #from the validation set

treshold = 0.7 # for extracting the very incorrect labelled samples
ratios=[0.80,0.85, 0.90] #ratios for the per sample accuracy

err_train = []
acc_train_reg = []
acc_train_seg = []
sample_acc_train_tot = []
worse_indices_train = []
already_seen_idx = []

err_valid = []
acc_valid_reg = []
acc_valid_seg = []
jacc_valid = []
sample_acc_valid_tot = []
patience = 0
worse_indices_valid =[]

print "Start training"
for epoch in range(num_epochs):
    start_time = time.time()
    #Cost train and acc train for this epoch
    cost_train_epoch = 0
    acc_train_epoch_reg = 0
    acc_train_epoch_seg = 0
    sample_acc_train_epoch = np.array([0.0 for i in range(len(ratios))])

    for i in range(n_batches_train):
        # Get minibatch (comment the next line if only 1 minibatch in training)
        train_batch = train_iter.next()
        X_train_batch, L_train_batch, SEG_train_batch = train_batch['data'], train_batch['regions'], train_batch['labels']
        L_train_batch = np.reshape(L_train_batch, np.prod(L_train_batch.shape))
        SEG_train_batch = np.reshape(SEG_train_batch, np.prod(SEG_train_batch.shape))

        # deb = debug_prediction(X_train_batch)
        # print(deb.shape)
        # print(X_train_batch.shape)
        # print(L_train_batch.shape)
        # quit()
        # One training step

        cost_train_batch, acc_train_batch_reg, acc_train_batch_seg, sample_acc_train_batch = train_fn(
                                X_train_batch,
                                L_train_batch,
                                SEG_train_batch,
                                weights_reg[L_train_batch].astype('float32'),
                                weights_seg[SEG_train_batch].astype('float32')
                            )

        sample_acc_train_batch_mean = [np.mean([(i>=ratio)
                        for i in sample_acc_train_batch]) for ratio in ratios]

        #Update epoch results
        cost_train_epoch += cost_train_batch
        acc_train_epoch_reg += acc_train_batch_reg
        acc_train_epoch_seg += acc_train_batch_seg
        sample_acc_train_epoch += sample_acc_train_batch_mean

    #Add epoch results
    err_train += [cost_train_epoch/n_batches_train]
    acc_train_reg += [acc_train_epoch_reg/n_batches_train]
    acc_train_seg += [acc_train_epoch_seg/n_batches_train]
    sample_acc_train_tot += [sample_acc_train_epoch/n_batches_train]

    # Validation
    cost_val_epoch = 0
    acc_val_epoch_reg = 0
    acc_val_epoch_seg = 0
    sample_acc_valid_epoch = np.array([0.0 for i in range(len(ratios))])
    jacc_val_epoch = np.zeros((2, n_classes))

    for i in range(n_batches_val):
        # Get minibatch (comment the next line if only 1 minibatch in training)
        val_batch = val_iter.next()
        X_val_batch, L_val_batch, SEG_val_batch = val_batch['data'], val_batch['regions'], val_batch['labels']
        L_val_batch = np.reshape(L_val_batch, np.prod(L_val_batch.shape))
        SEG_val_batch = np.reshape(SEG_val_batch, np.prod(SEG_val_batch.shape))

        # Validation step
        cost_val_batch, acc_val_batch_reg, acc_val_batch_seg, sample_acc_valid_batch, jacc_val_batch = valid_fn(
                                X_val_batch,
                                L_val_batch,
                                SEG_val_batch,
                                weights_reg[L_val_batch].astype('float32'),
                                weights_seg[SEG_val_batch].astype('float32')
                            )

        sample_acc_valid_batch_mean = [np.mean([(i>=ratio)
                        for i in sample_acc_valid_batch]) for ratio in ratios]

        #Update epoch results
        cost_val_epoch += cost_val_batch
        acc_val_epoch_reg += acc_val_batch_reg
        acc_val_epoch_seg += acc_val_batch_seg
        sample_acc_valid_epoch += sample_acc_valid_batch_mean
        jacc_val_epoch += jacc_val_batch

    #Add epoch results
    err_valid += [cost_val_epoch/n_batches_val]
    acc_valid_reg += [acc_val_epoch_reg/n_batches_val]
    acc_valid_seg += [acc_val_epoch_seg/n_batches_val]
    sample_acc_valid_tot += [sample_acc_valid_epoch/n_batches_val]
    jacc_perclass_valid = jacc_val_epoch[0, :] / jacc_val_epoch[1, :]
    jacc_valid += [np.mean(jacc_perclass_valid)]

    out_str = "EPOCH %i: Avg cost train %f, acc reg train %f, acc segmentation train %f, cost val %f, acc reg val %f, acc segmentation val %f, val jaccard % f took %f s"
    out_str = out_str % (epoch, err_train[epoch],
                         acc_train_reg[epoch],
                         acc_train_seg[epoch],
                         err_valid[epoch],
                         acc_valid_reg[epoch],
                         acc_valid_seg[epoch],
                         jacc_valid[epoch],
                         time.time()-start_time)

    out_str2 = 'Per sample accuracy (ratios ' + str(ratios) + ') '
    out_str2 += ' train ' +str(sample_acc_train_tot[epoch])
    out_str2 += ' valid ' + str(sample_acc_valid_tot[epoch])
    print out_str
    print out_str2

    with open(os.path.join(savepath, "fcn1D_output.log"), "a") as f:
        f.write(out_str + "\n")

    if epoch == 0:
        best_acc_val = jacc_valid[epoch]
    elif epoch > 1 and jacc_valid[epoch] > best_acc_val:
        print('saving best (and last) model')
        best_acc_val = jacc_valid[epoch]
        patience = 0
        np.savez(os.path.join(savepath, 'new_fcn1D_model_best.npz'),
                 *lasagne.layers.get_all_param_values(simple_net_output))
        np.savez(os.path.join(savepath , "fcn1D_errors_best.npz"),
                 err_train=err_train, acc_train=acc_train_seg,
                 err_valid=err_valid, acc_valid=acc_valid_seg, jacc_valid=jacc_valid)
        np.savez(os.path.join(savepath, 'new_fcn1D_model_last.npz'),
                 *lasagne.layers.get_all_param_values(simple_net_output))
        np.savez(os.path.join(savepath , "fcn1D_errors_last.npz"),
                 err_train=err_train, acc_train=acc_train_seg,
                 err_valid=err_valid, acc_valid=acc_valid_seg, jacc_valid=jacc_valid)
    else:
        patience += 1
        print('saving last model')
        np.savez(os.path.join(savepath, 'new_fcn1D_model_last.npz'),
                 *lasagne.layers.get_all_param_values(simple_net_output))
        np.savez(os.path.join(savepath , "fcn1D_errors_last.npz"),
                 err_train=err_train, acc_train=acc_train_seg,
                 err_valid=err_valid, acc_valid=acc_valid_seg)
    # Finish training if patience has expired or max nber of epochs reached

    if patience == max_patience or epoch == num_epochs-1:
        if savepath != loadpath:
            print('Copying model and other training files to {}'.format(loadpath))
            copy_tree(savepath, loadpath)
        break
