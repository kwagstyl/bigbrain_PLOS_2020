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
from data_loader.cortical_layers import CorticalLayersDataset
# from data_loader.cortical_layers_w_regions import CorticalLayersRegionsDataset
from simple_model_1path import build_simple_model
from metrics import weighted_crossentropy, accuracy, jaccard, dice_loss

def segmentation_to_indexes(y_true):
    '''Converts segmentation labels to indexes labels, each index indicating
    a change of layer'''

    out = np.zeros(y_true.shape, dtype=np.uint8)
    for i in range(out.shape[0]):
            for j in range(out.shape[1]-1):
                    out[i,j] = 1 if y_true[i,j] != y_true[i,j+1] else 0

    return out

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

#Training loop hyperparameters
weight_decay=0.001
penalty_transitions=0.00005 # penalizes detecting too many or too few transitions
num_epochs=500
max_patience=50
resume=False
learning_rate_value = 0.0005 #learning rate is defined below as a theano variable.

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
exp_name = 'indexes_model'
exp_name += '_lrate=' + str(learning_rate_value)
exp_name += '_fil=' + str(n_filters)
exp_name += '_fsizes=' + str(filter_size)
exp_name += '_depth=' + str(depth)
exp_name += '_data=' + smooth_or_raw
exp_name += '_decay=' + str(weight_decay)
exp_name += '_penalty=' + str(penalty_transitions)
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
target_var = T.ivector('target_var') #n_example*ray_size
weight_vector = T.fvector('weight_vector')

learn_step = shared(np.array(learning_rate_value,
                            dtype=_FLOATX))

# dataset iterator
nb_in_channels = 2 if smooth_or_raw == 'both' else 1
use_threads = False if smooth_or_raw == 'both' else True

# ORIG: CorticalLayersDataset
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
    n_layers=n_layers)

# ORIG: CorticalLayersDataset
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
    n_layers=n_layers)

n_batches_train = train_iter.nbatches
n_batches_val = val_iter.nbatches
n_classes = 2
void_labels = train_iter.void_labels

# Define model
simple_net_output, net = build_simple_model(input_var,
                    filter_size = filter_size,
                    n_filters = n_filters,
                    depth = depth,
                    block= block,
                    nb_in_channels = nb_in_channels,
                    n_classes = n_classes)

# load weights to balance training error
Class=np.loadtxt('/Tmp/cucurulg/datasets/cortical_layers/6layers_segmentation/training_cls.txt')
def compute_class_weights(Class):
    #get unique labels and number of pixels of each class
    unique, counts = np.unique(Class,return_counts=True)
    #calculate freq(c) number of pixels per class divided by the total number of pixels in images where c is present
    freq=counts.astype(float)/Class.size
    return np.median(freq)/freq

weights=compute_class_weights(Class)

# Define training functions

print "Defining and compiling training functions"

prediction = lasagne.layers.get_output(simple_net_output[0])

# deb_pred = theano.function([input_var], prediction)

# Loss function
#loss = weighted_crossentropy(prediction, target_var, weight_vector)
loss = dice_loss(prediction, target_var)
loss = loss.mean()

# Add regularization
if weight_decay > 0:
    weightsl2 = regularize_network_params(
        simple_net_output, lasagne.regularization.l2)
    loss += weight_decay * weightsl2

# Add penalty to enforce the same number of transitions:
if penalty_transitions > 0:
    true_prediction = T.reshape(target_var, (-1, 200))
    prediction_reshape = T.reshape(prediction, (-1, 200, 2))
    penalty_loss = abs(prediction_reshape[:,:,1] - true_prediction).sum(axis=1)
    loss += penalty_transitions * penalty_loss.mean()

# train accuracy
train_acc, train_sample_acc = accuracy(prediction, target_var, void_labels)

# Define the update function
params = lasagne.layers.get_all_params(simple_net_output, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=learn_step)

# training function
train_fn = theano.function([input_var, target_var],
                           [loss, train_acc, train_sample_acc, prediction],
                           updates=updates)
print "Done"

print "Defining and compiling valid functions"
valid_prediction = lasagne.layers.get_output(simple_net_output[0],
                                            deterministic=True)
valid_loss = dice_loss(valid_prediction, target_var)
valid_loss = valid_loss.mean()

valid_acc, valid_sample_acc = accuracy(valid_prediction, target_var, void_labels)
valid_jacc = jaccard(valid_prediction, target_var, n_classes)

valid_fn = theano.function([input_var, target_var],
                [valid_loss, valid_acc, valid_sample_acc, valid_jacc, valid_prediction])
print "Done"

# Training loop parameters
plot_results_train = False #from the training set
plot_results_valid = False #from the validation set

treshold = 0.7 # for extracting the very incorrect labelled samples
ratios=[0.80,0.85, 0.90] #ratios for the per sample accuracy

err_train = []
acc_train = []
acc_seg_train = []
sample_acc_train_tot = []
worse_indices_train = []
already_seen_idx = []

err_valid = []
acc_valid = []
jacc_valid = []
sample_acc_valid_tot = []
patience = 0
worse_indices_valid =[]

def compute_batch_accuracy(y_pred, y_true, batch_size):
    y_pred = y_pred.reshape((batch_size, 200))
    segmentation = np.zeros((batch_size, 200), dtype=np.int32)

    for i in range(y_pred.shape[0]):
        n_transitions = y_pred[i].sum()
        # start with 0 if there are only 5 transitions
        current = 0 if n_transitions != 5 else 1
        for j in range(y_pred.shape[1]):
            if y_pred[i][j] == 1:
                current += 1
                # skip layer 4 if there are only 6 transitions
                if current == 4 and n_transitions == 6:
                    current += 1

            segmentation[i][j] = current

    y_true = y_true.reshape((-1))
    segmentation = segmentation.reshape((-1))

    acc = segmentation == y_true

    return acc.mean()


print "Start training"
for epoch in range(num_epochs):
    start_time = time.time()
    #Cost train and acc train for this epoch
    cost_train_epoch = 0
    acc_train_epoch = 0
    acc_segmentation_epoch = 0
    sample_acc_train_epoch = np.array([0.0 for i in range(len(ratios))])

    for i in range(n_batches_train):
        # Get minibatch (comment the next line if only 1 minibatch in training)
        train_batch = train_iter.next()
        X_train_batch, L_train_batch = train_batch['data'], train_batch['labels']

        # Convert layer segmentation labels to a index labeling, where a 1
        # indicates a change of layer. Very sparse
        layer_change = segmentation_to_indexes(L_train_batch)

        L_train_batch = np.reshape(layer_change, np.prod(layer_change.shape))

        # One training step
        cost_train_batch, acc_train_batch, sample_acc_train_batch, out_predictions = train_fn(
            X_train_batch, L_train_batch)

        predicted_transitions = np.argmax(out_predictions, axis=1)
        # print(predicted_transitions.sum(), layer_change.sum())
        # print(predicted_transitions.reshape(X_train_batch.shape[0], 200)[0])

        acc_segmentation = compute_batch_accuracy(predicted_transitions, train_batch['labels'], X_train_batch.shape[0])

        sample_acc_train_batch_mean = [np.mean([(i>=ratio)
                        for i in sample_acc_train_batch]) for ratio in ratios]

        #Update epoch results
        cost_train_epoch += cost_train_batch
        acc_train_epoch += acc_train_batch
        acc_segmentation_epoch += acc_segmentation
        sample_acc_train_epoch += sample_acc_train_batch_mean

    #Add epoch results
    err_train += [cost_train_epoch/n_batches_train]
    acc_train += [acc_train_epoch/n_batches_train]
    acc_seg_train += [acc_segmentation_epoch/n_batches_train]
    sample_acc_train_tot += [sample_acc_train_epoch/n_batches_train]

    # Validation
    cost_val_epoch = 0
    acc_val_epoch = 0
    sample_acc_valid_epoch = np.array([0.0 for i in range(len(ratios))])
    jacc_val_epoch = np.zeros((2, n_classes))

    for i in range(n_batches_val):
        # Get minibatch (comment the next line if only 1 minibatch in training)
        val_batch = val_iter.next()
        X_val_batch, L_val_batch = val_batch['data'], val_batch['labels']

        # Convert layer segmentation labels to a index labeling, where a 1
        # indicates a change of layer. Very sparse
        layer_change = segmentation_to_indexes(L_val_batch)
        L_val_batch = np.reshape(layer_change, np.prod(layer_change.shape))

        # Validation step
        cost_val_batch, acc_val_batch, sample_acc_valid_batch, jacc_val_batch, out_predictions = valid_fn(
            X_val_batch, L_val_batch)

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

    out_str = "EPOCH %i: Avg cost train %f, acc train %f, acc seg train %f,"+        ", cost val %f, acc val %f, jacc val %f took %f s"
    out_str = out_str % (epoch, err_train[epoch],
                         acc_train[epoch],
                         acc_seg_train[epoch],
                         err_valid[epoch],
                         acc_valid[epoch],
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
