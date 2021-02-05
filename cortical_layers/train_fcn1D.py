#!/usr/bin/env python2
#Inspire du fichier train_fcn8.py

import os
import argparse
import time
from getpass import getuser
from distutils.dir_util import copy_tree

import numpy as np
import theano
import theano.tensor as T
from theano import config
import lasagne
from lasagne.regularization import regularize_network_params
from lasagne.objectives import categorical_crossentropy

from fcn_1D_general import buildFCN_1D
from metrics import jaccard, accuracy, crossentropy
from data_loader.cortical_layers import CorticalLayersDataset

_FLOATX = config.floatX
if getuser() == 'larocste':
    SAVEPATH = '/Tmp/larocste/cortical_layers'
    LOADPATH = '/data/lisatmp4/larocste/cortical_layers'
    WEIGHTS_PATH = LOADPATH
else:
    raise ValueError('Unknown user : {}'.format(getuser()))



def train(
          learn_step=0.001,
          weight_decay=1e-4,
          num_epochs=500,
          max_patience=100,
          data_augmentation={},
          savepath=None, loadpath=None,
          batch_size=None,
          resume=False,
          conv_before_pool= [1],
          n_conv_bottom = 1,
          merge_type='sum',
          n_filters = 64,
          filter_size = 3,
          pool_size = 2,
          dropout = 0.5,
          shuffle_at_each_epoch = True,
          minibatches_subset = 0):

    #
    # Prepare load/save directories
    #
    exp_name = 'fcn1D'
    exp_name += '_dataugm' if bool(data_augmentation) else ''
    exp_name += '_' + str(conv_before_pool)
    exp_name += '_' + str(n_filters) +'filt'
    exp_name += '_' + 'batchs='+(str(batch_size) if batch_size is not None else 'none')
    exp_name += '_' + 'botconv=' + str(n_conv_bottom)
    exp_name += '_' + 'lrate=' + str(learn_step)
    exp_name += '_' + 'pat=' + str(max_patience)
    exp_name += '_' + 'merge=' + merge_type
    exp_name += '_' + 'fsize=' + str(filter_size)
    exp_name += '_' + 'psize=' + str(pool_size)
    exp_name += ('_noshuffle'+str(minibatches_subset)+'batch') if not shuffle_at_each_epoch else ''

    if savepath is None:
        raise ValueError('A saving directory must be specified')

    dataset = 'cortical_layers'
    savepath = os.path.join(savepath, dataset, exp_name)
    loadpath = os.path.join(loadpath, dataset, exp_name)
    print 'Savepath : ' + str(savepath)
    print 'Loadpath : ' + str(loadpath)

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
    input_var = T.tensor3('input_var') #n_example*nb_in_channels*ray_size
    target_var = T.ivector('target_var') #n_example*ray_size

    #
    # Build dataset iterator
    #

    train_iter = CorticalLayersDataset(
        which_set='train',
        batch_size=batch_size[0],
        data_augm_kwargs=data_augmentation,
        shuffle_at_each_epoch = shuffle_at_each_epoch,
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=True)

    val_iter = CorticalLayersDataset(
        which_set='valid',
        batch_size=batch_size[1],
        shuffle_at_each_epoch = shuffle_at_each_epoch,
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=True)

    #Test set not configured yet (additional data?)
    test_iter = None

    n_batches_train = train_iter.nbatches
    n_batches_val = val_iter.nbatches
    n_batches_test = test_iter.nbatches if test_iter is not None else 0
    n_classes = train_iter.non_void_nclasses
    void_labels = train_iter.void_labels
    nb_in_channels = train_iter.data_shape[0]

    print('------------------------------')
    print('Learning rate ' +str(learn_step))
    print('Max patience ' +str(max_patience))
    print('Batch size ' +str(batch_size))
    print('Shuffle ? ' + str(shuffle_at_each_epoch) + ', Minibatch subset? ' + str(minibatches_subset))

    print "Batch. train: %d, val %d, test %d" % (n_batches_train, n_batches_val,
                                                 n_batches_test)
    print "Nb of classes: %d" % (n_classes)
    print "Nb. of input channels: %d" % (nb_in_channels)

    #
    # Build network
    #
    convmodel = buildFCN_1D(input_var,
                        n_classes=n_classes,
                        n_in_channels = nb_in_channels,
                        conv_before_pool= conv_before_pool,
                        n_conv_bottom = n_conv_bottom,
                        merge_type= merge_type,
                        n_filters = n_filters,
                        filter_size = filter_size,
                        pool_size = pool_size,
                        dropout =dropout,
                        layer = ['probs'])
    # To print each layer, uncomment this:
    lays = lasagne.layers.get_all_layers(convmodel)
    for l in lays:
         print l, l.output_shape

    #import pdb; pdb.set_trace()

    #
    # Define and compile theano functions
    #
    print "Defining and compiling training functions"

    print convmodel[0]
    prediction = lasagne.layers.get_output(convmodel[0])
    loss = categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    #loss = crossentropy(prediction, target_var, void_labels)




    if weight_decay > 0:
        weightsl2 = regularize_network_params(
            convmodel, lasagne.regularization.l2)
        loss += weight_decay * weightsl2

    train_acc = accuracy(prediction, target_var, void_labels)

    params = lasagne.layers.get_all_params(convmodel, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=learn_step)

    train_fn = theano.function([input_var, target_var], [loss, train_acc],
                                                            updates=updates)

    print "Defining and compiling test functions"
    test_prediction = lasagne.layers.get_output(convmodel,
                                                deterministic=True)[0]
    test_loss = categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    #test_loss = crossentropy(test_prediction, target_var, void_labels)
    test_acc = accuracy(test_prediction, target_var, void_labels)
    test_jacc = jaccard(test_prediction, target_var, n_classes)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc,
                                                       test_jacc])

    #
    # Train
    #
    err_train = []
    acc_training = []
    err_valid = []
    acc_valid = []
    jacc_valid = []
    patience = 0


    print train_iter.next()
    import pdb; pdb.set_trace()


    # Training main loop
    print "Start training"
    for epoch in range(num_epochs):
        #print('epoch' + str(epoch))
        # Single epoch training and validation
        start_time = time.time()
        cost_train_tot = 0
        acc_train_tot = 0

        # Train
        if minibatches_subset > 0:
            n_batches_val = minibatches_subset
            n_batches_train = minibatches_subset
        for i in range(n_batches_train):
            # Get minibatch
            X_train_batch, L_train_batch = train_iter.next()
            L_train_batch = np.reshape(L_train_batch,
                                       np.prod(L_train_batch.shape))

            # Training step
            cost_train, acc_train = train_fn(X_train_batch, L_train_batch)

            cost_train_tot += cost_train
            acc_train_tot += acc_train

        err_train += [cost_train_tot/n_batches_train]
        acc_training += [acc_train_tot/n_batches_train]

        # Validation
        cost_val_tot = 0
        acc_val_tot = 0
        jacc_val_tot = np.zeros((2, n_classes))


        for i in range(n_batches_val):
            # Get minibatch
            X_val_batch, L_val_batch = val_iter.next()
            L_val_batch = np.reshape(L_val_batch,
                                     np.prod(L_val_batch.shape))

            # Validation step
            cost_val, acc_val, jacc_val = val_fn(X_val_batch, L_val_batch)


            cost_val_tot += cost_val
            acc_val_tot += acc_val
            jacc_val_tot += jacc_val

        err_valid += [cost_val_tot/n_batches_val]
        acc_valid += [acc_val_tot/n_batches_val]
        jacc_perclass_valid = jacc_val_tot[0, :] / jacc_val_tot[1, :]
        jacc_valid += [np.mean(jacc_perclass_valid)]


        out_str = "EPOCH %i: Avg cost train %f, acc train %f"+\
            ", cost val %f, acc val %f, jacc val %f took %f s"


        out_str = out_str % (epoch, err_train[epoch],
                             acc_training[epoch],
                             err_valid[epoch],
                             acc_valid[epoch],
                             jacc_valid[epoch],
                             time.time()-start_time)
        print out_str

        with open(os.path.join(savepath, "fcn1D_output.log"), "a") as f:
            f.write(out_str + "\n")

        # Early stopping and saving stuff
        if epoch == 0:
            best_jacc_val = jacc_valid[epoch]
        elif epoch > 1 and jacc_valid[epoch] > best_jacc_val:
            print('saving best model:')
            best_jacc_val = jacc_valid[epoch]
            patience = 0
            np.savez(os.path.join(savepath, 'new_fcn1D_model_best.npz'),
                     *lasagne.layers.get_all_param_values(convmodel))
            np.savez(os.path.join(savepath , "fcn1D_errors_best.npz"),
                     err_train=err_train, acc_train=acc_training,
                     err_valid=err_valid,
                     acc_valid=acc_valid, jacc_valid=jacc_valid)
        else:
            patience += 1
            print('saving last model')
            np.savez(os.path.join(savepath, 'new_fcn1D_model_last.npz'),
                     *lasagne.layers.get_all_param_values(convmodel))
            np.savez(os.path.join(savepath , "fcn1D_errors_last.npz"),
                     err_train=err_train, acc_train=acc_training,
                     err_valid=err_valid,
                     acc_valid=acc_valid, jacc_valid=jacc_valid)
        # Finish training if patience has expired or max nber of epochs
        # reached
        if patience == max_patience or epoch == num_epochs-1:
            if test_iter is not None:
                # Load best model weights
                with np.load(os.path.join(savepath, 'new_fcn1D_model_best.npz')) as f:
                    param_values = [f['arr_%d' % i]
                                    for i in range(len(f.files))]
                nlayers = len(lasagne.layers.get_all_params(convmodel))
                lasagne.layers.set_all_param_values(convmodel,
                                                    param_values[:nlayers])
                # Test
                cost_test_tot = 0
                acc_test_tot = 0
                jacc_num_test_tot = np.zeros((1, n_classes))
                jacc_denom_test_tot = np.zeros((1, n_classes))
                for i in range(n_batches_test):
                    # Get minibatch
                    X_test_batch, L_test_batch = test_iter.next()
                    L_test_batch = np.reshape(L_test_batch,
                                              np.prod(L_test_batch.shape))

                    # Test step
                    cost_test, acc_test, jacc_test = \
                        val_fn(X_test_batch, L_test_batch)
                    jacc_num_test, jacc_denom_test = jacc_test

                    acc_test_tot += acc_test
                    cost_test_tot += cost_test
                    jacc_num_test_tot += jacc_num_test
                    jacc_denom_test_tot += jacc_denom_test

                err_test = cost_test_tot/n_batches_test
                acc_test = acc_test_tot/n_batches_test
                jacc_test = np.mean(jacc_num_test_tot / jacc_denom_test_tot)

                out_str = "FINAL MODEL: err test % f, acc test %f, jacc test %f"
                out_str = out_str % (err_test,
                                     acc_test,
                                     jacc_test)
                print out_str
            if savepath != loadpath:
                print('Copying model and other training files to {}'.format(loadpath))
                copy_tree(savepath, loadpath)

            # End
            return

def main():
    # parser = argparse.ArgumentParser(description='FCN 1D model training')
    #
    # parser.add_argument('-learning_rate',
    #                     default=0.0001,
    #                     help='Learning Rate')
    # parser.add_argument('-weight_decay',
    #                     default=0.0,
    #                     help='regularization constant')
    # parser.add_argument('--num_epochs',
    #                     '-ne',
    #                     type=int,
    #                     default=750,
    #                     help='Optional. Int to indicate the max'
    #                     'number of epochs.')
    # parser.add_argument('-max_patience',
    #                     type=int,
    #                     default=25,
    #                     help='Max patience')
    # parser.add_argument('-data_augmentation',
    #                    type=dict,
    #                    default={'horizontal_flip': True, 'fill_mode':'constant'},
    #                    help='use data augmentation')
    # parser.add_argument('-n_conv_bottom',
    #                     type = int,
    #                     default = 2,
    #                     help ='Number of convolution bottom path')
    # parser.add_argument('-merge_type',
    #                     default = 'sum',
    #                     help ='Merge type ("sum" or "concat")')
    # parser.add_argument('-n_filters',
    #                     type = int,
    #                     default = 64,
    #                     help ='Number of filters for first convolution')
    # parser.add_argument('-dropout',
    #                     default = 0.5,
    #                     help = 'Dropout probability')
    #args = parser.parse_args()


    train(learn_step = 0.0001,#float(args.learning_rate),
          weight_decay=0.0,#float(args.weight_decay),
          num_epochs = 750,#int(args.num_epochs),
          max_patience = 25,#int(args.max_patience),
          data_augmentation={'horizontal_flip': True, 'fill_mode':'constant'},
          batch_size=[1,1,1],
          savepath=SAVEPATH,
          loadpath=LOADPATH,
          conv_before_pool= [3,3,3],
          n_conv_bottom = 2,#args.n_conv_bottom,
          merge_type='sum',#args.merge_type,
          n_filters = 64,#args.n_filters,
          filter_size = 3,
          pool_size = 2,
          dropout = 0.5, #float(args.dropout),
          shuffle_at_each_epoch = True,
          minibatches_subset = 0)




if __name__ == "__main__":
    main()
