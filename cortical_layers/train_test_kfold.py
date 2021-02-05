# coding: utf-8

# In[1]:


import os

from profile_functions import profile2indices_post_process, expand_to_fill, indices2surfaces, get_neighbours, confidence, import_surface

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

from matplotlib import pyplot as plt
from metrics import jaccard, accuracy, crossentropy, weighted_crossentropy
from data_loader.cortical_layers_w_regions_kfold_val_train_test import CorticalLayersDataset

from simple_model_1path import build_simple_model
from profile_functions import layer_error


# In[2]:





def train_test_kfold(resolution,kfold,val_fold,test_fold, max_patience=50):
    _FLOATX = config.floatX
    SAVEPATH = '/data1/users/kwagstyl/bigbrain/cortical_layers'
    LOADPATH = '/data1/users/kwagstyl/bigbrain/cortical_layers'
    WEIGHTS_PATH = LOADPATH
    print('starting '+str(resolution))
    n_filters=64
    filter_size = [49]
    depth = 6
    data_augmentation={}
    block = 'bn_relu_conv'
    weight_decay=0.001
    num_epochs = 500
    max_patience =max_patience
    resume=False
    learning_rate_value = 0.0005
    #max_clipping=30
    prefix='training_'+str(resolution)+'_'
    batch_size=[1000,1000,1]
    smooth_or_raw = 'both'
    shuffle_at_each_epoch = True
    minibatches_subset=0
    n_layers=6
    
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
    exp_name += '_pat=' + str(max_patience)
    exp_name += '_kfold=' + str(kfold)
    exp_name += '_val=' + str(val_fold)
    exp_name += '_test=' + str(test_fold)
    exp_name += '_batch_size=' + str(batch_size[0])
    exp_name += ('_noshuffle'+str(minibatches_subset)+'batch') if not shuffle_at_each_epoch else ''




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



    # In[6]:


    input_var = T.tensor3('input_var') #n_example*nb_in_channels*ray_size
    target_var = T.ivector('target_var') #n_example*ray_size
    weight_vector = T.fvector('weight_vector')

    learn_step=  theano.shared(np.array(learning_rate_value, dtype=theano.config.floatX))




    # In[7]:


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



    # In[8]:


    #
    simple_net_output, net = build_simple_model(input_var,
                        filter_size = filter_size,
                        n_filters = n_filters,
                        depth = depth,
                        block= block,
                        nb_in_channels = nb_in_channels,
                        n_classes = n_classes)

    #simple_net_output = last layer of the simple_model net


    # In[9]:


    Class=np.loadtxt(os.path.join(LOADPATH,'6layers_segmentation_August2018/training_cls.txt'))
    def compute_class_weights(Class):
        #get unique labels and number of pixels of each class
        unique, counts = np.unique(Class,return_counts=True)
        #calculate freq(c) number of pixels per class divided by the total number of pixels in images where c is present
        freq=counts.astype(float)/Class.size
        return np.median(freq)/freq

    weights=compute_class_weights(Class)


    # In[10]:


    def clipping(x_train, l_train, max_clipping = 10):
        """ Clip profiles at either end by random amounts. Augments data"""
        k=-1
        for p, l in zip(x_train, l_train):
            k+=1
            new_p=p.copy()
            cut_top=np.random.randint(max_clipping)
            cut_bottom=np.random.randint(max_clipping)
            if cut_top >0:
                p=np.delete(p,range(cut_top),axis=1)
                l=np.delete(l,range(cut_top))
            if cut_bottom >0:
                l=np.delete(l,p.shape[1]-np.arange(cut_bottom))
                p=np.delete(p,p.shape[1]-np.arange(cut_bottom),axis=1)
                #interpolate back to full length.
            x = np.linspace(0, len(p), x_train.shape[2])
            new_p[0,:]=np.interp(x,range(p.shape[1]),p[0,:])
            new_p[1,:]=np.interp(x,range(p.shape[1]),p[1,:])
            new_label=np.round(np.interp(x,range(p.shape[1]),l))
            x_train[k]=new_p
            l_train[k]=new_label
        return x_train, l_train

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


    # In[12]:


    print "Defining and compiling training functions"

    convolved = smooth_convolution(simple_net_output[0], n_classes)

    prediction, convolved = lasagne.layers.get_output([simple_net_output[0], convolved])

    #prediction = lasagne.layers.get_output([simple_net_output[0]])
    #loss = categorical_crossentropy(prediction, target_var)
    #loss = loss.mean()

    loss = weighted_crossentropy(prediction, target_var, weight_vector)
    loss = loss.mean()


    if weight_decay > 0:
        weightsl2 = regularize_network_params(
            simple_net_output, lasagne.regularization.l2)
        loss += weight_decay * weightsl2

    #if smooth_penalty > 0:
    #    smooth_cost = T.sum(abs(convolved), axis=(1,2))
    #   loss += smooth_penalty * smooth_cost.mean()

    train_acc, train_sample_acc = accuracy(prediction, target_var, void_labels)

    params = lasagne.layers.get_all_params(simple_net_output, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=learn_step)

    train_fn = theano.function([input_var, target_var, weight_vector], [loss, train_acc, train_sample_acc, prediction],
                               updates=updates)#, profile=True)

    print "Done"
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
    test_acc, test_sample_acc = accuracy(test_prediction, target_var, void_labels)
    test_jacc = jaccard(test_prediction, target_var, n_classes)
    
    test_fn = theano.function([input_var, target_var, weight_vector], [test_loss, test_acc,
                                                    test_sample_acc,test_jacc, T.argmax(test_prediction, axis=1)])
    print "Done"
    predict = theano.function([input_var], lasagne.layers.get_output(net['probs_reshape'],
                                            deterministic=True))



    # In[23]:





    # In[27]:


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


    # In[ ]:


    import time


    # In[101]:


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
           # print(X_train_batch[0])
    #        if max_clipping > 0:
    #            X_train_batch, L_train_batch = clipping(X_train_batch, L_train_batch, max_clipping)
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

            if test_fold is not None:
                cost_test_epoch=0
                acc_test_epoch=0
                sample_acc_test_epoch=np.array([0.0 for i in range(len(ratios))])
                jacc_test_epoch=np.zeros((2,n_classes))
                     #test the best iteration and save results for comparisons
                layer_predictions=[]
                 #test the best iteration and save results for comparisons
                for i in range(n_batches_test):
                    test_batch = test_iter.next()
                    X_test_batch, L_test_batch = test_batch['data'], test_batch['labels']
                    L_store=L_test_batch.copy()
                    L_test_batch = np.reshape(L_test_batch, np.prod(L_test_batch.shape))

            # Testing step
                    cost_test_batch, acc_test_batch, sample_acc_test_batch, jacc_test_batch, test_pred = test_fn(
                                        X_test_batch, L_test_batch, weights[L_test_batch].astype('float32'))
                    test_pred = np.argmax(predict(X_test_batch),axis=2)
                    for l,p in zip(L_store,test_pred):
                    
                        layer_predictions.append(layer_error(p,l))

                    sample_acc_test_batch_mean = [np.mean([(i>=ratio)
                                        for i in sample_acc_test_batch]) for ratio in ratios]



                    cost_test_epoch += cost_test_batch
                    acc_test_epoch += acc_test_batch
                    sample_acc_test_epoch += sample_acc_test_batch_mean
                    jacc_test_epoch += jacc_test_batch
            
                layer_predictions=np.array(layer_predictions)
                layer_std=np.std(layer_predictions,axis=0)
                layer_mean=np.mean(layer_predictions,axis=0)
                err_test = [cost_test_epoch/n_batches_test]
                acc_test = [acc_test_epoch/n_batches_test]
                sample_acc_test_tot = [sample_acc_test_epoch/n_batches_test]
                jacc_perclass_test = jacc_test_epoch[0, :] / jacc_test_epoch[1, :]
                jacc_test = [np.mean(jacc_perclass_test)]

                np.savez(os.path.join(savepath , "fcn1D_test_errors_best.npz"),
                 err_test=err_test, acc_test=acc_test, jacc_test=jacc_test, layer_std=layer_std, layer_mean=layer_mean)

            
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


def test_full_model(resolution, kfold, val_fold,test_fold, max_patience=50):
    from scipy import stats
    print('testing model on full dataset')
    _FLOATX = config.floatX
    SAVEPATH = '/data1/users/kwagstyl/bigbrain/cortical_layers'
    LOADPATH = '/data1/users/kwagstyl/bigbrain/cortical_layers'
    test_dir = '/data1/users/kwagstyl/bigbrain/TestData'
    WEIGHTS_PATH = LOADPATH
    print('testing '+str(resolution))
    n_filters=64
    filter_size = [49]
    depth = 6
    data_augmentation={}
    block = 'bn_relu_conv'
    weight_decay=0.001
    num_epochs = 500
    max_patience =max_patience
    resume=False
    learning_rate_value = 0.0005
    #max_clipping=30
    prefix='training_'+str(resolution)+'_'
    batch_size=[1000,1000,1]
    smooth_or_raw = 'both'
    shuffle_at_each_epoch = True
    minibatches_subset=0
    n_layers=6

    

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
    exp_name += '_pat=' + str(max_patience)
    exp_name += '_kfold=' + str(kfold)
    exp_name += '_val=' + str(val_fold)
    exp_name += '_test=' + str(test_fold)
    exp_name += '_batch_size=' + str(batch_size[0])
    exp_name += ('_noshuffle'+str(minibatches_subset)+'batch') if not shuffle_at_each_epoch else ''



    if smooth_or_raw =='both':
        nb_in_channels = 2
        use_threads = False
    else:
        nb_in_channels = 1
        use_threads = True

#    train_iter = CorticalLayersDataset(
 #           which_set='train',
  #          prefix=prefix,
   #         smooth_or_raw = smooth_or_raw,
    #        batch_size=batch_size[0],
     #       return_one_hot=False,
      #      return_01c=False,
       #     return_list=False,
        #    use_threads=use_threads,
         #   preload=False,
          #  n_layers=n_layers,
           # kfold=kfold, # if None, kfold = number of regions (so there is one fold per region)
           # val_fold=val_fold, # it will use the first fold for validation
           # test_fold=test_fold) # t

    n_classes = 8 #train_iter.non_void_nclasses
    #void_labels = train_iter.void_labels
    


    dataset = str(n_layers)+'cortical_layers'
    weight_path = os.path.join(WEIGHTS_PATH, dataset, exp_name, 'new_fcn1D_model_best.npz')
    savepath = os.path.join(savepath, dataset, exp_name)
    loadpath = os.path.join(loadpath, dataset, exp_name)
    input_var = T.tensor3('input_var') #n_example*nb_in_channels*ray_size
    target_var = T.ivector('target_var') #n_example*ray_size

    simple_net_output, net = build_simple_model(input_var,
                    filter_size = filter_size,
                    n_filters = n_filters,
                    depth = depth,
                    block= block,
                    nb_in_channels = nb_in_channels,
                    n_classes = n_classes)


    print 'Done building model'
    with np.load(weight_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    
    nlayers=len(lasagne.layers.get_all_params(simple_net_output))
    lasagne.layers.set_all_param_values(simple_net_output, param_values[:nlayers])

    print 'Done assigning weights'

    

    pred = theano.function([input_var], lasagne.layers.get_output(net['probs_reshape'],
                                            deterministic=True))
    hemis=['right','left']
    for hemi in hemis:
            Raw = os.path.join(test_dir,'raw_'+hemi+'_'+str(resolution)+'.txt')
            Geo = os.path.join(test_dir,'geo_'+hemi+'_'+str(resolution)+'.txt')
            print('loading geo profiles')
            geo=np.genfromtxt(Geo)
            print('loading raw profiles')
            raw=np.genfromtxt(Raw)
            #normalising
            print('normalising')
#            g_means=np.expand_dims(np.mean(geo,axis=1),1)
 #           g_std = np.expand_dims(np.std(geo, axis=1),1)
#            _,white_coords,_,_=import_surface(os.path.join(test_dir,'white_'+hemi+'_up.obj')) 
            geodesic_coords = np.loadtxt(os.path.join(test_dir,'visual_geodesic_distances_{}.txt'.format(hemi)))
            #slope, intercept, r_value, p_value, std_err = stats.linregress(white_coords[:,1], g_means)
#            print(slope)
            #geo = (geo-g_means)/g_std 
            #raw = (raw-g_means)/g_std 
            Data=np.hstack([geo,raw])
            Data=np.nan_to_num(Data)
            print('reshaping profiles')
            Data=np.reshape(Data,[len(geo),2,200],order='A').astype('float32')
            for k,row in enumerate(Data):
                #fixed slope at -11.34
                Data[k]=Data[k]-geodesic_coords[k]*-11.34
#                 Data[k]=Data[k]-white_coords[:,1]*-
            #deal with pale occipital cortex
            zs=(np.mean(Data[:,0],axis=1)-np.mean(np.mean(Data[geodesic_coords==0,0],axis=1)))/np.std(np.mean(Data[geodesic_coords==0,0],axis=1))
            Data[np.logical_and(geodesic_coords==0,zs<-1)]=Data[np.logical_and(geodesic_coords==0,zs<-1)]+4000
            #there were problems with the pred function only returning homgeneous results
            print('setting empties')
            Dividers=np.round(np.linspace(0,len(geo),50)).astype(int)
            Indices=np.zeros((len(geo),n_classes-1)).astype(int)
            confidences=np.zeros((len(geo),n_classes))
            k=-1
            for D in range(len(Dividers)-1):
                if k % 1000 ==0:
                    print(str(100*k/len(geo))+'% done')
             #   print('predicting profiles')
                predictions=pred(Data[Dividers[D]:Dividers[D+1]])
                predicted_labels=np.argmax(predictions, axis=2)
              #  print('done predicting, now profile 2 indices')
                for b, p in enumerate(predicted_labels):
                    k+=1
                    Indices[k,:]=profile2indices_post_process(p.tolist())
                    confidences[k,:]=confidence(predictions[b],Indices[k,:])
            np.savetxt(os.path.join(test_dir, 'confidence'+hemi+'_'+str(resolution)+'.txt'),confidences,fmt='%.2f')
            #interpolate zeros across the surface
            neighbours=get_neighbours(os.path.join(test_dir,'white_'+hemi+'_up.obj'))
            filled = expand_to_fill(Indices,neighbours)
            np.savetxt(os.path.join(test_dir,'indices'+hemi+'_'+str(resolution)+'_nonzeros.txt'),filled,fmt='%i')


