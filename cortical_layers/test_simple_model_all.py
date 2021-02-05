
# coding: utf-8

# In[23]:


#!/usr/bin/env python2
#Inspire du fichier train_fcn8.py

import os
import argparse
import time
from getpass import getuser
from distutils.dir_util import copy_tree

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

from fcn_1D_general import buildFCN_1D
from metrics import jaccard, accuracy, crossentropy
from cortical_layers import CorticalLayersDataset
from simple_model_1path import build_simple_model



# In[24]:


input_var = T.tensor3('input')


# In[25]:


exp_name = 'simple_model_lrate=0.0005_fil=64_fsizes=[49]_depth=6_data=both_decay=0.001_smooth=0_pat=50'
#exp_name = 'simple_model_lrate=0.0005_fil=64_fsizes=[25]_depth=8_data=both_decay=0.001_pat=50'
#exp_name = 'simple_model_lrate=0.0005_fil=64_fsizes=[25]_depth=8_data=both_decay=0.001_smooth=0.0_pat=50_kfold=10_val=0_test=1_batch_size=1000'
exp_name = '/data1/users/kwagstyl/bigbrain/datasets/6cortical_layers_all/simple_model_lrate=0.0005_fil=64_fsizes=[49]_depth=6_data=both_decay=0.001_smooth=0.0_pat=50_kfold=10_val=0_test=1_batch_size=1000'

#Model hyperparameters
n_filters = 64
filter_size = [49]  
depth  = 6
data_augmentation={} #{'horizontal_flip': True, 'fill_mode':'constant'}
block = 'bn_relu_conv'

#Training loop hyperparameters
weight_decay=0.001
num_epochs=500
max_patience=50
resume=False
learning_rate_value = 0.0005
#learning rate is defined below as a theano variable.
#Hyperparameters for the dataset loader
batch_size=[256,256,1]
smooth_or_raw = 'both'
shuffle_at_each_epoch = True 
minibatches_subset = 5
n_layers=6


# In[26]:


net = build_simple_model(input_var, filter_size = filter_size)


# In[27]:


SAVEPATH = '/data1/users/kwagstyl/bigbrain/datasets/'
LOADPATH = '/data1/users/kwagstyl/bigbrain/datasets/'
WEIGHTS_PATH = LOADPATH


# In[28]:


build_model_name = False

if build_model_name :
    exp_name = 'simple_model'
    exp_name += '_lrate=' + str(learning_rate_value)
    exp_name += '_fil=' + str(n_filters)
    exp_name += '_fsizes=' + str(filter_size)
    #uncomment this line if new version of train_function
    exp_name += '_depth=' + str(depth)
    exp_name += '_' + smooth_or_raw
    exp_name += '_decay=' + str(weight_decay)
    exp_name += '_pat=' + str(max_patience)
    exp_name += ('_noshuffle'+str(minibatches_subset)+'batch') if not shuffle_at_each_epoch else ''

    print exp_name
    
    


# In[29]:


dataset = '6cortical_layers_all'
weight_path = os.path.join(WEIGHTS_PATH, dataset, exp_name, 'new_fcn1D_model_best.npz')


# In[30]:


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
    n_layers=n_layers)

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

test_iter = None


n_batches_train = train_iter.nbatches
n_batches_val = val_iter.nbatches
n_batches_test = test_iter.nbatches if test_iter is not None else 0
n_classes = train_iter.non_void_nclasses
void_labels = train_iter.void_labels



# In[31]:


#
# Define symbolic variables
#
input_var = T.tensor3('input_var') #n_example*nb_in_channels*ray_size
target_var = T.ivector('target_var') #n_example*ray_size

learn_step=  theano.shared(np.array(learning_rate_value, dtype=theano.config.floatX))


# In[32]:


#
# Build model and assign trained weights
#
simple_net_output, net = build_simple_model(input_var,
                    filter_size = filter_size,
                    n_filters = n_filters,
                    depth = depth,
                    block= block,
                    nb_in_channels = nb_in_channels,
                    n_classes = n_classes)
#must be set to 1 for the new models
                    #in the last version, last_filter_size was uncorrectly set to 3
                    #so, in order to recover/resassign weights correctly, must be a
                    #"new" parameter

print 'Done building model'

with np.load(weight_path) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    
nlayers=len(lasagne.layers.get_all_params(simple_net_output))
lasagne.layers.set_all_param_values(simple_net_output, param_values[:nlayers])

print 'Done assigning weights'


# In[33]:


print "Defining and compiling test functions"
test_prediction = lasagne.layers.get_output(simple_net_output[0],
                                            deterministic=True)
test_loss = categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()
test_acc, test_acc_per_sample =accuracy(test_prediction, target_var, void_labels)
test_jacc = jaccard(test_prediction, target_var, n_classes)

test_fn = theano.function([input_var, target_var], [test_loss, test_acc,
                                                   test_jacc, test_acc_per_sample])
print "Done"


# In[34]:


#Function computing the prediction with current parameters (for visualization)
pred = theano.function([input_var], lasagne.layers.get_output(net['probs_reshape'],
                                            deterministic=True))


# In[35]:


#To visualize the ray 
def make_2Darray(arr, height = 25):
    arr = np.reshape(arr, (1,arr.shape[0]))
    x = np.repeat(arr, height, 0)
    return x


# In[36]:


def profile2indices(profile):
    if 8 in profile or len(set(profile))<5:
        #If nonsense profile, return all indices as zeros.
        Indices=[0,0,0,0,0,0,0]
        return Indices
    else:
        try :
            if profile.index(0) < 100:
                profile[0:profile.index(0)]=[0]*profile.index(0)
                if 8 in profile or len(set(profile))<5:
                    Indices=[0,0,0,0,0,0,0]
                    return Indices
        except ValueError:
            pass
        try :
        #Get index of first layer 1
            Indices=[profile.index(1)]
            #print profile.index(1)
        except ValueError:
        #If no layer 1s, get first nonzero,
        #  sometimes layer 1 is ripped off but we still want locations of other layers
            Indices=[next((i for i, x in enumerate(profile) if x), None)]
        #then set all before that to 1, to get rid of some nonsense
        profile[0:Indices[0]]=[1]*Indices[0]
        try :
            Indices.append(profile.index(2))
        except ValueError:
            try :
                Indices.append(len(profile)-profile[::-1].index(1)-1)
            except ValueError:
                return [0,0,0,0,0,0,0]
        #If no layer 2
        #then set all before that to 2,
        profile[0:Indices[1]]=[2]*Indices[1]
        try :
            Indices.append(profile.index(3))
        except ValueError:
        #If no layer 3
            try :
                Indices.append(profile.index(4))
            except ValueError:
                #if no layer 3 or 4, nonsense
                try :
                    Indices.append(profile.index(5))
                except ValueError:
#                    print("error b")
                    return [0,0,0,0,0,0,0]
        profile[0:Indices[2]]=[3]*Indices[2]
        try :
            Indices.append(profile.index(4))
        except ValueError:
            Indices.append(len(profile)-profile[::-1].index(3)-1)
        profile[0:Indices[3]]=[4]*Indices[3]
        try :
            Indices.append(profile.index(5))
        except ValueError:
            Indices.append(len(profile)-profile[::-1].index(4)-1)
        profile[0:Indices[4]]=[5]*Indices[4]
        try :
            Indices.append(profile.index(6))
        except ValueError:
            Indices.append(len(profile)-profile[::-1].index(5)-1)
        #get last occurance of index 4.
        try:
            Indices.append(len(profile)-profile[::-1].index(6)-1)
        except ValueError:
            Indices.append(len(profile)-profile[::-1].index(5)-1)
    return Indices;


# In[45]:

hemis=['left','right']
for hemi in hemis:
    Raw = '/data1/users/kwagstyl/bigbrain/TestData/raw_'+hemi+'_20.txt'
    Geo = '/data1/users/kwagstyl/bigbrain/TestData/geo_'+hemi+'_20.txt'
    print('loading geo profiles')
    geo=np.loadtxt(Geo)
    print('loading raw profiles')
    raw=np.loadtxt(Raw)
    print('stacking profiles')
    Data=np.hstack([geo,raw])
    print('reshaping profiles')
    Data=np.reshape(Data,[len(geo),2,200],order='A').astype('float32')
    Dividers=np.round(np.linspace(0,len(geo),30)).astype(int)
    Indices=np.zeros((len(geo),7))
    k=-1
    for D in range(len(Dividers)-1):
        print str(100* k / float(len(geo))) + '% done'
        print('predicting profiles')
        predicted_labels=np.argmax(pred(Data[Dividers[D]:Dividers[D+1]]), axis=2)
        print('done predicting, now profile 2 indices')
        for p in predicted_labels:
            k+=1
            Is=profile2indices(p.tolist())
            Indices[k,:]=Is
    np.savetxt(hemi+'_indices.txt',Indices,fmt='%i')

