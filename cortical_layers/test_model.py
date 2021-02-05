import numpy as np
import os
import time
from distutils.dir_util import copy_tree
import theano
import theano.tensor as T
from theano import config
import lasagne
from data_loader.cortical_layers import CorticalLayersDataset
from simple_model_1path import build_simple_model
from metrics import jaccard, accuracy, crossentropy, weighted_crossentropy

_FLOATX = config.floatX

SAVEPATH = '/Tmp/cucurulg/cortical_layers'
LOADPATH = '/data/lisatmp4/cucurulg/cortical_layers'
WEIGHTS_PATH = LOADPATH

savepath=SAVEPATH
loadpath=LOADPATH

# experiment folder name
exp_name = "simple_model_lrate=0.0005_fil=64_fsizes=[25]_depth=8_data=both_decay=0.001_pat=50"
dataset = "6cortical_layers_all"

savepath = os.path.join(savepath, dataset, exp_name)
loadpath = os.path.join(loadpath, dataset, exp_name)

# copy files to /Tmp
if savepath != loadpath:
    print('Copying model and other training files to {}'.format(savepath))
    copy_tree(loadpath, savepath)



str_arguments = {
    'block' : None,
    'smooth_or_raw' : None
}
int_arguments = {
    'n_filters' : None,
    'depth'  : None,
    'n_layers' : None
}
list_arguments = {
    'filter_size' : None,
    'batch_size': None
}

with open(os.path.join(savepath, "config.txt")) as f:
    for line in f:
        if " = " not in line:
            continue
        key, value = line.rstrip().split(" = ")
        if key in str_arguments:
            str_arguments[key] = value
        if key in int_arguments:
            int_arguments[key] = int(value)
        if key in list_arguments:
            list_arguments[key] = eval(value)

args = {}
args.update(int_arguments)
args.update(str_arguments)
args.update(list_arguments)

input_var = T.tensor3('input_var') #n_example*nb_in_channels*ray_size
target_var = T.ivector('target_var') #n_example*ray_size
weight_vector = T.fvector('weight_vector')

if args['smooth_or_raw'] =='both':
    nb_in_channels = 2
    use_threads = False
else:
    nb_in_channels = 1
    use_threads = True

val_iter = CorticalLayersDataset(
    which_set='valid',
    smooth_or_raw = args['smooth_or_raw'],
    batch_size=args['batch_size'][1],
    shuffle_at_each_epoch = True,
    return_one_hot=False,
    return_01c=False,
    return_list=False,
    use_threads=use_threads,
    preload=True,
    n_layers=args['n_layers'])

n_batches_val = val_iter.nbatches
n_classes = val_iter.non_void_nclasses
void_labels = val_iter.void_labels

simple_net_output, net = build_simple_model(input_var,
                    filter_size = args['filter_size'],
                    n_filters = args['n_filters'],
                    depth = args['depth'],
                    block= args['block'],
                    nb_in_channels = nb_in_channels,
                    n_classes = n_classes)

Class=np.loadtxt('/Tmp/cucurulg/datasets/cortical_layers/6layers_segmentation/training_cls.txt')
def compute_class_weights(Class):
    #get unique labels and number of pixels of each class
    unique, counts = np.unique(Class,return_counts=True)
    #calculate freq(c) number of pixels per class divided by the total number of pixels in images where c is present
    freq=counts.astype(float)/Class.size
    return np.median(freq)/freq

weights=compute_class_weights(Class)

print "Defining and compiling valid functions"

valid_prediction = lasagne.layers.get_output(simple_net_output[0], deterministic=True)

with np.load(savepath+'/new_fcn1D_model_best.npz') as f:
     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(simple_net_output, param_values)

valid_loss = weighted_crossentropy(valid_prediction, target_var, weight_vector)
valid_loss = valid_loss.mean()

valid_acc, valid_sample_acc = accuracy(valid_prediction, target_var, void_labels)
valid_jacc = jaccard(valid_prediction, target_var, n_classes)

valid_fn = theano.function([input_var, target_var, weight_vector],
                           [valid_loss, valid_acc, valid_sample_acc, valid_jacc, valid_prediction])#,profile=True)
print "Done"

err_valid = []
acc_valid = []
jacc_valid = []
sample_acc_valid_tot = []

treshold = 0.7 # for extracting the very incorrect labelled samples
ratios=[0.80,0.85, 0.90] #ratios for the per sample accuracy

# Validation
cost_val_epoch = 0
acc_val_epoch = 0
sample_acc_valid_epoch = np.array([0.0 for i in range(len(ratios))])
jacc_val_epoch = np.zeros((2, n_classes))

start_time = time.time()
for i in range(n_batches_val):

    # Get minibatch (comment the next line if only 1 minibatch in training)
    val_batch = val_iter.next()
    X_val_batch, L_val_batch = val_batch['data'], val_batch['labels']
    L_val_batch = np.reshape(L_val_batch, np.prod(L_val_batch.shape))

    # Validation step
    cost_val_batch, acc_val_batch, sample_acc_valid_batch, jacc_val_batch, predicted = valid_fn(X_val_batch, L_val_batch, weights[L_val_batch].astype('float32'))

    # shape of predictions is (batch_size*200,7)
    predicted = predicted.reshape((-1,200,7))
    predicted = predicted.argmax(axis=2) # convert prediction to segmentation
    print(predicted[0])
    print(L_val_batch.reshape((-1, 200))[0])

    quit()

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

out_str = "Cost val %f, acc val %f, jacc val %f took %f s"
out_str = out_str % (err_valid[0],
                     acc_valid[0],
                     jacc_valid[0],
                     time.time()-start_time)
out_str2 = 'Per sample accuracy (ratios ' + str(ratios) + ') '
out_str2 += ' valid ' + str(sample_acc_valid_tot[0])
print out_str
print out_str2
