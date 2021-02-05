import numpy as np
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DropoutLayer, ReshapeLayer, \
        NonlinearityLayer, DimshuffleLayer, ConcatLayer
from lasagne.layers import batch_norm, BatchNormLayer
from lasagne.layers import Pool1DLayer as PoolLayer
from lasagne.layers import Conv1DLayer as ConvLayer
from lasagne.layers import Upscale1DLayer as UpscaleLayer
from lasagne.layers import PadLayer, DenseLayer
from lasagne.layers import ElemwiseSumLayer, ElemwiseMergeLayer
from lasagne.nonlinearities import softmax, linear, rectify


def conv_bn_relu(net, incoming_layer, depth, num_filters, filter_size,
                    pad = 'same'):
    net['conv'+str(depth)] = ConvLayer(net[incoming_layer],
                num_filters = num_filters,
                filter_size = filter_size,
                pad = pad,
                nonlinearity=None)
    net['bn'+str(depth)] = BatchNormLayer(net['conv'+str(depth)])
    net['relu'+str(depth)] = NonlinearityLayer( net['bn'+str(depth)],
                                    nonlinearity = rectify)
    incoming_layer = 'relu'+str(depth)

    return incoming_layer


def bn_relu_conv(net, incoming_layer, depth, num_filters, filter_size,
                    pad = 'same'):

    net['bn'+str(depth)] = BatchNormLayer(net[incoming_layer])
    net['relu'+str(depth)] = NonlinearityLayer( net['bn'+str(depth)],
                                    nonlinearity = rectify)
    net['conv'+str(depth)] = ConvLayer(net['relu'+str(depth)],
                num_filters = num_filters,
                filter_size = filter_size,
                pad = pad,
                nonlinearity=None)
    incoming_layer = 'conv'+str(depth)

    return incoming_layer


def build_offset_model(input_var,
        filter_size=[25],
        n_filters = 64,
        n_classes = 7,
        poolings = [0,2,0,2,0],
        hidden_size = 512,
        n_hidden = 2,
        last_filter_size = 1,
        nb_in_channels = 1,
        block = 'bn_relu_conv',
        out_nonlin = softmax,
        dropout_p = 0.5):
    '''
    Parameters:
    -----------
    input_var : theano tensor
    filter_size : list of odd int (to fit with same padding),
                size of filter_size list determines the number of
                convLayer to Concatenate
    n_filters : int, number of filters for each convLayer
    n_classes : int
    poolings : list of int, represents the structure of the network, each
            number corresponds to a layer of the model, indicating the pooling
            stride. depth = len(poolings)
    hidden_size: number of units in fully connected layers
    n_hidden: number of fully connected layers
    last_filter_size : int, must be set to 1 (the older version had
            a last_filter_size of 3, that was an error
            the argument is there to be able to reassign weights correctly
            when testing)
    out_nonlin : default=softmax, non linearity function
    dropout_p : probability of setting values to zero in the hidden layers
    '''

    net = {}

    net['input'] = InputLayer((None, nb_in_channels, 200), input_var)
    incoming_layer = 'input'

    depth = len(poolings)

    for d in range(depth):
        if block == 'bn_relu_conv':
            incoming_layer = bn_relu_conv(net, incoming_layer, depth = d,
                            num_filters= n_filters, filter_size=filter_size[0])
        elif block == 'conv_bn_relu':
            incoming_layer = conv_bn_relu(net, incoming_layer, depth = d,
                            num_filters= n_filters, filter_size=filter_size[0])

        if poolings[d]:
            net['pool'+str(depth)] = PoolLayer(net[incoming_layer], poolings[d])
            incoming_layer = 'pool'+str(depth)

            # increase the number of filters by the same ratio in which we decrease
            # spatial resolution
            n_filters *= poolings[d]

    for i in range(n_hidden):
        name = 'dense_hidden'+str(i)
        net[name] = DenseLayer(net[incoming_layer], num_units=hidden_size)
        incoming_layer = name
        if dropout_p:
            name = 'dense_dropout'+str(i)
            net[name] = DropoutLayer(net[incoming_layer], p=dropout_p)
            incoming_layer = name

    #Output layer
    # it has N output layers, one for each transition
    outputs = []
    for i in range(n_classes):
        net['dense_out_'+str(i)] = DenseLayer(net[incoming_layer], num_units=1)
        outputs.append(net['dense_out_'+str(i)])

    return outputs, net
