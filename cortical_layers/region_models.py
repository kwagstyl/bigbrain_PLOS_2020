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
                    pad = 'same', name=''):
    net[name+'conv'+str(depth)] = ConvLayer(net[incoming_layer],
                num_filters = num_filters,
                filter_size = filter_size,
                pad = pad,
                nonlinearity=None)
    net[name+'bn'+str(depth)] = BatchNormLayer(net['conv'+str(depth)])
    net[name+'relu'+str(depth)] = NonlinearityLayer( net['bn'+str(depth)],
                                    nonlinearity = rectify)
    incoming_layer = name+'relu'+str(depth)

    return incoming_layer


def bn_relu_conv(net, incoming_layer, depth, num_filters, filter_size,
                    pad = 'same', name=''):

    net[name+'bn'+str(depth)] = BatchNormLayer(net[incoming_layer])
    net[name+'relu'+str(depth)] = NonlinearityLayer( net['bn'+str(depth)],
                                    nonlinearity = rectify)
    net[name+'conv'+str(depth)] = ConvLayer(net['relu'+str(depth)],
                num_filters = num_filters,
                filter_size = filter_size,
                pad = pad,
                nonlinearity=None)
    incoming_layer = name+'conv'+str(depth)

    return incoming_layer

def build_region_model(input_var,
        filter_size=[25],
        n_filters = 64,
        n_classes = 41,
        poolings = [0,2,0,2,0],
        hidden_size = 512,
        n_hidden = 2,
        last_filter_size = 1,
        nb_in_channels = 1,
        block = 'bn_relu_conv',
        out_nonlin = softmax):
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

    for i in range(n_hidden):
        name = 'dense_hidden'+str(i)
        net[name] = DenseLayer(net[incoming_layer], num_units=hidden_size)
        incoming_layer = name

    #Output layer
    net['dense_out'] = DenseLayer(net[incoming_layer], num_units=n_classes)
    incoming_layer = 'dense_out'

    net['last_layer'] = NonlinearityLayer(net[incoming_layer],
                    nonlinearity = out_nonlin)
    incoming_layer = 'last_layer'

    return [net[l] for l in ['last_layer']], net

def build_simple_join_model(input_var,
        filter_size=[25],
        n_filters = 64,
        n_filters_reg = 64,
        n_classes = 6,
        n_regions = 6,
        depth = 1,
        region_depth = 3,
        poolings = [0,2,0,2,0],
        hidden_size = 512,
        n_hidden = 2,
        last_filter_size = 1,
        nb_in_channels = 1,
        block = 'bn_relu_conv',
        #bn_relu_conv = False, #unused for now
        out_nonlin = softmax):
    '''
    Parameters:
    -----------
    input_var : theano tensor
    filter_size : list of odd int (to fit with same padding),
                size of filter_size list determines the number of
                convLayer to Concatenate
    n_filters : int, number of filters for each convLayer
    n_filters_reg : int, number of filters for each convLayer in the region
                classification network
    n_classes : int
    depth : int, number of stacked convolution before concatenation
    region_depth: int, index of layer that the region classification network
                will take as input.
    poolings : list of int, represents the structure of the region
            classification network, each number corresponds to a layer of the
            model, indicating the pooling stride. depth = len(poolings)
    hidden_size: number of units in fully connected layers of the class model
    n_hidden: number of fully connected layers in the class model
    last_filter_size : int, must be set to 1 (the older version had
            a last_filter_size of 3, that was an error
            the argument is there to be able to reassign weights correctly
            when testing)
    out_nonlin : default=softmax, non linearity function
    '''


    net = {}

    net['input'] = InputLayer((None, nb_in_channels, 200), input_var)
    incoming_layer = 'input'
    #incoming_layer = 'input'

    #Convolution layers
    for d in range(depth):
        if block == 'bn_relu_conv':
            incoming_layer = bn_relu_conv(net, incoming_layer, depth = d,
                            num_filters= n_filters, filter_size=filter_size[0])

        elif block == 'conv_bn_relu':
            incoming_layer = conv_bn_relu(net, incoming_layer, depth = d,
                            num_filters= n_filters, filter_size=filter_size[0])

        if d == region_depth:
            # this is the layer that the region classification network will
            # take as input
            region_incoming_layer = incoming_layer

    #Output layer
    net['final_conv'] = ConvLayer(net[incoming_layer],
                    num_filters = n_classes,
                    filter_size = last_filter_size,
                    pad='same')
    incoming_layer = 'final_conv'

    #DimshuffleLayer and ReshapeLayer to fit the softmax implementation
    #(it needs a 1D or 2D tensor, not a 3D tensor)
    net['final_dimshuffle'] = DimshuffleLayer(net[incoming_layer], (0,2,1))
    incoming_layer = 'final_dimshuffle'

    layerSize = lasagne.layers.get_output(net[incoming_layer]).shape
    net['final_reshape'] = ReshapeLayer(net[incoming_layer],
                                (T.prod(layerSize[0:2]),layerSize[2]))
                                # (200*batch_size,n_classes))
    incoming_layer = 'final_reshape'


    #This is the layer that computes the prediction
    net['last_layer'] = NonlinearityLayer(net[incoming_layer],
                    nonlinearity = out_nonlin)
    incoming_layer = 'last_layer'

    #Layers needed to visualize the prediction of the network
    net['probs_reshape'] = ReshapeLayer(net[incoming_layer],
                    (layerSize[0], layerSize[1], n_classes))
    incoming_layer = 'probs_reshape'

    net['probs_dimshuffle'] = DimshuffleLayer(net[incoming_layer], (0,2,1))

    # Region classifier net starting from a given layer in the segmentation
    # network stored in region_incoming_layer
    for d in range(len(poolings)):
        if block == 'bn_relu_conv':
            region_incoming_layer = bn_relu_conv(net, region_incoming_layer, depth = d,
                            num_filters= n_filters_reg, filter_size=filter_size[0], name='reg_')
        elif block == 'conv_bn_relu':
            region_incoming_layer = conv_bn_relu(net, region_incoming_layer, depth = d,
                            num_filters= n_filters_reg, filter_size=filter_size[0], name='reg_')

        if poolings[d]:
            net['pool'+str(depth)] = PoolLayer(net[region_incoming_layer], poolings[d])
            region_incoming_layer = 'pool'+str(depth)

    for i in range(n_hidden):
        name = 'dense_hidden'+str(i)
        net[name] = DenseLayer(net[region_incoming_layer], num_units=hidden_size)
        region_incoming_layer = name

    #Output layer
    net['dense_out'] = DenseLayer(net[region_incoming_layer], num_units=n_regions)
    region_incoming_layer = 'dense_out'

    net['last_layer_reg'] = NonlinearityLayer(net[region_incoming_layer],
                    nonlinearity = out_nonlin)
    region_incoming_layer = 'last_layer_reg'

    # [net[l] for l in ['last_layer']] : used to directly compute the output
    #                       of the network
    # net : dictionary containing each layer {name : Layer instance}
    return [net[l] for l in ['last_layer', 'last_layer_reg']], net
