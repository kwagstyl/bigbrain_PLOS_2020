import numpy as np
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DropoutLayer, ReshapeLayer, \
        NonlinearityLayer, DimshuffleLayer, ConcatLayer
from lasagne.layers import batch_norm, BatchNormLayer
from lasagne.layers import Pool1DLayer as PoolLayer
from lasagne.layers import Conv1DLayer as ConvLayer
from lasagne.layers import DenseLayer
from lasagne.layers import Upscale1DLayer as UpscaleLayer
from lasagne.layers import PadLayer
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


class dense_model():
    def __init__(self):
        pass

    def build_model(self,
        input_var,
        nb_in_channels = 2,
        n_classes = 8):

        net = {}

        net['input'] = InputLayer((None, nb_in_channels, 200), input_var)
        incoming_layer = 'input'

        net['conv'] = ConvLayer(net[incoming_layer], num_filters = 1, filter_size=3, pad = 'same')
        incoming_layer = 'conv'


        net['dense'] = DenseLayer(net[incoming_layer], num_units =100)
        incoming_layer = 'dense'

        net['last_layer'] = DenseLayer(net[incoming_layer], num_units = n_classes,
                            nonlinearity = softmax)

        self.net = net['last_layer']
        self.dict_net = net
        return [net[l] for l in ['last_layer']], net


class classif_model():


    def __init__(self):
        pass


    def build_model(
            self,
            input_var,
            n_classes = 8,
            filter_size=[25],
            #conv_before_pool = [2,2],
            n_filters = 64,
            depth = 1,
            nb_in_channels = 2,
            pool_factor = 2,
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
        n_classes : int
        depth : int, number of stacked convolution before concatenation
        last_filter_size : int, must be set to 1 (the older version had
                a last_filter_size of 3, that was an error
                the argument is there to be able to reassign weights correctly
                when testing)
        out_nonlin : default=softmax, non linearity function
        '''

        if depth>7:
            print "DEPTH MUST BE <=7, will change to 7"
            depth = 7
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

            net['pool'+str(d)] = PoolLayer(net[incoming_layer], pool_size = pool_factor)
            incoming_layer = 'pool'+str(d)
        #Output layer
        net['final_conv'] = DenseLayer(net[incoming_layer],
                        num_units = n_classes,
                        nonlinearity = out_nonlin)
        incoming_layer = 'final_conv'

        self.net = net['final_conv']
        self.dict_net = net
        return [net[l] for l in ['final_conv']], net


if __name__ == '__main__':
    print 'hey'
    model = classif_model()

    stuff = model.build_model(input_var = T.tensor3('input var'), depth = 7)

    for layer in model.dict_net:
        print layer, model.dict_net[layer], model.dict_net[layer].output_shape
