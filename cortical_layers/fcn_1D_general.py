import numpy as np
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DropoutLayer, ReshapeLayer, \
        NonlinearityLayer, DimshuffleLayer, ConcatLayer
from lasagne.layers import Pool1DLayer as PoolLayer
from lasagne.layers import Conv1DLayer as ConvLayer
from lasagne.layers import Upscale1DLayer as UpscaleLayer
from lasagne.layers import PadLayer
from lasagne.layers import ElemwiseSumLayer, ElemwiseMergeLayer
from lasagne.nonlinearities import softmax, linear


def buildFCN_1D(input_var,n_classes , n_in_channels = 1,
                        conv_before_pool= [1], n_conv_bottom = 1, merge_type='sum',
                        n_filters = 64, filter_size = 3, pool_size = 2,
                        out_nonlin = softmax, dropout=0.5,
                        layer=['probs_dimshuffle']):

    '''
    Build 1D conv net models

    Parameters
    ------------
    input_var : theano tensor, input of the network
    n_classes : int, number of classes (6 in cortical_layers)
    n_in_channels : int, nb of channels in input_var (1 grayscale, 3 RGB)
    conv_before_pool : list of int, number of convolutions before each
        pooling layer
    n_conv_bottom : int, nb of convolution between last pooling and
        first upsampling (bottleneck part)
    merge_type : string, 'sum' or 'concat', type of long skip connections between
        contracting path and expansive path
    n_filters: int, number of filters of each convolution (increases every
        time resolution is downsampled)
    filter_size : int, size of filter used in convolution
    pool_size : int, size/factor of pooling/upsampling
    out_nonlin : output non linearity
    dropout: float, dropout probability
    layer : string, last layer of the output (use 'probs' to fit with softmax
        implementation)

    '''

    print('------------------------------------------------')
    print('Hyperparameters : ')
    print('Number of classes : ' + str(n_classes))
    print('Number of input channels : ' + str(n_in_channels))
    print('Conv before each pooling : ' + str(conv_before_pool))
    print('Number of bottom convolution : ' + str(n_conv_bottom))
    print('Merge type : ' + merge_type)
    print('Number of filters (initially) : ' + str(n_filters))
    print('Filter size : ' + str(filter_size))
    print('Pool size : ' + str(pool_size))
    print('Dropout probability : ' +str(dropout))
    print('------------------------------------------------')

    net = {}

    if len(conv_before_pool) == 0:
        raise ValueError('No conv/pooling layers!')

    #Building contracting path
    net, incoming_layer = down_path(net, input_var,
                        n_in_channels,conv_before_pool,
                        n_filters, filter_size, pool_size, dropout)

    #Building bottleneck path
    net, incoming_layer = bottom_path(net, incoming_layer,
                        conv_before_pool, n_conv_bottom,
                        n_filters, filter_size)

    #Building expanding path
    net, incoming_layer = up_path(net, incoming_layer,
                        conv_before_pool, n_conv_bottom, merge_type,
                        n_filters, filter_size, pool_size)

    #Building softmax/output path
    net= output_path(net, incoming_layer, n_classes,
                        filter_size, out_nonlin)
    print('Done building conv 1D net')

    return  [net[el] for el in layer],net

def down_path(net, input_var, n_in_channels,conv_before_pool,
                        n_filters, filter_size, pool_size, dropout):
    '''
    Build fcn 1D contracting path. The contracting path is built by combining
    convolution and pooling layers, at least until the last concatenation is
    reached.
    Output : net, last_layer of this net

    Parameters
    ----------
    Same as above
    '''
    n_pool = len(conv_before_pool)

    net['input'] = InputLayer((None, n_in_channels, 200), input_var)
    incoming_layer = 'input'

    for p in range(1, n_pool+1):
        nb_conv = conv_before_pool[p-1]
        for c in range(1,nb_conv+1):

            #Add conv layer
            net['conv'+str(p)+'_'+str(c)] = ConvLayer(
                    net[incoming_layer], n_filters*(2**(p-1)), filter_size,
                    pad='same')
            incoming_layer = 'conv'+str(p)+'_'+str(c) #update future incoming layer's name

        #For now, dropout only applied before the last pooling (just before
        #bottleneck path)
        if p==n_pool and dropout>0:
            net['dropout'+str(p)] = DropoutLayer(net[incoming_layer], p=dropout)
            incoming_layer = 'dropout'+str(p)

        #Add pooling layer
        net['pool' + str(p)] = PoolLayer(net[incoming_layer], pool_size)
        incoming_layer = 'pool' + str(p)

    return net, incoming_layer

def bottom_path(net, incoming_layer,
                conv_before_pool, n_conv_bottom,
                n_filters, filter_size):
    '''
    Build the conv layers between last pooling and first upsampling
    (bottleneck path)

    Parameters
    ----------
    Same as above
    incoming_layer : string, name of last layer from contractive layers
    '''

    n_pool = len(conv_before_pool)

    for c in range(1, n_conv_bottom+1):
        net['conv'+str(n_pool+1)+'_'+str(c)] = ConvLayer(
            net[incoming_layer], n_filters*(2**n_pool), filter_size,
            pad='same')
        incoming_layer = 'conv'+str(n_pool+1)+'_'+str(c)

    return net, incoming_layer

def up_path(net, incoming_layer,
                conv_before_pool, n_conv_bottom, merge_type,
                n_filters, filter_size, pool_size):
    '''
    Build the expansive path (combination of upsampling+conv+merge+conv)

    Parameters
    ----------
    Same as above
    incoming_layer : string, name of last layer from bottleneck layers
    '''
    n_unpool = len(conv_before_pool)

    i = n_unpool + (1 if n_conv_bottom > 0 else 0) #conv stage number

    for p in range(n_unpool, 0, -1):

        #Upscale layer by scale-factor
        net['upscale'+str(p)] = UpscaleLayer(
                net[incoming_layer], scale_factor = pool_size,mode='repeat')
        incoming_layer = 'upscale'+str(p)


        #First convolution after upsampling (part of deconv layer)
        #full padding to recover a larger image, that will then be
        #cropped in elementwise sum/concat layer to fit with the
        #layer in contractive path (to recover the odd sizes)
        net['deconv'+str(p)] = ConvLayer(
                net[incoming_layer],n_filters*(2**(p-1)), filter_size,
                pad='full') #pad='full' is necessary here
        incoming_layer = 'deconv'+str(p)

        nb_conv_stage = conv_before_pool[p-1]
        #This is the output of the last convolution layer before
        #the corresponding pooling layer in the contractive path
        layer_to_be_merged = 'conv'+str(p)+'_'+str(nb_conv_stage)

        if merge_type == 'sum':
            net['sum'+str(p)] = ElemwiseSumLayer(
                    [net[incoming_layer], net[layer_to_be_merged]],
                    cropping = [None, None, 'center'])
            incoming_layer = 'sum'+str(p)

        elif merge_type =='concat':
            #Since it increases the number of feature maps,
            #next convolution will downsample that number
            #to keep same nb of feature maps after all.
            net['concat'+str(p)] = ConcatLayer(
                    [net[incoming_layer], net[layer_to_be_merged]],
                    cropping = [None, None, 'center'])
            incoming_layer = 'concat'+str(p)
        else:
            raise Exception('No other merge type implemented')

        i+=1 #Only to name the following conv layers

        for c in range(2, nb_conv_stage+1):
            net['conv'+str(i)+'_'+str(c)] = ConvLayer(
                    net[incoming_layer],n_filters*(2**(p-1)), filter_size,
                    pad='same')
            incoming_layer = 'conv'+str(i)+'_'+str(c)

    return net, incoming_layer

def output_path(net, incoming_layer, n_classes,
                    filter_size, out_nonlin):
    '''
    Build the output path (including last conv layer to have n_classes
    feature maps). Dimshuffle layers to fit with softmax implementation

    Parameters
    ----------
    Same as above
    incoming_layer : string, name of last layer from bottleneck layers
    '''

    #Final convolution (n_classes feature maps) with filter_size = 1
    net['final_conv'] = ConvLayer(net[incoming_layer], n_classes, 1)

    #DimshuffleLayer and all this stuff is necessary to fit with softmax
    #implementation. In training, we specify layer = ['probs'] to have the
    #right layer but the 2 last reshape layers are necessary only to visualize
    #data.
    net['final_dimshuffle'] = DimshuffleLayer(net['final_conv'], (0,2,1))

    laySize = lasagne.layers.get_output(net['final_dimshuffle']).shape
    net['final_reshape'] = ReshapeLayer(net['final_dimshuffle'],
                            (T.prod(laySize[0:2]),laySize[2]))


    net['probs'] = NonlinearityLayer(net['final_reshape'], nonlinearity = out_nonlin)

    net['probs_reshape'] = ReshapeLayer(net['probs'],
                            (laySize[0], laySize[1], n_classes))

    net['probs_dimshuffle'] = DimshuffleLayer(net['probs_reshape'], (0,2,1))


    return net

if __name__=="__main__":
    input_var = T.tensor3('input_var')
    last_layer = buildFCN_1D(input_var = input_var, n_classes= 6,
        n_in_channels = 1,  merge_type = 'sum', layer=['probs'],
        n_filters = 64, conv_before_pool=[2,2,2,2], n_conv_bottom=3,
        pool_size=2)

    lays = lasagne.layers.get_all_layers(last_layer)
    for l in lays:
        print l, l.output_shape
