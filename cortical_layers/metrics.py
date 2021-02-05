import theano.tensor as T
import numpy as np
from theano import config
import theano

from lasagne.objectives import squared_error as squared_error_lasagne

_FLOATX = config.floatX
_EPSILON = 10e-8


def jaccard(y_pred, y_true, n_classes, one_hot=False):

    assert (y_pred.ndim == 2) or (y_pred.ndim == 1)

    # y_pred to indices
    if y_pred.ndim == 2:
        y_pred = T.argmax(y_pred, axis=1)

    if one_hot:
        y_true = T.argmax(y_true, axis=1)

    # Compute confusion matrix
    cm = T.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cm = T.set_subtensor(
                cm[i, j], T.sum(T.eq(y_pred, i) * T.eq(y_true, j)))

    # Compute Jaccard Index
    TP_perclass = T.cast(cm.diagonal(), _FLOATX)
    FP_perclass = cm.sum(1) - TP_perclass
    FN_perclass = cm.sum(0) - TP_perclass

    num = TP_perclass
    denom = TP_perclass + FP_perclass + FN_perclass

    return T.stack([num, denom], axis=0)


def accuracy(y_pred, y_true, void_labels=[], one_hot=False):

    assert (y_pred.ndim == 2)
    assert (y_true.ndim == 1) or ((y_true.ndim == 2) and  y_true.shape[1] == 1)

    # y_pred to indices
    y_pred = T.argmax(y_pred, axis=1)
    # Compute accuracy
    acc = T.eq(y_pred, y_true).astype(_FLOATX)

    reshape_per_sample = T.reshape(acc, (-1, 200))
    acc_per_sample = T.mean(reshape_per_sample, axis=1)

    return acc.mean(), acc_per_sample

def accuracy_regions(y_pred, y_true, one_hot=False):
    assert (y_pred.ndim == 2)
    assert (y_true.ndim == 1) or ((y_true.ndim == 2) and  y_true.shape[1] == 1)

    # y_pred to indices
    y_pred = T.argmax(y_pred, axis=1)
    # Compute accuracy
    acc = T.eq(y_pred, y_true).astype(_FLOATX)

    return acc.mean()


def crossentropy(y_pred, y_true, void_labels, one_hot=False):
    # Clip predictions
    y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)

    if one_hot:
        y_true = T.argmax(y_true, axis=1)

    # Create mask
    mask = T.ones_like(y_true, dtype=_FLOATX)
    for el in void_labels:
        mask = T.set_subtensor(mask[T.eq(y_true, el).nonzero()], 0.)

    # Modify y_true temporarily
    y_true_tmp = y_true * mask
    y_true_tmp = y_true_tmp.astype('int32')

    # Compute cross-entropy
    loss = T.nnet.categorical_crossentropy(y_pred, y_true_tmp)

    # Compute masked mean loss
    loss *= mask
    loss = T.sum(loss) / T.sum(mask)

    return loss



def weighted_crossentropy(y_pred, y_true, weights):
    # Clip predictions
    y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    #calculate crossentropy
    loss = T.nnet.categorical_crossentropy(y_pred, y_true)
  #  y_true= T.cast(y_true, 'int32')
 #   print weights[0]
    loss = loss * weights
    return loss



def binary_crossentropy(y_pred, y_true):
    # Clip predictions to avoid numerical instability
    y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)

    loss = T.nnet.binary_crossentropy(y_pred, y_true)
    return loss.mean()


def entropy(y_pred):
    # Clip predictions to avoid numerical instability
    y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)

    ent = - T.sum(y_pred * T.log(y_pred), axis=1)

    return ent.mean()


def squared_error_h(y_pred, y_true):

    coef = np.linspace(_EPSILON, 1, len(y_pred)+1)[:-1]

    error_list = [((a_i - b_i)**2).mean() for
                  a_i, b_i in zip(y_pred, y_true)]
    error_list = error_list * coef

    return sum(error_list)


def squared_error(y_pred, y_true, void):

    if void.shape[0] != 3:
        loss_aux = squared_error_lasagne(y_pred, y_true[:, :void, :, :]).mean(axis=1)
        mask = y_true[:, :void, :, :].sum(axis=1)
    else:  # assumes b,c,0,1
        loss_aux = squared_error_lasagne(y_pred, y_true).mean(axis=1)
        mask = T.neq(y_true.sum(1), sum(void))

    loss_aux = loss_aux * mask
    loss = loss_aux.sum()/mask.sum()

    return loss

def dice_loss(y_pred, y_true):
    '''
    Computes the dice loss.
    y_pred is one hot encoded, y_true is a vector of 0s and 1s
    From: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L23
    '''
    smooth = 1
    y_pred_f = T.flatten(y_pred[:,1])
    y_true_f = T.flatten(y_true)

    intersection = T.sum(y_true_f * y_pred_f)
    return -(2. * intersection + smooth) / (T.sum(y_true_f) + T.sum(y_pred_f) + smooth)




