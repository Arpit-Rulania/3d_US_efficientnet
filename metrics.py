import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import sum, flatten
from tensorflow.keras import backend as K

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

def dice_coe(y_true, y_pred):
    
    #print("-----------##----------")
    smooth = 1.
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)
    
    """
    epsilon = K.epsilon()
    axis = identify_axis(y_true.get_shape())

    y_pred = K.clip(y_pred,epsilon,1-epsilon)
    y_true = K.clip(y_true,epsilon,1-epsilon)

    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum((y_true * (1-y_pred))**2, axis=axis)
    fp = K.sum(((1-y_true) * y_pred)**2, axis=axis)
    d_cl = (2*tp + epsilon)/(2*tp + fn + fp + epsilon)
    loss = K.mean(1-d_cl)
    return loss
    """


def dice_loss(y_true,y_pred):
    return -dice_coe(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coe(y_true, y_pred)
