# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:41:28 2017

@author: Jie Hou
"""


from six.moves import range

import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Dense, Reshape, Activation, Dropout, Lambda, add, concatenate, Concatenate, multiply
from keras.layers import GlobalAveragePooling2D, Permute
from keras.layers.convolutional import Conv2D, Conv1D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import initializers, regularizers
import numpy as np

import tensorflow as tf
import sys
sys.setrecursionlimit(10000)
# Helper to build a conv -> BN -> relu block
def _conv_bn_relu1D_test(filters, kernel_size, strides,use_bias=True):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,use_bias=use_bias,
                             kernel_initializer="he_normal", activation='relu', padding="same")(input)
        return Activation("sigmoid")(conv)
    return f

# Helper to build a conv -> BN -> relu block
def _conv_bn_relu1D(filters, kernel_size, strides,use_bias=True, kernel_initializer = "he_normal"):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same")(input)
        norm = BatchNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

def _in_relu_conv1D(filters, kernel_size, strides,use_bias=True, kernel_initializer = "he_normal"):
    def f(input):
        act = _in_relu(input)
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same")(act)
        return conv
    return f

def _conv_in_relu1D(filters, kernel_size, strides,use_bias=True, kernel_initializer = "he_normal"):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same")(input)
        norm = InstanceNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

def _bn_relu(input):
    norm = BatchNormalization(axis=-1)(input)
    return Activation("relu")(norm)

def _in_relu(input):
    norm = InstanceNormalization(axis=-1)(input)
    return Activation("relu")(norm)

def _bn_relu_conv2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None, dilation_rate=(1,1)):
    def f(input):
        act = _bn_relu(input)
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer, dilation_rate = dilation_rate)(act)
        return conv
    return f

def _in_relu_conv2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None, dilation_rate=(1,1)):
    def f(input):
        act = _in_relu(input)
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer, dilation_rate = dilation_rate)(act)
        return conv
    return f

def _conv_bn_relu2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None, dilation_rate=(1,1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer, dilation_rate = dilation_rate)(input)
        # norm = BatchNormalization(axis=-1)(conv)
        norm = BatchNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

def _conv_in_relu2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None, dilation_rate=(1,1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer, dilation_rate = dilation_rate)(input)
        norm = InstanceNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

def _conv_relu1D(filters, kernel_size, strides, use_bias=True, kernel_initializer = "he_normal"):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same")(input)
        return Activation("relu")(conv)
    return f

# Helper to build a conv -> BN -> relu block
def _conv_relu2D(filters, nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal", dilation_rate=(1, 1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides, use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
        # norm = BatchNormalization(axis=1)(conv)
        return Activation("relu")(conv)
    
    return f

def _in_sigmoid(input):
    norm = InstanceNormalization(axis=-1)(input)
    return Activation("sigmoid")(norm)

def _in_sigmoid_conv2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None):
    def f(input):
        act = _in_sigmoid(input)
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer)(act)
        return conv
    return f

def _conv_in_sigmoid2D(filters, nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal", dilation_rate=(1, 1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
        norm = InstanceNormalization(axis=-1)(conv)
        return Activation("sigmoid")(conv)
    
    return f

def _conv_bn_sigmoid2D(filters, nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal", dilation_rate=(1, 1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
        norm = BatchNormalization(axis=-1)(conv)
        return Activation("sigmoid")(conv)
    
    return f

# Helper to build a conv -> BN -> softmax block
def _conv_bn_softmax1D(filters, kernel_size, strides, name,use_bias=True, kernel_initializer = "he_normal"):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same",name="%s_conv" % name)(input)
        norm = BatchNormalization(axis=-1,name="%s_nor" % name)(conv)
        return Dense(units=3, kernel_initializer=kernel_initializer,name="%s_softmax" % name, activation="softmax")(norm)
    
    return f



def _weighted_mean_squared_error(weight):

    def loss(y_true, y_pred):
        #set 20A as thresold
        # y_bool = Lambda(lambda x: x <= 20.0)(y_pred)
        y_bool = K.cast((y_true <= 20.0), dtype='float32')
        y_bool_invert = K.cast((y_true > 20.0), dtype='float32')
        y_mean = K.mean(y_true)
        y_pred_below = y_pred * y_bool 
        y_pred_upper = y_pred * y_bool_invert 
        y_true_below = y_true * y_bool 
        y_true_upper = y_true * y_bool_invert 
        # y_pred_upper = multiply([y_pred, y_bool_invert])
        # y_true_below = multiply([y_true, y_bool])
        # y_true_upper = multiply([y_true, y_bool_invert])
        weights1 = 1
        # weights2 = 0
        weights2 = 1/(1 + K.square(y_pred_upper/y_mean))
        return K.mean(K.square((y_pred_below-y_true_below))*weights1) + K.mean(K.square((y_pred_upper-y_true_upper))*weights2)
        # return add([K.mean(K.square((y_pred_below-y_true_below))*weights1), K.mean(K.square((y_pred_upper-y_true_upper))*weights2)], axis= -1)
    return loss

def _weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

def _weighted_binary_crossentropy(pos_weight=1, neg_weight=1):

    def loss(y_true, y_pred):
        binary_crossentropy = K.binary_crossentropy(y_true, y_pred)

        weights = y_true * pos_weight + (1. - y_true) * neg_weight

        weighted_binary_crossentropy_vector = weights * binary_crossentropy

        return K.mean(weighted_binary_crossentropy_vector)
    return loss

def _weighted_binary_crossentropy_shield(pos_weight=1, neg_weight=1, shield=0):

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
        # cross-entropy loss with weighting
        out = -(y_true * K.log(y_pred)*pos_weight+ (1.0 - y_true) * K.log(1.0 - y_pred)*neg_weight)
        return K.mean(out, axis=-1)
    return loss

def MaxoutAct(input, filters, kernel_size, output_dim, padding='same', activation = "relu"):
    output = None
    for _ in range(output_dim):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(input)
        activa = Activation(activation)(conv)
        maxout_out = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(activa)
        if output is not None:
            output = concatenate([output, maxout_out], axis=-1)
        else:
            output = maxout_out
    return output

def MaxoutCov(input, output_dim):
    output = None
    for i in range(output_dim):
        section = Lambda(lambda x:x[:,:,:,2*i:2*i+1])(input)
        maxout_out = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(section)
        if output is not None:
            output = concatenate([output, maxout_out], axis=-1)
        else:
            output = maxout_out
    return output

class InstanceNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis '+str(self.axis)+' of input tensor should have a defined dimension but the layer received an input with shape '+str(input_shape)+ '.')
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape, name='gamma', initializer=initializers.random_normal(1.0, 0.02))
        self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros')
        self.built = True

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
        return K.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon)


def identity_Block_sallow(input, filters, kernel_size, with_conv_shortcut=False,use_bias=True, mode='sum'):
    x = _conv_relu1D(filters=filters, kernel_size=3, strides=1,use_bias=use_bias)(input)
    x = _conv_relu1D(filters=filters, kernel_size=kernel_size, strides=1,use_bias=use_bias)(x)
    if mode=='sum':
        x = add([x, input])
    elif mode=='concat':
        x = concatenate([x, input], axis=-1)
    return x

def identity_Block_deep(input, filters, kernel_size, with_conv_shortcut=False,use_bias=True, mode='sum'):
    x = _conv_relu1D(filters=filters*2, kernel_size=1, strides=1,use_bias=use_bias)(input)
    x = _conv_relu1D(filters=filters, kernel_size=kernel_size, strides=1,use_bias=use_bias)(x)
    x = _conv_relu1D(filters=filters, kernel_size=1, strides=1,use_bias=use_bias)(x)
    if with_conv_shortcut:
        shortcut = _conv_relu1D(filters=filters, kernel_size=kernel_size, strides=1,use_bias=use_bias)(input)
        if mode=='sum':
            x = add([x, shortcut])
        elif mode=='concat':
            x = concatenate([x, shortcut], axis=-1)
        return x
    else:
        if mode=='sum':
            x = add([x, input])
        elif mode=='concat':
            x = concatenate([x, input], axis=-1)
        return x

def identity_Block_sallow_2D(input, filters, nb_row, nb_col, with_conv_shortcut=False,use_bias=True, mode='sum', kernel_initializer='he_normal', dilation_rate=(1, 1)):
    x = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
    x = Activation("relu")(x)
    x = InstanceNormalization(axis=-1)(x)
    # x = Dropout(0.4)(x)
    x = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(x)
    if with_conv_shortcut:
        shortcut = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same")(input)
        if mode=='sum':
            x = add([x, shortcut])
        elif mode=='concat':
            x = concatenate([x, shortcut], axis=-1)
        x = Activation("relu")(x)
        return x
    else:
        if mode=='sum':
            x = add([x, input])
        elif mode=='concat':
            x = concatenate([x, input], axis=-1)
        x = Activation("relu")(x)
        return x

def identity_Block_deep_2D(input, filters, nb_row, nb_col, with_conv_shortcut=False,use_bias=True, mode='sum', kernel_initializer='he_normal', dilation_rate=(1,1 )):
    # x = Conv2D(filters=np.int32(filters/4), kernel_size=(1, 1), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
    # x = Activation("relu")(x)
    # x = Conv2D(filters=np.int32(filters/4), kernel_size=(3, 3), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(x)
    # x = Activation("relu")(x)
    # x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(x)
    x = _conv_bn_relu2D(filters=filters, nb_row = 1, nb_col = 1, strides=(1, 1), use_bias=use_bias, kernel_initializer=kernel_initializer)(input)
    x = _conv_bn_relu2D(filters=filters, nb_row = nb_row, nb_col = nb_col, strides=(1, 1), use_bias=use_bias, kernel_initializer=kernel_initializer)(x)
    x = _conv_bn_relu2D(filters=filters, nb_row=1, nb_col = 1, strides=(1, 1), use_bias=use_bias, kernel_initializer=kernel_initializer)(x)
    if with_conv_shortcut:
        # shortcut = _conv_bn_relu2D(filters=filters, nb_row = nb_row, nb_col = nb_col, strides=(1, 1), kernel_initializer=kernel_initializer)(input)
        shortcut = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
        if mode=='sum':
            x = add([x, shortcut])
        elif mode=='concat':
            x = concatenate([x, shortcut], axis=-1)
        x = Activation("relu")(x)
        return x
    else:
        if mode=='sum':
            x = add([x, input])
        elif mode=='concat':
            x = concatenate([x, input], axis=-1)
        x = Activation("relu")(x)
        return x

def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3

def _shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    # stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    # stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    stride_width = 1
    stride_height = 1
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    # if stride_width > 1 or stride_height > 1 or not equal_channels:
    #     shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
    #                       kernel_size=(1, 1),
    #                       strides=(stride_width, stride_height),
    #                       padding="valid",
    #                       kernel_initializer="he_normal",
    #                       kernel_regularizer=regularizers.l2(0.0001))(input)
    if not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal")(input)
    return add([shortcut, residual])

def _residual_block(block_function, filters, repetitions, is_first_layer=False, use_SE=False):
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                # init_strides = (2, 2)
                init_strides = (1, 1)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0), use_SE = use_SE)(input)
        return input

    return f

def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, use_SE=False):
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal")(input)
                           # ,
                           # kernel_regularizer=regularizers.l2(1e-4)
        else:
            conv1 = _in_relu_conv2D(filters=filters, nb_row=3, nb_col=3,
                                  strides=init_strides)(input)

        residual = _in_relu_conv2D(filters=filters, nb_row=3, nb_col=3)(conv1)
        if use_SE == True:
            residual = squeeze_excite_block(residual)
        return _shortcut(input, residual)
    return f

def basic_block_pre(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, use_SE=False):
    def f(input):
        residual = input
        conv1 = _conv_in_dropout_relu2D(filters=filters, nb_row=3, nb_col=3,strides=init_strides, dropout_rate=0.2)(input)#dropout_rate=0.2
        conv2 = _conv_in_dropout_relu2D(filters=filters, nb_row=3, nb_col=3,strides=init_strides, dropout_rate=0.2)(conv1)
        if use_SE == True:
            conv2 = squeeze_excite_block(conv2)
        return _shortcut(residual, conv2)
    return f

def basic_block_other(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, use_SE=False):
    def f(input):
        residual = input
        conv1 = _conv_in_dropout_relu2D(filters=filters, nb_row=3, nb_col=3,strides=init_strides)(input)#
        conv2 = _conv_in_dropout_relu2D(filters=filters, nb_row=3, nb_col=3,strides=init_strides)(conv1)
        if use_SE == True:
            conv2 = squeeze_excite_block(conv2)
        return _shortcut(residual, conv2)
    return f

def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, use_SE=False):
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=regularizers.l2(1e-4))(input)
        else:
            conv_1_1 = _in_relu_conv2D(filters=filters, nb_row=1, nb_col=1,
                                     strides=init_strides)(input)

        conv_3_3 = _in_relu_conv2D(filters=filters, nb_row=3, nb_col=3)(conv_1_1)
        residual = _in_relu_conv2D(filters=filters * 2, nb_row=1, nb_col=1)(conv_3_3)
        if use_SE == True:
            residual = squeeze_excite_block(residual)
        return _shortcut(input, residual)

    return f

def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def _bn_relu_K(x, bn_name=None, relu_name=None):
    """Helper to build a BN -> relu block
    """
    # norm = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
    norm = InstanceNormalization(axis=-1, name=bn_name)(x)
    return Activation("relu", name=relu_name)(norm)

def _conv_bn_relu_K(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", regularizers.l2(1.e-4))

    def f(x):
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding=padding,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=conv_name)(x)
        return _bn_relu_K(x, bn_name=bn_name, relu_name=relu_name)

    return f

def _bn_relu_conv_K(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", regularizers.l2(1.e-4))

    def f(x):
        activation = _bn_relu_K(x, bn_name=bn_name, relu_name=relu_name)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,
                      name=conv_name)(activation)
                      # kernel_regularizer=kernel_regularizer,

    return f

def _shortcut_K(input_feature, residual, conv_name_base=None, bn_name_base=None):
    input_shape = K.int_shape(input_feature)
    residual_shape = K.int_shape(residual)
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input_feature
    # 1 X 1 conv if shape is different. Else identity.
    if not equal_channels:
        print('reshaping via a convolution...')
        if conv_name_base is not None:
            conv_name_base = conv_name_base + '1'
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding="valid",
                          kernel_initializer="he_normal",
                          name=conv_name_base)(input_feature) 
                          # kernel_regularizer=regularizers.l2(0.0001),
        if bn_name_base is not None:
            bn_name_base = bn_name_base + '1'
        # shortcut = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base)(shortcut)
        # shortcut = InstanceNormalization(axis=CHANNEL_AXIS, name=bn_name_base)(shortcut)

    return add([shortcut, residual])

def _residual_block_K(block_function, filters, blocks, stage, transition_strides=None, transition_dilation_rates=None, dilation_rates=None, 
    is_first_layer=False, dropout=None, residual_unit=_bn_relu_conv_K, use_SE = False):
    if transition_dilation_rates is None:
        transition_dilation_rates = [(1, 1)] * blocks
    if transition_strides is None:
        transition_strides = [(1, 1)] * blocks
    if dilation_rates is None:
        dilation_rates = [1] * blocks

    def f(x):
        for i in range(blocks):
            is_first_block = is_first_layer and i == 0
            x = block_function(filters=filters, stage=stage, block=i,
                               transition_strides=transition_strides[i],
                               dilation_rate=dilation_rates[i],
                               is_first_block_of_first_layer=is_first_block,
                               dropout=dropout,
                               residual_unit=residual_unit, use_SE = use_SE)(x)
        return x

    return f

def _block_name_base_K(stage, block):
    """Get the convolution name base and batch normalization name base defined by
    stage and block.
    If there are less than 26 blocks they will be labeled 'a', 'b', 'c' to match the
    paper and keras and beyond 26 blocks they will simply be numbered.
    """
    if block < 27:
        block = '%c' % (block + 97)  # 97 is the ascii number for lowercase 'a'
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'
    return conv_name_base, bn_name_base

def basic_block_K(filters, stage, block, transition_strides=(1, 1), dilation_rate=(1, 1), is_first_block_of_first_layer=False, dropout=None, residual_unit=_bn_relu_conv_K, use_SE = False):
    def f(input_features):
        conv_name_base, bn_name_base = _block_name_base_K(stage, block)
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            x = Conv2D(filters=filters, kernel_size=(3, 3),
                       strides=(1, 1),
                       dilation_rate=dilation_rate,
                       padding="same",
                       kernel_initializer="he_normal",
                       name=conv_name_base + '2a')(input_features)
                       # kernel_regularizer=regularizers.l2(1e-4),
        else:
            x = residual_unit(filters=filters, kernel_size=(3, 3),
                              strides=(1, 1),
                              dilation_rate=dilation_rate,
                              conv_name_base=conv_name_base + '2a',
                              bn_name_base=bn_name_base + '2a')(input_features)

        if dropout is not None:
            x = Dropout(dropout)(x)

        x = residual_unit(filters=filters, kernel_size=(3, 3),
                          conv_name_base=conv_name_base + '2b',
                          bn_name_base=bn_name_base + '2b')(x)

        if use_SE == True:
            x = squeeze_excite_block(x)
        return _shortcut_K(input_features, x)

    return f

def bottle_neck_K(filters, stage, block, transition_strides=(1, 1), dilation_rate=(1, 1), is_first_block_of_first_layer=False, dropout=None, residual_unit=_bn_relu_conv_K, use_SE = False):
    def f(input_feature):
        conv_name_base, bn_name_base = _block_name_base_K(stage, block)
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            x = Conv2D(filters=filters, kernel_size=(1, 1),
                       strides=transition_strides,
                       dilation_rate=dilation_rate,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=regularizers.l2(1e-4),
                       name=conv_name_base + '2a')(input_feature)
        else:
            x = residual_unit(filters=filters, kernel_size=(1, 1),
                              strides=transition_strides,
                              dilation_rate=dilation_rate,
                              conv_name_base=conv_name_base + '2a',
                              bn_name_base=bn_name_base + '2a')(input_feature)

        if dropout is not None:
            x = Dropout(dropout)(x)

        x = residual_unit(filters=filters, kernel_size=(3, 3),
                          conv_name_base=conv_name_base + '2b',
                          bn_name_base=bn_name_base + '2b')(x)

        if dropout is not None:
            x = Dropout(dropout)(x)

        x = residual_unit(filters=filters * 2, kernel_size=(1, 1),
                          conv_name_base=conv_name_base + '2c',
                          bn_name_base=bn_name_base + '2c')(x)
        if use_SE == True:
            x = squeeze_excite_block(x)
        return _shortcut_K(input_feature, x)

    return f

def DilatedRes_with_paras_2D(kernel_size,feature_2D_num,use_bias,hidden_type,filters,nb_layers,opt, initializer = "he_normal", loss_function = "weighted_BCE", weight_p=1.0, weight_n=1.0):
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(None,None,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)
    
    _handle_dim_ordering()
    ######################### now merge new data to new architecture
    DNCON4_2D_input = contact_input
    DNCON4_2D_conv = DNCON4_2D_input
    DNCON4_2D_conv = InstanceNormalization(axis=-1)(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation('relu')(DNCON4_2D_conv)
    DNCON4_2D_conv = Conv2D(128, 1, padding = 'same')(DNCON4_2D_conv)
    DNCON4_2D_conv = MaxoutAct(DNCON4_2D_conv, filters=4, kernel_size=(1,1), output_dim=64, padding='same', activation = "relu")

    DNCON4_2D_conv = _conv_bn_relu_K(filters=filters, kernel_size=7, strides=1)(DNCON4_2D_conv)

    # DNCON4_2D_conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1,1),use_bias=True, kernel_initializer=initializer, padding="same")(DNCON4_2D_conv)
    block = DNCON4_2D_conv
    filters = filters
    residual_unit = _bn_relu_conv_K
    transition_dilation_rate = [(1, 1),(3, 3),(5, 5),(7, 7)]
    # transition_dilation_rate = [(1, 1)]
    repetitions=[3, 4, 6, 3]
    dropout = None
    for i, r in enumerate(repetitions):
        transition_dilation_rates = transition_dilation_rate * r
        # transition_dilation_rates = transition_dilation_rate
        transition_strides = [(1, 1)] * r  

        # block = _residual_block(basic_block, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
        block = _residual_block_K(basic_block_K, filters=filters,
                                stage=i, blocks=r,
                                is_first_layer=(i == 0),
                                dropout=dropout,
                                transition_dilation_rates=transition_dilation_rates,
                                transition_strides=transition_strides,
                                residual_unit=residual_unit, use_SE = True)(block)

    # for i, r in enumerate(repetitions):
    #     block = _residual_block(basic_block, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
    # Last activation
    block = _bn_relu_K(block)
    DNCON4_2D_conv = block

    DNCON4_2D_conv = _conv_in_sigmoid2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
    loss = loss_function
        
    DNCON4_2D_out = DNCON4_2D_conv
    DNCON4_RES = Model(inputs=contact_input, outputs=DNCON4_2D_out)
    DNCON4_RES.compile(loss=loss, metrics=['accuracy'], optimizer=opt)
    DNCON4_RES.summary()
    return DNCON4_RES

##############################attention model configs###############################
from attention_channel_utils import att_channel_config1,att_channel_config2, att_channel_config3    
from keras.layers import Softmax, Multiply, Conv3D, Add

def select_channel_config(att_config=0):
    if att_config==1:
        return att_channel_config1
    elif att_config==2:
        return att_channel_config2
    elif att_config==3:
        return att_channel_config3
    else:
        return None

def get_stretch_weight(region_size,depth=1):
    w = np.zeros((region_size,region_size,depth,depth*region_size*region_size))
    for k in range(depth):
        for i in range(region_size):
            for j in range(region_size):
                w[i,j,k,k*region_size*region_size+region_size*i+j] = 1
    return np.asarray(w)   
    
def regional_attention_layers(input_layer,region_size=3,att_outdim=16):
    depth = input_layer.shape.as_list()[3]
    stretch_layer = Conv2D(region_size*region_size*depth, (region_size,region_size),  \
                           padding='same',activation=None, use_bias=False, \
                           bias_constraint=None, trainable=False, name = 'strech_layer')
    out_tensor = stretch_layer(input_layer)
    out_tensor1 = out_tensor
    if att_outdim > 0:
        out_tensor = Dense(att_outdim)(out_tensor1)
    out_tensor2 = Dense(region_size*region_size*depth)(out_tensor1)
    out_tensor3 = Softmax(3,name='attention_weights')(out_tensor2)
    out_tensor4 = Multiply()([out_tensor3,out_tensor])
    out_tensor5 = Lambda(lambda x: K.sum(x,3,True))(out_tensor4)
    
    return out_tensor5

def get_stretch_weight3D(region_size):
    w = np.zeros((region_size,region_size,1,1,region_size*region_size))
    for i in range(region_size):
        for j in range(region_size):
            w[i,j,0,0,region_size*i+j] = 1
    return np.asarray(w)

def regional_attention_layersV1(input_layer,region_size=3,att_outdim=0):
    input_layer = Lambda(lambda x:K.expand_dims(x))(input_layer)
    
    stretch_layer3D = Conv3D(region_size*region_size, (region_size,region_size,1),  \
                               padding='same',activation=None, use_bias=False, \
                               bias_constraint=None, trainable=False, name = 'strech_layer')
    
    out_tensor = stretch_layer3D(input_layer)
    out_tensor1 = out_tensor
    if att_outdim > 0:
        out_tensor1 = Dense(att_outdim)(out_tensor1)
    out_tensor2 = Dense(region_size*region_size)(out_tensor1)
    out_tensor3 = Softmax(4,name='attention_weights')(out_tensor2)
    out_tensor4 = Multiply()([out_tensor3,out_tensor])
    out_tensor5 = Lambda(lambda x: K.sum(x,4,False),name='attention_output')(out_tensor4)
    
    return out_tensor5

def regional_attention_layersV2(input_layer,region_size=3,att_outdim=0):
    input_layer = Lambda(lambda x:K.expand_dims(x))(input_layer)
    
    stretch_layer3D = Conv3D(region_size*region_size, (region_size,region_size,1),  \
                               padding='same',activation=None, use_bias=False, \
                               bias_constraint=None, trainable=False, name = 'strech_layer')
    
    out_tensor = stretch_layer3D(input_layer)
    
    att_in_layers_list = []
    for i in range(out_tensor.shape.as_list()[3]):
        att_input1 = Lambda(lambda x: x[:, :, :,i:(i+1):,:], name='attention_input_'+str(i))(out_tensor)
        att_input2 = Dense(region_size*region_size)(att_input1)
        att_in_layers_list.append(att_input2)
    
    if len(att_in_layers_list)>1:
        out_tensor1 = Concatenate(axis=3)(att_in_layers_list)
    else:
        out_tensor1 = att_in_layers_list[0]
    out_tensor2 = Softmax(4,name='attention_weights')(out_tensor1)
    out_tensor3 = Multiply()([out_tensor2,out_tensor])
    out_tensor4 = Lambda(lambda x: K.sum(x,4,False))(out_tensor3)
    
    return out_tensor4

def select_regional_config(att_config=1):
    if att_config==2:
        return regional_attention_layersV2
    else:
        return regional_attention_layersV1

def channel_attention(kernel_size=3,feature_2D_num=481,use_bias=True,hidden_type='sigmoid', \
                       filters = 64,nb_layers = 34,opt = 'Adam', initializer = "he_normal", \
                       loss_function = "binary_crossentropy", weight_p=1.0,weight_n=1.0, \
                       att_config=0,kmax=7,att_outdim=16,insert_pos = 'none'):
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(None,None,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)
    
    _handle_dim_ordering()
    ######################### now merge new data to new architecture
    DNCON4_2D_input = contact_input
    DNCON4_2D_conv = DNCON4_2D_input
    DNCON4_2D_conv = InstanceNormalization(axis=-1)(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation('relu')(DNCON4_2D_conv)
    attention_config = select_channel_config(att_config)
    if (not (attention_config is None)) and insert_pos == 'head':
        DNCON4_2D_conv = attention_config(DNCON4_2D_conv,kmax=kmax,att_outdim=att_outdim)
    
    DNCON4_2D_conv = Conv2D(128, 1, padding = 'same')(DNCON4_2D_conv)
    DNCON4_2D_conv = MaxoutAct(DNCON4_2D_conv, filters=4, kernel_size=(1,1), output_dim=64, \
                               padding='same', activation = "relu")

    DNCON4_2D_conv = _conv_bn_relu_K(filters=filters, kernel_size=7, strides=1)(DNCON4_2D_conv)

    block = DNCON4_2D_conv
    filters = filters
    residual_unit = _bn_relu_conv_K
    transition_dilation_rate = [(1, 1),(3, 3),(5, 5),(7, 7)]
    # transition_dilation_rate = [(1, 1)]
    repetitions=[3, 4, 6, 3]
    dropout = None
    for i, r in enumerate(repetitions):
        transition_dilation_rates = transition_dilation_rate * r
        # transition_dilation_rates = transition_dilation_rate
        transition_strides = [(1, 1)] * r  

        block = _residual_block_K(basic_block_K, filters=filters,
                                stage=i, blocks=r,
                                is_first_layer=(i == 0),
                                dropout=dropout,
                                transition_dilation_rates=transition_dilation_rates,
                                transition_strides=transition_strides,
                                residual_unit=residual_unit, use_SE = True)(block)

    # Last activation
    block = _bn_relu_K(block)
    DNCON4_2D_conv = block
    if (not (attention_config is None)) and insert_pos == 'tail':
        DNCON4_2D_conv = attention_config(DNCON4_2D_conv,kmax=kmax,att_outdim=att_outdim)
        
    DNCON4_2D_conv = _conv_in_sigmoid2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), \
                                        kernel_initializer=initializer)(DNCON4_2D_conv)
    loss = loss_function
        
    DNCON4_2D_out = DNCON4_2D_conv
    DNCON4_RES = Model(inputs=contact_input, outputs=DNCON4_2D_out)
    DNCON4_RES.compile(loss=loss, metrics=['accuracy'], optimizer=opt)
    DNCON4_RES.summary()
    return DNCON4_RES


def regional_attention(kernel_size=3,feature_2D_num=481,use_bias=True,hidden_type='sigmoid', \
                       filters = 64,nb_layers = 34,opt = 'Adam', initializer = "he_normal", \
                       loss_function = "binary_crossentropy", weight_p=1.0,weight_n=1.0, \
                       region_size = 3,att_outdim=0,insert_pos = 'none'):
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(None,None,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)
    
    _handle_dim_ordering()
    ######################### now merge new data to new architecture
    DNCON4_2D_input = contact_input
    DNCON4_2D_conv = DNCON4_2D_input
    DNCON4_2D_conv = InstanceNormalization(axis=-1)(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation('relu')(DNCON4_2D_conv)

    
    DNCON4_2D_conv = Conv2D(128, 1, padding = 'same')(DNCON4_2D_conv)
    DNCON4_2D_conv = MaxoutAct(DNCON4_2D_conv, filters=4, kernel_size=(1,1), output_dim=64, \
                               padding='same', activation = "relu")
    if insert_pos == 'head':
        DNCON4_2D_conv = regional_attention_layers(DNCON4_2D_conv,region_size,att_outdim)
    DNCON4_2D_conv = _conv_bn_relu_K(filters=filters, kernel_size=7, strides=1)(DNCON4_2D_conv)

    block = DNCON4_2D_conv
    filters = filters
    residual_unit = _bn_relu_conv_K
    transition_dilation_rate = [(1, 1),(3, 3),(5, 5),(7, 7)]
    # transition_dilation_rate = [(1, 1)]
    repetitions=[3, 4, 6, 3]
    dropout = None
    for i, r in enumerate(repetitions):
        transition_dilation_rates = transition_dilation_rate * r
        # transition_dilation_rates = transition_dilation_rate
        transition_strides = [(1, 1)] * r  

        block = _residual_block_K(basic_block_K, filters=filters,
                                stage=i, blocks=r,
                                is_first_layer=(i == 0),
                                dropout=dropout,
                                transition_dilation_rates=transition_dilation_rates,
                                transition_strides=transition_strides,
                                residual_unit=residual_unit, use_SE = True)(block)

    # Last activation
    block = _bn_relu_K(block)
    DNCON4_2D_conv = block

    if insert_pos == 'last_conv':
        DNCON4_2D_conv = regional_attention_layers(DNCON4_2D_conv,region_size,att_outdim)
        
    DNCON4_2D_conv = _conv_in_sigmoid2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), \
                                        kernel_initializer=initializer)(DNCON4_2D_conv)
    if insert_pos == 'tail':
        DNCON4_2D_conv = regional_attention_layers(DNCON4_2D_conv,region_size,att_outdim)
    loss = loss_function
        
    DNCON4_2D_out = DNCON4_2D_conv
    DNCON4_RES = Model(inputs=contact_input, outputs=DNCON4_2D_out)
    DNCON4_RES.compile(loss=loss, metrics=['accuracy'], optimizer=opt)
    
    if insert_pos == 'head' or insert_pos == 'last_conv' or insert_pos == 'tail':
        depth = DNCON4_RES.get_layer('strech_layer').get_input_shape_at(0)[3]
        stretch_w = get_stretch_weight(region_size,depth=depth)
        DNCON4_RES.get_layer('strech_layer').set_weights([stretch_w])

    DNCON4_RES.summary()
    return DNCON4_RES

def regional_attention3D(kernel_size=3,feature_2D_num=481,use_bias=True,hidden_type='sigmoid', \
                       filters = 64,nb_layers = 34,opt = 'Adam', initializer = "he_normal", \
                       loss_function = "binary_crossentropy", weight_p=1.0,weight_n=1.0, \
                       region_size = 3,att_config = 1,att_outdim=0,insert_pos = 'none',L=None):
    
    regional_attention_layers3D = select_regional_config(att_config)
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(L,L,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape,name='input2d_layer')
    
    _handle_dim_ordering()
    ######################### now merge new data to new architecture
    DNCON4_2D_input = contact_input
    DNCON4_2D_conv = DNCON4_2D_input
    DNCON4_2D_conv = InstanceNormalization(axis=-1)(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation('relu')(DNCON4_2D_conv)

    
    DNCON4_2D_conv = Conv2D(128, 1, padding = 'same')(DNCON4_2D_conv)
    DNCON4_2D_conv = MaxoutAct(DNCON4_2D_conv, filters=4, kernel_size=(1,1), output_dim=64, \
                               padding='same', activation = "relu")
    if insert_pos == 'head':
        DNCON4_2D_conv = regional_attention_layers3D(DNCON4_2D_conv,region_size,att_outdim)
    DNCON4_2D_conv = _conv_bn_relu_K(filters=filters, kernel_size=7, strides=1)(DNCON4_2D_conv)

    block = DNCON4_2D_conv
    filters = filters
    residual_unit = _bn_relu_conv_K
    transition_dilation_rate = [(1, 1),(3, 3),(5, 5),(7, 7)]
    # transition_dilation_rate = [(1, 1)]
    repetitions=[3, 4, 6, 3]
    dropout = None
    for i, r in enumerate(repetitions):
        transition_dilation_rates = transition_dilation_rate * r
        # transition_dilation_rates = transition_dilation_rate
        transition_strides = [(1, 1)] * r  

        block = _residual_block_K(basic_block_K, filters=filters,
                                stage=i, blocks=r,
                                is_first_layer=(i == 0),
                                dropout=dropout,
                                transition_dilation_rates=transition_dilation_rates,
                                transition_strides=transition_strides,
                                residual_unit=residual_unit, use_SE = True)(block)

    # Last activation
    block = _bn_relu_K(block)
    DNCON4_2D_conv = block

    if insert_pos == 'last_conv':
        DNCON4_2D_conv = regional_attention_layers3D(DNCON4_2D_conv,region_size,att_outdim)
        
    DNCON4_2D_conv = _conv_in_sigmoid2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), \
                                        kernel_initializer=initializer)(DNCON4_2D_conv)
    if insert_pos == 'tail':
        DNCON4_2D_conv = regional_attention_layers3D(DNCON4_2D_conv,region_size,att_outdim)
    loss = loss_function
        
    DNCON4_2D_out = DNCON4_2D_conv
    DNCON4_RES = Model(inputs=contact_input, outputs=DNCON4_2D_out)
    DNCON4_RES.compile(loss=loss, metrics=['accuracy'], optimizer=opt)
    
    if insert_pos == 'head' or insert_pos == 'last_conv' or insert_pos == 'tail':
        stretch_w = get_stretch_weight3D(region_size)
        DNCON4_RES.get_layer('strech_layer').set_weights([stretch_w])

    DNCON4_RES.summary()
    return DNCON4_RES

############################################################################
from attention_utils import LayerNormalization, SelfAttention, basic_1d, ResNet1D
from keras.layers import Embedding
    
def GetPosEncodingMatrix(max_len, d_emb):
	pos_enc = np.array([
		[pos / np.power(1000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
		if pos != 0 else np.zeros(d_emb) 
			for pos in range(max_len)
			])
	pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
	pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
	return pos_enc

class PosEncodingLayer:
	def __init__(self, max_len, d_emb):
		self.pos_emb_matrix = Embedding(max_len, d_emb, trainable=False, \
						   weights=[GetPosEncodingMatrix(max_len, d_emb)])
	def get_pos_seq(self, x):
		x1 = Lambda(lambda x: x[:,:,0])(x)
		pos = K.cumsum(K.ones_like(x1, 'int32'), 1)
		return pos
	def __call__(self, seq, pos_input=False):
		x = seq
		if not pos_input: x = Lambda(self.get_pos_seq)(x)
		return self.pos_emb_matrix(x)
    
def PosEncodingLayer2d(x,d_emb = 64,max_len =1000):
    x1 = Lambda(lambda x: x[:,:,0,0])(x)
    ones1d = Lambda(lambda x:K.ones_like(x, 'int32'))(x1)
    pos = Lambda(lambda x:K.cumsum(x, 1))(ones1d)
    pos_emb_matrix = Embedding(max_len, d_emb//2, trainable=False, \
    						   weights=[GetPosEncodingMatrix(max_len, d_emb//2)])
    encode1d = pos_emb_matrix(pos)
    s = tf.shape(encode1d)
    encode1dp = Lambda(lambda x:tf.reshape(x, [s[0], s[2], s[1]]))(encode1d)
    encode1d_repeat = Lambda(lambda x:K.expand_dims(x))(encode1dp)
    
    x1_size1 = Lambda(lambda x: x[0,:])(x1)
    x1_size12d = Lambda(lambda x: K.expand_dims(x))(x1_size1)
    z1 = Lambda(lambda x: tf.transpose(x, [1, 0]))(x1_size12d)
    
    one_mat = Lambda(lambda x:K.ones_like(x, 'float32'))(z1)
    encode1d_repeat2d = Lambda(lambda x:tf.matmul(x[0],x[1]))([encode1d_repeat,one_mat])
    
    encode1d_row = Lambda(lambda x:tf.transpose(x, [0, 2, 3, 1]))(encode1d_repeat2d)
    encode1d_col = Lambda(lambda x:tf.transpose(x, [0, 2, 1, 3]))(encode1d_row)
    
    encode2d = Lambda(lambda x: tf.concat([x[0],x[1]],axis=3))([encode1d_row,encode1d_col])
    return encode2d
    
add_layer = Lambda(lambda x:x[0]+x[1], output_shape=lambda x:x[0]) 
    
def process_1d_input(input1d,attention_depth,n_head,config_1d = 0, pe = 0):
    # config_1d: 
    #    0 : self_attention
    #    1 : resnet1D
    # pe 
    #    0 : no positional embedding 
    #    1 : 1d sinusoid positional embedding
    #    2 : 1d and 2d sinusoid positional embedding
    #    3 : 2d sinusoid positional embedding
    DNCON4_1D_conv = Lambda(lambda x: x[:,0,:,:])(input1d)
    DNCON4_1D_conv = Conv1D(attention_depth, 3, padding = 'valid')(DNCON4_1D_conv)
    if pe == 1 or pe == 2:
        pos_emb = PosEncodingLayer(1000, attention_depth)
        DNCON4_1D_conv = add_layer([DNCON4_1D_conv, pos_emb(DNCON4_1D_conv)])
    if config_1d == 0:
        # self attention
        d_v = attention_depth
        DNCON4_1D_conv = SelfAttention(d_v,d_v,n_head=n_head)(DNCON4_1D_conv,DNCON4_1D_conv)
    elif config_1d == 1:
        # resnet1D
        blocks = [2, 2, 2]
        DNCON4_1D_conv = ResNet1D(blocks,basic_1d,attention_depth)(DNCON4_1D_conv)
    else:
        print("Undefined config_1d: "+str(config_1d))
        sys.exit(1)    
    return DNCON4_1D_conv

def sequence_attention_layer(input2d,input1d, config_merge = 0,n_head = 4, att_dropout=0.1,idx = 0):
    # both input1 and input2 are n by L by L by d, but we only want a row of input2 since it is 1d feature

    # config_merge: 
    #    0 : concat style
    #    1 : resnet style  
    attention_depth = input2d.shape.as_list()[3]
    n_head = min([n_head,attention_depth])
    
    ##################################################################
    # multihead attention, total dim=64 and n_head = 4
    d_q = d_k = d_v = attention_depth//n_head
#====================================clear implementation======================================    
#    heads = []
#    att_w_list = []
#    for i in range(n_head):
#        qs_layer = Dense(d_q, use_bias=False)
#        ks_layer = Dense(d_k, use_bias=False)
#        vs_layer = Dense(d_v, use_bias=False)
#    
#        qs = qs_layer(input1)  # [batch_size, len_q, n_head*d_k]
#        ks = ks_layer(DNCON4_1D_conv)
#        vs = vs_layer(DNCON4_1D_conv)
#    
#        ks_t = Permute((2,1))(ks)
#        QK = Lambda(lambda x:tf.einsum('ijkl,ilm->ijkm', x[0],x[1]))([qs, ks_t])
#        QK_norm = Lambda(lambda x: x * 1/np.sqrt(d_q))(QK)
#        attention_weights = Softmax(name='attention_weights_'+str(i))(QK_norm)
#        attention_weights1 = Dropout(att_dropout)(attention_weights)
#        attention_output_i = Lambda(lambda x:tf.einsum('ijkl,ilm->ijkm', x[0],x[1]))([attention_weights1,vs])
#        heads.append(attention_output_i)
#        att_w_list.append(attention_weights)
#        
#    attention_outputs = Concatenate()(heads) if n_head > 1 else heads[0]
#    attention_weights = Concatenate()(att_w_list) if n_head > 1 else heads[0]
#
#====================================efficient implementation======================================        
    qs_layer = Dense(n_head*d_q, use_bias=False)
    ks_layer = Dense(n_head*d_k, use_bias=False)
    vs_layer = Dense(n_head*d_v, use_bias=False)    
    qs = qs_layer(input2d)  # [batch_size, len_q, n_head*d_k]
    ks = ks_layer(input1d)
    vs = vs_layer(input1d)

    def reshape1d_split(x):
        s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
        x = tf.reshape(x, [s[0], s[1], n_head, s[2]//n_head])
        x = tf.transpose(x, [2, 0, 1, 3])  
        x = tf.reshape(x, [-1, s[1], s[2]//n_head])  # [n_head * batch_size, len_q, d_k]
        return x

    def reshape2d_split(x):
        s = tf.shape(x)   # [batch_size, len_v, len_v, n_head * d_v]
        x = tf.reshape(x, [s[0], s[1], s[2], n_head, s[3]//n_head])
        x = tf.transpose(x, [3, 0, 1, 2, 4])  
        x = tf.reshape(x, [-1, s[1], s[2], s[3]//n_head])  # [n_head * batch_size, len_v, len_v, d_v]
        return x
    
    qs = Lambda(reshape2d_split)(qs)
    ks = Lambda(reshape1d_split)(ks)
    vs = Lambda(reshape1d_split)(vs)

    ks_t = Permute((2,1))(ks)
    QK = Lambda(lambda x:tf.einsum('ijkl,ilm->ijkm', x[0],x[1]))([qs, ks_t])
    QK_norm = Lambda(lambda x: x * 1/np.sqrt(d_q))(QK)
    attention_weights = Softmax(name='attention_weights_layer'+str(idx))(QK_norm)
    attention_weights1 = Dropout(att_dropout)(attention_weights)
    attention_output = Lambda(lambda x:tf.einsum('ijkl,ilm->ijkm', x[0],x[1]))([attention_weights1,vs])

    def reshape2d_merge(x):
        s = tf.shape(x)   # [n_head * batch_size, len_v, len_v, d_v]
        x = tf.reshape(x, [n_head, -1, s[1], s[2], s[3]]) 
        x = tf.transpose(x, [1, 2, 3, 0, 4])
        x = tf.reshape(x, [-1, s[1], s[2], n_head*d_v])  # [batch_size, len_v, len_v, n_head * d_v]
        return x
    
    head = Lambda(reshape2d_merge)(attention_output)
            
    attention_outputs = Dense(attention_depth)(head)
    attention_outputs = Dropout(att_dropout)(attention_outputs)
    
    if config_merge == 0:
        out_layer = Concatenate()([input2d,attention_outputs])
    elif config_merge == 1:
        out_layer = Add()([input2d,attention_outputs])
    else:
        print("Undefined config_merge: "+str(config_merge))
        sys.exit(1)
    out_layer = LayerNormalization()(out_layer)
    return out_layer


def sequence_attention(kernel_size=3,feature_2D_num=(481,6),use_bias=True,hidden_type='sigmoid', 
                       filters = 64 ,nb_layers = 34,opt = 'Adam', initializer = "he_normal", 
                       loss_function = "binary_crossentropy", weight_p=1.0, weight_n=1.0, 
                       config_1d = 0, config_merge = 0,pe = 0,insert_pos = 'none'):
    # config_1d: 
    #    0 : self_attention
    #    1 : resnet1D
    # config_merge: 
    #    0 : concat style
    #    1 : resnet style   
    # pe 
    #    0 : no positional embedding 
    #    1 : 1d sinusoid positional embedding
    #    2 : 1d and 2d sinusoid positional embedding
    #    3 : 2d sinusoid positional embedding 

    if len(feature_2D_num) < 2:
        print("please check the feature number!")
        sys.exit(1)
    contact_feature_num_2D=feature_2D_num[0]
    contact_feature_num_1D=feature_2D_num[1]
    
    
    contact_input = Input(shape=(None,None,contact_feature_num_2D),name='contact_input2d')
    contact_input1d = Input(shape=(None,None,contact_feature_num_1D),name='contact_input1d')
        
    _handle_dim_ordering()
    ######################### now merge new data to new architecture
    DNCON4_2D_input = contact_input
    DNCON4_2D_conv = DNCON4_2D_input
    DNCON4_2D_conv = InstanceNormalization(axis=-1)(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation('relu')(DNCON4_2D_conv)
    DNCON4_2D_conv = Conv2D(128, 1, padding = 'same')(DNCON4_2D_conv)
    DNCON4_2D_conv = MaxoutAct(DNCON4_2D_conv, filters=4, kernel_size=(1,1), output_dim=64,
                               padding='same', activation = "relu")
    DNCON4_2D_conv = _conv_bn_relu_K(filters=filters, kernel_size=7, strides=1)(DNCON4_2D_conv)
    attention_depth = DNCON4_2D_conv.shape.as_list()[3]
    DNCON4_1D_conv = process_1d_input(contact_input1d,attention_depth,n_head = 4,
                                      config_1d = config_1d)
    if insert_pos == 'head':
        if pe ==2 or pe == 3:
            encode2d_layer = PosEncodingLayer2d(DNCON4_2D_conv, attention_depth)
            encode2d_val = add_layer([DNCON4_2D_conv, encode2d_layer])
        else:
            encode2d_val = DNCON4_2D_conv
        DNCON4_2D_conv = sequence_attention_layer(encode2d_val,DNCON4_1D_conv,
                                                  config_merge = config_merge,idx=0)

    block = DNCON4_2D_conv
    filters = filters
    residual_unit = _bn_relu_conv_K
    transition_dilation_rate = [(1, 1),(3, 3),(5, 5),(7, 7)]
    repetitions=[3, 4, 6, 3]
    dropout = None
    for i, r in enumerate(repetitions):
        transition_dilation_rates = transition_dilation_rate * r
        transition_strides = [(1, 1)] * r  

        block = _residual_block_K(basic_block_K, filters=filters,
                                stage=i, blocks=r,
                                is_first_layer=(i == 0),
                                dropout=dropout,
                                transition_dilation_rates=transition_dilation_rates,
                                transition_strides=transition_strides,
                                residual_unit=residual_unit, use_SE = True)(block)
        if insert_pos == 'middle' and i%2 == 1:
            if pe ==2 or pe == 3:
                encode2d_layer = PosEncodingLayer2d(block, attention_depth)
                encode2d_val = add_layer([block, encode2d_layer])
            else:
                encode2d_val = block           
            block = sequence_attention_layer(encode2d_val,DNCON4_1D_conv, 
                                             config_merge = config_merge,idx=i)

    # for i, r in enumerate(repetitions):
    #     block = _residual_block(basic_block, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
    # Last activation
    block = _bn_relu_K(block)
    DNCON4_2D_conv = block
    if insert_pos == 'last_conv':
        if pe ==2 or pe == 3:
            encode2d_layer = PosEncodingLayer2d(DNCON4_2D_conv, attention_depth)
            encode2d_val = add_layer([DNCON4_2D_conv, encode2d_layer])
        else:
            encode2d_val = DNCON4_2D_conv
        DNCON4_2D_conv = sequence_attention_layer(encode2d_val,DNCON4_1D_conv,
                                                  config_merge = config_merge,idx=0)

    
    DNCON4_2D_conv = _conv_in_sigmoid2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), 
                                    kernel_initializer=initializer)(DNCON4_2D_conv)
    loss = loss_function
    DNCON4_2D_out = DNCON4_2D_conv
    DNCON4_RES = Model(inputs=[contact_input,contact_input1d], outputs=DNCON4_2D_out)
    DNCON4_RES.compile(loss=loss, metrics=['accuracy'], optimizer=opt)
    DNCON4_RES.summary()
    return DNCON4_RES

if  __name__ == '__main__':
    m1 = regional_attention3D(feature_2D_num=441,att_outdim=16,att_config=1,insert_pos='last_conv')
    bbb = m1.to_json()