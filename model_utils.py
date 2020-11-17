import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.initializers import random_normal, Constant
from keras.regularizers import l2
from keras.layers import Lambda, Input, Layer, Dropout, Activation, Add, add
from keras.layers import Dense, TimeDistributed, Conv1D, Conv2D, Conv3D
from keras.layers import Concatenate, BatchNormalization, ZeroPadding1D
from keras.layers import Softmax, Multiply, Permute
from keras.layers import GlobalAveragePooling2D, Reshape, Bidirectional,GRU

def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if int(tf.__version__.split('.')[0])<2:
        if K.image_dim_ordering() == 'tf':
            ROW_AXIS = 1
            COL_AXIS = 2
            CHANNEL_AXIS = 3
        else:
            CHANNEL_AXIS = 1
            ROW_AXIS = 2
            COL_AXIS = 3
    else:
        if K.image_data_format() == 'channels_last':
            ROW_AXIS = 1
            COL_AXIS = 2
            CHANNEL_AXIS = 3
        else:
            CHANNEL_AXIS = 1
            ROW_AXIS = 2
            COL_AXIS = 3        

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

def _bn_relu_K(x, bn_name=None, relu_name=None):
    """Helper to build a BN -> relu block
    """
    # norm = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
    norm = InstanceNormalization(axis=-1, name=bn_name)(x)
    return Activation("relu", name=relu_name)(norm)

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

    def f(x):
        activation = _bn_relu_K(x, bn_name=bn_name, relu_name=relu_name)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,
                      name=conv_name)(activation)
                      # kernel_regularizer=kernel_regularizer,

    return f

def _residual_block_K(block_function, filters, blocks, stage, transition_strides=None,
                      transition_dilation_rates=None, dilation_rates=None, 
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

def basic_block_K(filters, stage, block, transition_strides=(1, 1), dilation_rate=(1, 1),
                  is_first_block_of_first_layer=False, dropout=None, residual_unit=_bn_relu_conv_K, use_SE = False):
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


class MaxoutActLayer(Layer):
    def __init__(self, filters=4, kernel_size=(1,1), output_dim=64, 
                 padding='same', activation = "relu", **kwargs):
        self.output_dim = output_dim
        self.convlayer = Conv2D(filters=filters, kernel_size=kernel_size, 
                                padding=padding)
        self.actlayer = Activation(activation)
        self.maxoutlayer = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))
        super(MaxoutActLayer, self).__init__(**kwargs)
        
    def call(self, x):
        output = None
        for _ in range(self.output_dim):
            conv = self.convlayer(x)
            activa = self.actlayer(conv)
            maxout_out = self.maxoutlayer(activa)
            if output is not None:
                output = Concatenate()([output, maxout_out])
            else:
                output = maxout_out
        return output      
     
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],self.output_dim)
    
class InstanceNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        super(InstanceNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis '+str(self.axis)+' undefined.')
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape, name='gamma', 
                                     initializer=random_normal(1.0, 0.02))
        self.beta = self.add_weight(shape=shape, name='beta', 
                                    initializer='zeros')
        self.built = True

    def call(self, inputs, training=None):
        if int(tf.__version__.split('.')[0])<2:
            mean, var = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
            out_layer = K.batch_normalization(inputs, mean, var, self.beta, 
                                              self.gamma, self.epsilon)
        else:
            mean, var = tf.nn.moments(inputs, axes=[0,1,2], keepdims=True)
            out_layer = K.batch_normalization(inputs, mean, var, self.beta, 
                                              self.gamma,-1,self.epsilon)
        return out_layer
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
def MaxoutAct(x, filters, kernel_size, output_dim, padding='same', activation = "relu"):
    output = None
    for _ in range(output_dim):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(x)
        activa = Activation(activation)(conv)
        maxout_out = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(activa)
        if output is not None:
            output = Concatenate(axis=-1)([output, maxout_out])
        else:
            output = maxout_out
    return output

def squeeze_excite_block(x, ratio=16):
    init = x
    filters = init.shape.as_list()[3]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

def permute_1_0(x):
    s = tf.shape(x)  
    x = tf.reshape(x, [s[1], s[0]])
    return x
        
def Tile1Dto2D(x):
    # x: (n,L,d)
    # output is (n,L,L,d),
    # in all first k dims the columns are rep of columns of k-th axis
    # in all last k dims the rows are rep of rows of k-th axis
    # rep columns first to (n,L,L,2d)
    x = Permute((2,1))(x)
    x = Lambda(lambda x:K.expand_dims(x))(x)
    one_vec = Lambda(lambda x:x[0,0,:,:]*0+1)(x)
    one_vec =  Lambda(lambda x:permute_1_0(x))(one_vec)
    col2d = Lambda(lambda x:K.dot(x[0],x[1]))([x,one_vec])
    col2d = Permute((2,3,1))(col2d)
    row2d = Permute((2,1,3))(col2d)
    out = Concatenate()([col2d,row2d])
    return out

class ScaledDotProductAttention():
	def __init__(self, attn_dropout=0.1):
		self.dropout = Dropout(attn_dropout)
	def __call__(self, q, k, v):
		temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
		attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/temper)([q, k])  
        # shape=(batch, q, k)
		attn = Activation('softmax')(attn)
		attn = self.dropout(attn)
		output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
		return output

class MultiHeadAttention():
    def __init__(self, n_head, d_model, dropout):
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout
        
        self.qs_layer = Dense(n_head*d_k, use_bias=False)
        self.ks_layer = Dense(n_head*d_k, use_bias=False)
        self.vs_layer = Dense(n_head*d_v, use_bias=False)

        self.attention = ScaledDotProductAttention()
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v):
        d_v = self.d_v
        n_head = self.n_head

        qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
        ks = self.ks_layer(k)
        vs = self.vs_layer(v)

        def reshape1(x):
            s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
            x = tf.reshape(x, [s[0], s[1], n_head, s[2]//n_head])
            x = tf.transpose(x, [2, 0, 1, 3])  
            x = tf.reshape(x, [-1, s[1], s[2]//n_head])  # [n_head * batch_size, len_q, d_k]
            return x
        
        qs = Lambda(reshape1)(qs)
        ks = Lambda(reshape1)(ks)
        vs = Lambda(reshape1)(vs)

        head = self.attention(qs, ks, vs)  

        def reshape2(x):
            s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
            x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
            x = tf.transpose(x, [1, 2, 0, 3])
            x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
            return x
        head = Lambda(reshape2)(head)

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        return outputs

def get_stretch_weight(region_size):
    w = np.zeros((region_size,region_size,1,1,region_size*region_size))
    for i in range(region_size):
        for j in range(region_size):
            w[i,j,0,0,region_size*i+j] = 1
    return np.asarray(w)

def ScaledDotProductAttention2D(x, region_size, d_k = 16, att_drop=0.1, 
                                idx = 0, output_score = True, n_head = None):
    temper = int(np.sqrt(d_k))
    stretch_layer = Conv3D(region_size*region_size, 
                            (region_size,region_size,1),padding='same',
                            activation=None, use_bias=False, 
                            bias_constraint=None, trainable=False, 
                            kernel_initializer=Constant(get_stretch_weight(region_size)))
    
    ks = Lambda(lambda x:K.expand_dims(x))(x[1])
    vs = Lambda(lambda x:K.expand_dims(x))(x[2])
    
    ks = stretch_layer(ks)
    vs = stretch_layer(vs)
    
    ks = Permute((4,1,2,3))(ks)
    QK = Multiply()([x[0],ks])
    QK = Lambda(lambda x: K.sum(x,4)/temper)(QK)
    attention_score = Softmax(1)(QK)
    attention_score = Dropout(att_drop)(attention_score)
    
    if output_score:
        s = tf.shape(attention_score)
        attention_score = Lambda(lambda x:tf.reshape(x, [s[0]//n_head, s[1], s[2], s[3], -1]),
                                 name = 'attention_score2D_'+str(idx))(attention_score)
        attention_score = Lambda(lambda x:tf.reshape(x, [-1, s[1], s[2], s[3]]))(attention_score)
    
    vs = Permute((3,4,1,2))(vs)
    QKV = Multiply()([attention_score,vs])
    QKV = Lambda(lambda x: K.sum(x,2))(QKV)
    QKV = Permute((2,3,1))(QKV)
    return QKV

class MultiHeadAttention2D():
    def __init__(self, region_size = 3, n_head = 4, d_model = 64, 
                 dropout = 0.1,idx = 0,output_score = True):
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout
        self.qs_layer = Dense(n_head*d_k, use_bias=False)
        self.ks_layer = Dense(n_head*d_k, use_bias=False)
        self.vs_layer = Dense(n_head*d_v, use_bias=False)
        self.w_o = Dense(d_model)
        self.d_model = d_model
        self.idx = idx
        self.region_size = region_size
        self.output_score = output_score
        
    def __call__(self, x):
        n_head = self.n_head
        qs = self.qs_layer(x[0])  # [batch_size, l,l, n_head*d_k]
        ks = self.ks_layer(x[1])
        vs = self.vs_layer(x[2])

        def reshape1(x):
            s = tf.shape(x)   # [batch_size, l,l, n_head * d_k]
            x = tf.reshape(x, [s[0], s[1], s[2], n_head, s[3]//n_head])
            x = tf.transpose(x, [3, 0, 1, 2, 4])  
            x = tf.reshape(x, [-1, s[1], s[2], s[3]//n_head])  
            # [n_head * batch_size, l, l, d_k]
            return x
        
        qs = Lambda(reshape1)(qs)
        ks = Lambda(reshape1)(ks)
        vs = Lambda(reshape1)(vs)

        head = ScaledDotProductAttention2D([qs, ks, vs], self.region_size, 
                                           self.d_k, self.dropout, self.idx, 
                                           self.output_score, self.n_head)
        
        def reshape2(x):
            s = tf.shape(x)   # [n_head * batch_size, l,l, d_v]
            x = tf.reshape(x, [n_head, -1, s[1], s[2], s[3]]) 
            x = tf.transpose(x, [1, 2, 3, 0, 4])
            x = tf.reshape(x, [-1, s[1], s[2], self.n_head*self.d_v])  
            # [batch_size, l,l, n_head * d_v]
            return x
        
        head = Lambda(reshape2)(head)
        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        return outputs


    
def basic_1d(filters, stage=0, block=0, kernel_size=3, numerical_name=False,
    stride=None):

    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = ZeroPadding1D(
            padding=1, 
            name="padding{}{}_branch2a".format(stage_char, block_char)
        )(x)
        
        y = Conv1D(
            filters,
            kernel_size,
            strides=stride,
            use_bias=False,
            name="res{}{}_branch2a".format(stage_char, block_char)
        )(y)
        
        y = BatchNormalization(
            epsilon=1e-5,
            name="bn{}{}_branch2a".format(stage_char, block_char)
        )(y)
        
        y = Activation(
            "relu",
            name="res{}{}_branch2a_relu".format(stage_char, block_char)
        )(y)

        y = ZeroPadding1D(
            padding=1,
            name="padding{}{}_branch2b".format(stage_char, block_char)
        )(y)
        
        y = Conv1D(
            filters,
            kernel_size,
            use_bias=False,
            name="res{}{}_branch2b".format(stage_char, block_char)
        )(y)
        
        y = BatchNormalization(
            epsilon=1e-5,
            name="bn{}{}_branch2b".format(stage_char, block_char)
        )(y)

        if block == 0:
            shortcut = Conv1D(
                filters,
                1,
                strides=stride,
                use_bias=False,
                name="res{}{}_branch1".format(stage_char, block_char)
            )(x)

            shortcut = BatchNormalization(
                epsilon=1e-5,
                name="bn{}{}_branch1".format(stage_char, block_char)
            )(shortcut)
        else:
            shortcut = x

        y = Add(
            name="res{}{}".format(stage_char, block_char)
        )([y, shortcut])
        
        y = Activation(
            "relu",
            name="res{}{}_relu".format(stage_char, block_char)
        )(y)

        return y

    return f


def bottleneck_1d(filters, stage=0, block=0, kernel_size=3, 
                  numerical_name=False, stride=None, freeze_bn=False):
    if stride is None:
        stride = 1 if block != 0 or stage == 0 else 2

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = Conv1D(
            filters,
            1,
            strides=stride,
            use_bias=False,
            name="res{}{}_branch2a".format(stage_char, block_char)
        )(x)

        y = BatchNormalization(
            epsilon=1e-5,
            name="bn{}{}_branch2a".format(stage_char, block_char)
        )(y)

        y = Activation(
            "relu",
            name="res{}{}_branch2a_relu".format(stage_char, block_char)
        )(y)

        y = ZeroPadding1D(
            padding=1,
            name="padding{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = Conv1D(
            filters,
            kernel_size,
            use_bias=False,
            name="res{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = BatchNormalization(
            epsilon=1e-5,
            name="bn{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = Activation(
            "relu",
            name="res{}{}_branch2b_relu".format(stage_char, block_char)
        )(y)

        y = Conv1D(
            filters * 4,
            1,
            use_bias=False,
            name="res{}{}_branch2c".format(stage_char, block_char)
        )(y)

        y = BatchNormalization(
            epsilon=1e-5,
            name="bn{}{}_branch2c".format(stage_char, block_char)
        )(y)

        if block == 0:
            shortcut = Conv1D(
                filters * 4,
                1,
                strides=stride,
                use_bias=False,
                name="res{}{}_branch1".format(stage_char, block_char)
            )(x)

            shortcut = BatchNormalization(
                epsilon=1e-5,
                name="bn{}{}_branch1".format(stage_char, block_char)
            )(shortcut)
        else:
            shortcut = x

        y = Add(
            name="res{}{}".format(stage_char, block_char)
        )([y, shortcut])

        y = Activation(
            "relu",
            name="res{}{}_relu".format(stage_char, block_char)
        )(y)

        return y

    return f

def ContactTransformerV5(kernel_size=3,feature_2D_num=(441,56),use_bias=True,hidden_type='sigmoid', \
                       filters = 64,nb_layers = 34,opt = 'Adam', initializer = "he_normal", \
                       loss_function = "binary_crossentropy", weight_p=1.0,weight_n=1.0, \
                       region_size = 3,config_1d = 1,att_outdim=0,insert_pos = 'none',L=None):
    
    input2d = Input(shape=(None,None,feature_2D_num[0]), name='input2d')
    input1d = Input(shape=(None,None,feature_2D_num[1]), name='input1d')
    
    x1d = InstanceNormalization()(input1d)
    x1d = Lambda(lambda x: x[:,0,:,1::2])(x1d)
    
    x1d = Conv1D(filters,3,activation='relu',padding='same')(x1d)
    
    if tf.test.is_gpu_available(cuda_only=True) and int(tf.__version__.split('.')[0])<2:
        x1d = Bidirectional(keras.layers.CuDNNGRU(filters//4,return_sequences = True))(x1d)
    else:
        x1d = Bidirectional(GRU(filters//4,return_sequences = True,
                                recurrent_activation='sigmoid',reset_after=True))(x1d)
    x1d = MultiHeadAttention(4,filters,0.1)(x1d,x1d,x1d)
    
    x1d = Tile1Dto2D(x1d)
    
    DNCON4_1D_conv = Conv2D(filters=filters, kernel_size=(3,3), strides=1,
                            padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1.e-4))(x1d)
    DNCON4_1D_conv = InstanceNormalization()(DNCON4_1D_conv)
    DNCON4_1D_conv = Activation("relu")(DNCON4_1D_conv)
    
    _handle_dim_ordering()
    DNCON4_2D_conv = input2d
    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation('relu')(DNCON4_2D_conv)
    DNCON4_2D_conv = Conv2D(128, 1, padding = 'same')(DNCON4_2D_conv)
    DNCON4_2D_conv = MaxoutAct(DNCON4_2D_conv, filters=4, kernel_size=(1,1), 
                               output_dim=64, padding='same', activation = "relu")  
    
    DNCON4_2D_conv = Concatenate()([DNCON4_2D_conv,DNCON4_1D_conv])
    DNCON4_2D_conv = Conv2D(filters=filters, kernel_size=7, strides=1,
                            padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1.e-4))(DNCON4_2D_conv)
    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation("relu")(DNCON4_2D_conv)
    
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
    DNCON4_2D_conv = block
    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation("relu")(DNCON4_2D_conv)

    final_out = Conv2D(filters=1, kernel_size=(1, 1), strides=(1,1),use_bias=True,
                             kernel_initializer='he_normal', padding="same",
                             dilation_rate=(1,1))(DNCON4_2D_conv)
    final_out = Activation("sigmoid")(final_out)
        
    model = Model(inputs=[input2d,input1d], outputs=final_out)
    model.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)

    return model


def ContactTransformerV7(kernel_size=3,feature_2D_num=(441,56),use_bias=True,hidden_type='sigmoid', \
                       filters = 64,nb_layers = 34,opt = 'Adam', initializer = "he_normal", \
                       loss_function = "binary_crossentropy", weight_p=1.0,weight_n=1.0, \
                       region_size = 3,config_1d = 1,att_outdim=0,insert_pos = 'none'):
    
    input2d = Input(shape=(None,None,feature_2D_num[0]), name='input2d')
    input1d = Input(shape=(None,None,feature_2D_num[1]), name='input1d')
    _handle_dim_ordering()
    DNCON4_2D_conv = Concatenate()([input2d,input1d])
    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation('relu')(DNCON4_2D_conv)
    DNCON4_2D_conv = Conv2D(128, 1, padding = 'same')(DNCON4_2D_conv)
    DNCON4_2D_conv = MaxoutAct(DNCON4_2D_conv, filters=4, kernel_size=(1,1), 
                               output_dim=64, padding='same', activation = "relu")  
    DNCON4_2D_conv = Conv2D(filters=filters, kernel_size=7, strides=1,
                            padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1.e-4))(DNCON4_2D_conv)
    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation("relu")(DNCON4_2D_conv)
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
    DNCON4_2D_conv = block
    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation("relu")(DNCON4_2D_conv)
    DNCON4_2D_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1),use_bias=True,
                             kernel_initializer='he_normal', padding="same",
                             dilation_rate=(1,1))(DNCON4_2D_conv)    
    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation("relu")(DNCON4_2D_conv)
    if int(tf.__version__.split('.')[0])<2:
        dm = DNCON4_2D_conv.shape[3].value
    else:
        dm = DNCON4_2D_conv.shape[3]
    DNCON4_2D_att = MultiHeadAttention2D(region_size,d_model=dm)([DNCON4_2D_conv,DNCON4_2D_conv,DNCON4_2D_conv])
    DNCON4_2D_conv = Add()([DNCON4_2D_conv,DNCON4_2D_att])
    final_out = Conv2D(filters=1, kernel_size=(1, 1), strides=(1,1),use_bias=True,
                             kernel_initializer='he_normal', padding="same",
                             dilation_rate=(1,1))(DNCON4_2D_conv)
    final_out = Activation("sigmoid")(final_out)
    model = Model(inputs=[input2d,input1d], outputs=final_out)
    model.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)
    return model 

def baselineModel(kernel_size=3,feature_2D_num=(441,56),use_bias=True,hidden_type='sigmoid', \
                       filters = 64,nb_layers = 34,opt = 'Adam', initializer = "he_normal", \
                       loss_function = "binary_crossentropy", weight_p=1.0,weight_n=1.0, \
                       region_size = 3,config_1d = 1,att_outdim=0,insert_pos = 'none'):
    
    input2d = Input(shape=(None,None,feature_2D_num[0]), name='input2d')
    input1d = Input(shape=(None,None,feature_2D_num[1]), name='input1d')
    _handle_dim_ordering()
    DNCON4_2D_conv = Concatenate()([input2d,input1d])
    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation('relu')(DNCON4_2D_conv)
    DNCON4_2D_conv = Conv2D(128, 1, padding = 'same')(DNCON4_2D_conv)
    DNCON4_2D_conv = MaxoutAct(DNCON4_2D_conv, filters=4, kernel_size=(1,1), 
                               output_dim=64, padding='same', activation = "relu")  
    DNCON4_2D_conv = Conv2D(filters=filters, kernel_size=7, strides=1,
                            padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1.e-4))(DNCON4_2D_conv)
    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation("relu")(DNCON4_2D_conv)
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
    DNCON4_2D_conv = block
    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation("relu")(DNCON4_2D_conv)
    DNCON4_2D_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1),use_bias=True,
                             kernel_initializer='he_normal', padding="same",
                             dilation_rate=(1,1))(DNCON4_2D_conv)    
    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
    DNCON4_2D_conv = Activation("relu")(DNCON4_2D_conv)
#    DNCON4_2D_att = MultiHeadAttention2D(region_size,d_model=DNCON4_2D_conv.shape[3].value)([DNCON4_2D_conv,DNCON4_2D_conv,DNCON4_2D_conv])
#    DNCON4_2D_conv = Add()([DNCON4_2D_conv,DNCON4_2D_att])
    final_out = Conv2D(filters=1, kernel_size=(1, 1), strides=(1,1),use_bias=True,
                             kernel_initializer='he_normal', padding="same",
                             dilation_rate=(1,1))(DNCON4_2D_conv)
    final_out = Activation("sigmoid")(final_out)
    model = Model(inputs=[input2d,input1d], outputs=final_out)
    model.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)
    return model 

def extract_sequence_weights(X,model):
    input_layer = model.get_input_at(0)
    att1d_score = model.get_layer('activation_1').get_output_at(0)
    
    def reshape_att1d(x):
        s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
        x = tf.reshape(x, [1,s[0], s[1], s[2]])
        return x
    
    att_out = Lambda(reshape_att1d)(att1d_score)
    att_viz = Model(input_layer,att_out)
    attention_score_mat = att_viz.predict(X)[0,:,:,:]
    return attention_score_mat

def extract_regional_weights(X,model):
    input_layer = model.get_input_at(0)
    att_out = model.get_layer('softmax_1').get_output_at(0)
    
    def reshape_att2d(x):
        s = tf.shape(x)
        x = tf.reshape(x, [1,s[0], s[1], s[2], s[3]])
        return x
    att_out1 = Lambda(reshape_att2d)(att_out)
    att_viz = Model(input_layer,att_out1)
    
    attention_score_mat = att_viz.predict(X)
    return attention_score_mat[0,:,:,:,:]
