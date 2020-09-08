import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.initializers import random_normal, Constant
from keras.regularizers import l2
from keras.layers import Lambda, Input, Layer, Dropout, Activation, Add
from keras.layers import Dense, TimeDistributed, Conv1D, Conv2D, Conv3D
from keras.layers import Concatenate, BatchNormalization, ZeroPadding1D
from keras.layers import MaxPooling1D, Softmax, Multiply, Embedding, Permute
from keras.layers import GlobalAveragePooling2D, Reshape, CuDNNGRU, Bidirectional

from Model_construct import _residual_block_K, _bn_relu_conv_K, basic_block_K, _handle_dim_ordering

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
        mean, var = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
        return K.batch_normalization(inputs, mean, var, self.beta, self.gamma,
                                     self.epsilon)
        
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

################ 1D attention helper functions#################
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

class PositionwiseFeedForward():
	def __init__(self, d_hid, d_inner_hid, dropout=0.1):
		self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
		self.w_2 = Conv1D(d_hid, 1)
		self.layer_norm = InstanceNormalization()
		self.dropout = Dropout(dropout)
	def __call__(self, x):
		output = self.w_1(x) 
		output = self.w_2(output)
		output = self.dropout(output)
		output = Add()([output, x])
		return self.layer_norm(output)

class EncoderLayer():
	def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
		self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
		self.norm_layer = InstanceNormalization()
	def __call__(self, enc_input):
		output = self.self_att_layer(enc_input, enc_input, enc_input)
		output = self.norm_layer(Add()([enc_input, output]))
		output = self.pos_ffn_layer(output)
		return output
    
def GetPosEncodingMatrix(max_len, d_emb):
	pos_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
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
    
class SelfAttention():
	def __init__(self, d_model, d_inner_hid, n_head, layers=3, dropout=0.1):
		self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]
	def __call__(self, x, active_layers=999):
		for enc_layer in self.layers[:active_layers]:
			x = enc_layer(x)
		return x    

################ 2D attention helper functions#################
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

class PositionwiseFeedForward2D():
	def __init__(self, d_hid, d_inner_hid, dropout=0.1):
		self.w_1 = Conv2D(d_inner_hid, (1,1), activation='relu')
		self.w_2 = Conv2D(d_hid, (1,1))
		self.layer_norm = InstanceNormalization()
		self.dropout = Dropout(dropout)
	def __call__(self, x):
		output = self.w_1(x) 
		output = self.w_2(output)
		output = self.dropout(output)
		output = Add()([output, x])
		return self.layer_norm(output)

class EncoderLayer2D():
    # self attention + add&norm + ffn + add&norm
	def __init__(self, stretch_layer3D, region_size, d_model, d_inner_hid, 
              n_head, dropout=0.1):
		self.self_att_layer = MultiHeadAttention2D(region_size, n_head, 
                                             d_model, dropout=dropout)
		self.pos_ffn_layer = PositionwiseFeedForward2D(d_model, d_inner_hid, 
                                                  dropout=dropout)
		self.norm_layer = InstanceNormalization()
	def __call__(self, x):
		output = self.self_att_layer(x, x, x)
		output = self.norm_layer(Add()([x, output]))
		output = self.pos_ffn_layer(output)
		return output

class SelfAttention2D():
	def __init__(self, region_size, d_model, d_inner_hid, n_head, layers=3, dropout=0.1):
		stretch_layer3D = Conv3D(region_size*region_size, (region_size,region_size,1),
                         padding='same',activation=None, use_bias=False, 
                         bias_constraint=None, trainable=False, name = 'strech_layer')
		self.layers = [EncoderLayer2D(stretch_layer3D, region_size, d_model,
                                d_inner_hid, n_head, dropout) for _ in range(layers)]
	def __call__(self, x, active_layers=999):
		for enc_layer in self.layers[:active_layers]:
			x = enc_layer(x)
		return x


################ resnet helper functions#################    
    
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


class ResNet1D(Layer):
    def __init__(self, blocks, block, features ,numerical_names=None,
                 **kwargs):

        if numerical_names is None:
            numerical_names = [True] * len(blocks)
            
        self.features = features            
        self.blocks = blocks
        self.block = block
        self.numerical_names = numerical_names
        super(ResNet1D, self).__init__(**kwargs)


    def call(self,inputs):
        x = ZeroPadding1D(padding=3, name="padding_conv1")(inputs)
        x = Conv1D(8, 3, strides=1, use_bias=False, name="conv1")(x)
        x = BatchNormalization(epsilon=1e-5, name="bn_conv1")(x)
        x = Activation("relu", name="conv1_relu")(x)
        x = MaxPooling1D(3, strides=1, padding="same", name="pool1")(x)

        for stage_id, iterations in enumerate(self.blocks):
            for block_id in range(iterations):
                x = self.block(
                    self.features,
                    stage_id,
                    block_id,
                    numerical_name=(block_id > 0 and self.numerical_names[stage_id]),
                )(x)

        return x
        
def Encoder1D(x,dim,n_head=4,config_1d = 0,pe = 0,layers=6):
    """
    preprocess layer:
        slice the first row
        slice odd index of last axis only
        normalization
        make it the same depth as 2d input with conv1d
    """
    x1d = Lambda(lambda x: x[:,0,:,1::2])(x)
    x1d = InstanceNormalization()(x1d)
    x1d = Conv1D(filters = dim,kernel_size = 3)(x1d)
    x1d = Activation('relu')(x1d)
    if config_1d == 0:
         # self attention
        if pe == 1 or pe == 2:
            pos_emb = PosEncodingLayer(1000, dim)
            x1d = Lambda(lambda x:x[0]+x[1], output_shape=lambda x:x[0])([x1d, pos_emb(x1d)])
        encoder = SelfAttention(dim, dim, n_head, layers, dropout=0.1)
        x_out = encoder(x1d)         
    else:
        blocks = [2, 2, 2]
        x_out = ResNet1D(blocks,basic_1d,dim)(x1d)
        x_out = Conv1D(dim,3,activation='relu')(x_out)
        x_out = InstanceNormalization()(x_out)
    return x_out

    
def sequence_attention_layer(input2d,input1d, config_merge = 0,n_head = 4, att_dropout=0.1,idx = 0):
    # input2d: (batch_size,l,l,depth)
    # input1d: (batch_size,l,depth)
    # config_merge: 
    #    0 : concat style
    #    1 : resnet style  
    attention_depth = input2d.shape.as_list()[3]
    n_head = min([n_head,attention_depth])
    
    d_q = d_k = d_v = attention_depth//n_head
     
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
        x = tf.reshape(x, [-1, s[1], s[2], s[3]//n_head])  
        # [n_head * batch_size, len_v, len_v, d_v]
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
    else:
        out_layer = Add()([input2d,attention_outputs])
    out_layer = InstanceNormalization()(out_layer)
    return out_layer
    
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



def Decoder2D(DNCON4_2D_tensor,DNCON4_1D_tensor,filters,pe,config_merge,
              region_size=3,layers = 3,n_head = 4):
    if pe ==2 or pe == 3:
        encode2d_layer = PosEncodingLayer2d(DNCON4_2D_tensor, filters)
        encode2d_val = add_layer([DNCON4_2D_tensor, encode2d_layer])
    else:
        encode2d_val = DNCON4_2D_tensor
    strech_layer_decoder = Conv3D(region_size*region_size, (region_size,region_size,1),
                             padding='same',activation=None, use_bias=False, 
                             bias_constraint=None, trainable=False, name = 'strech_layer_decoder')
    for i in range(layers):
        encode2d_val = EncoderLayer2D(strech_layer_decoder,region_size,filters,
                                      filters,n_head = n_head)(encode2d_val)
        encode2d_val = sequence_attention_layer(encode2d_val,DNCON4_1D_tensor, 
                                                config_merge,n_head = n_head, 
                                                att_dropout=0.1,idx = i)
        encode2d_val = PositionwiseFeedForward2D(filters,filters)(encode2d_val)
    return encode2d_val
        
def get_stretch_weight3D(region_size):
    w = np.zeros((region_size,region_size,1,1,region_size*region_size))
    for i in range(region_size):
        for j in range(region_size):
            w[i,j,0,0,region_size*i+j] = 1
    return np.asarray(w)

#def ContactTransformer(kernel_size=3,feature_2D_num=(441,56),use_bias=True,
#                       hidden_type='sigmoid',filters=64,nb_layers=34,opt='Adam',
#                       initializer = "he_normal", loss_function = "binary_crossentropy", 
#                       weight_p=1.0,weight_n=1.0, config_1d = 0, 
#                       config_merge = 1,pe = 2, insert_pos = 'none'):
#    _handle_dim_ordering()
#    region_size = 3
#    att_layers = 2
#    input2d = Input(shape=(None,None,feature_2D_num[0]), name='input2d')
#    input1d = Input(shape=(None,None,feature_2D_num[1]), name='input1d')
# 
#    DNCON4_2D_conv = InstanceNormalization(axis=-1)(input2d)
#    DNCON4_2D_conv = Activation('relu')(DNCON4_2D_conv)
#    DNCON4_2D_conv = Conv2D(128, 1, padding = 'same')(DNCON4_2D_conv)
#    DNCON4_2D_conv = MaxoutActLayer()(DNCON4_2D_conv)  
#    DNCON4_2D_conv = Conv2D(filters=filters, kernel_size=7, strides=1,
#                            padding='same', kernel_initializer='he_normal',
#                            kernel_regularizer=l2(1.e-4))(DNCON4_2D_conv)
#    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
#    DNCON4_2D_conv = Activation("relu")(DNCON4_2D_conv)
# 
#    block = DNCON4_2D_conv
#    filters = filters
#    residual_unit = _bn_relu_conv_K
#    transition_dilation_rate = [(1, 1),(3, 3),(5, 5),(7, 7)]
#    repetitions=[3, 4, 6, 3]
#    dropout = None
#    for i, r in enumerate(repetitions):
#        transition_dilation_rates = transition_dilation_rate * r
#        transition_strides = [(1, 1)] * r  
# 
#        block = _residual_block_K(basic_block_K, filters=filters,
#                                stage=i, blocks=r,
#                                is_first_layer=(i == 0),
#                                dropout=dropout,
#                                transition_dilation_rates=transition_dilation_rates,
#                                transition_strides=transition_strides,
#                                residual_unit=residual_unit, use_SE = True)(block)
# 
# 
#    DNCON4_2D_conv = block
#    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
#    DNCON4_2D_conv = Activation("relu")(DNCON4_2D_conv)
#     
#    DNCON4_2D_tensor = Conv2D(32,(1,1),padding='same',
#                              kernel_initializer = "he_normal")(DNCON4_2D_conv)
#    dim = DNCON4_2D_tensor.shape.as_list()[3]
#    DNCON4_1D_tensor = Encoder1D(input1d,dim,n_head = 4, 
#                                 config_1d = config_1d, pe = pe,layers = att_layers)
#    DNCON4_2D_out = Decoder2D(DNCON4_2D_tensor,DNCON4_1D_tensor,dim,pe,
#                              config_merge,region_size=region_size,layers = att_layers)
#     
#    final_out = Conv2D(1,(1,1),padding='same',kernel_initializer = "he_normal")(DNCON4_2D_out)
#    final_out = InstanceNormalization()(final_out)
#    final_out = Activation("sigmoid")(final_out)
# 
#    model = Model(inputs=[input2d,input1d], outputs=final_out)
#    model.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)
#    stretch_w = get_stretch_weight3D(region_size)
##    model.get_layer('strech_layer').set_weights([stretch_w])
#    model.get_layer('strech_layer_decoder').set_weights([stretch_w])
#    return model

def ContactTransformerV5(kernel_size=3,feature_2D_num=(441,56),use_bias=True,hidden_type='sigmoid', \
                       filters = 64,nb_layers = 34,opt = 'Adam', initializer = "he_normal", \
                       loss_function = "binary_crossentropy", weight_p=1.0,weight_n=1.0, \
                       region_size = 3,config_1d = 1,att_outdim=0,insert_pos = 'none',L=None):
    
    input2d = Input(shape=(None,None,feature_2D_num[0]), name='input2d')
    input1d = Input(shape=(None,None,feature_2D_num[1]), name='input1d')
    
    x1d = InstanceNormalization()(input1d)
    x1d = Lambda(lambda x: x[:,0,:,1::2])(x1d)
    
    x1d = Conv1D(filters,3,activation='relu',padding='same')(x1d)
    x1d = Bidirectional(CuDNNGRU(filters//4,return_sequences = True))(x1d)
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

def ContactTransformerV6(kernel_size=3,feature_2D_num=(441,56),use_bias=True,hidden_type='sigmoid', \
                       filters = 64,nb_layers = 34,opt = 'Adam', initializer = "he_normal", \
                       loss_function = "binary_crossentropy", weight_p=1.0,weight_n=1.0, \
                       region_size = 3,config_1d = 1,att_outdim=0,insert_pos = 'none'):
    
    input2d = Input(shape=(None,None,feature_2D_num[0]), name='input2d')
    input1d = Input(shape=(None,None,feature_2D_num[1]), name='input1d')
    
    x1d = InstanceNormalization()(input1d)
    x1d = Lambda(lambda x: x[:,0,:,1::2])(x1d)
    x1d = Conv1D(filters,3,activation='relu',padding='same')(x1d)
    x1d = Bidirectional(CuDNNGRU(filters//4,return_sequences = True))(x1d)
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
    
    DNCON4_2D_att = MultiHeadAttention2D(region_size,d_model=DNCON4_2D_conv.shape[3].value)([DNCON4_2D_conv,DNCON4_2D_conv,DNCON4_2D_conv])
    DNCON4_2D_conv = Add()([DNCON4_2D_conv,DNCON4_2D_att])
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
    DNCON4_2D_att = MultiHeadAttention2D(region_size,d_model=DNCON4_2D_conv.shape[3].value)([DNCON4_2D_conv,DNCON4_2D_conv,DNCON4_2D_conv])
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

from keras.layers import Subtract
def baselineModelSym(kernel_size=3,feature_2D_num=(441,56),use_bias=True,hidden_type='sigmoid', \
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
    final_out = Conv2D(filters=1, kernel_size=(1, 1), strides=(1,1),use_bias=True,
                             kernel_initializer='he_normal', padding="same",
                             dilation_rate=(1,1))(DNCON4_2D_conv)
    
    final_out_t = Permute((2,1,3))(final_out)
    f1 = Subtract()([final_out,final_out_t])
    out= Lambda(lambda x: x**2)(f1)
    sym_loss_out = GlobalAveragePooling2D(name = 'mse_out')(out)
    final_out = Activation("sigmoid",name = 'map_out')(final_out)
    model = Model(inputs=[input2d,input1d], outputs=[final_out,sym_loss_out])
    def customLoss(yTrue,yPred):
        return yPred*region_size
    losses = {
	"map_out": loss_function,
	"mse_out": customLoss
    }
    model.compile(loss=losses, optimizer=opt)
    return model 



def ContactTransformerV8(kernel_size=3,feature_2D_num=(441,56),use_bias=True,hidden_type='sigmoid', \
                       filters = 64,nb_layers = 34,opt = 'Adam', initializer = "he_normal", \
                       loss_function = "binary_crossentropy", weight_p=1.0,weight_n=1.0, \
                       region_size = 3,config_1d = 1,att_outdim=0,insert_pos = 'none',L=None):
    
    input2d = Input(shape=(None,None,feature_2D_num[0]), name='input2d')
    input1d = Input(shape=(None,None,feature_2D_num[1]), name='input1d')
    
    x1d = InstanceNormalization()(input1d)
    x1d = Lambda(lambda x: x[:,0,:,1::2])(x1d)
    
    x1dr = Conv1D(32,3,padding='same')(x1d)
    x1dr = InstanceNormalization()(x1dr)
    x1dr = Activation('relu')(x1dr)
    
    x1dr = Bidirectional(CuDNNGRU(filters//4,return_sequences = True))(x1dr)
    x1dr = MultiHeadAttention(4,filters,0.1)(x1dr,x1dr,x1dr)
    
    x1d = Add()([x1d,x1dr])
    x1d = InstanceNormalization()(x1d)
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

def ContactTransformerV9(kernel_size=3,feature_2D_num=(441,56),use_bias=True,hidden_type='sigmoid', \
                       filters = 64,nb_layers = 34,opt = 'Adam', initializer = "he_normal", \
                       loss_function = "binary_crossentropy", weight_p=1.0,weight_n=1.0, \
                       region_size = 3,config_1d = 1,att_outdim=0,insert_pos = 'none',L=None):
    
    input2d = Input(shape=(None,None,feature_2D_num[0]), name='input2d')
    input1d = Input(shape=(None,None,feature_2D_num[1]), name='input1d')
    
    x1d = InstanceNormalization()(input1d)
    x1d = Lambda(lambda x: x[:,0,:,1::2])(x1d)
    
    x1dr = Conv1D(32,3,padding='same')(x1d)
    x1dr = InstanceNormalization()(x1dr)
    x1dr = Activation('relu')(x1dr)
    
    x1dr = Bidirectional(CuDNNGRU(filters//4,return_sequences = True))(x1dr)
    x1dr = MultiHeadAttention(4,filters,0.1)(x1dr,x1dr,x1dr)
    
    x1d = Add()([x1d,x1dr])
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
    DNCON4_2D_att = MultiHeadAttention2D(region_size,d_model=DNCON4_2D_conv.shape[3].value)([DNCON4_2D_conv,DNCON4_2D_conv,DNCON4_2D_conv])
    DNCON4_2D_conv = Add()([DNCON4_2D_conv,DNCON4_2D_att])
    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
    final_out = Conv2D(filters=1, kernel_size=(1, 1), strides=(1,1),use_bias=True,
                             kernel_initializer='he_normal', padding="same",
                             dilation_rate=(1,1))(DNCON4_2D_conv)
    final_out = Activation("sigmoid")(final_out)
        
    model = Model(inputs=[input2d,input1d], outputs=final_out)
    model.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)

    return model

def ContactTransformerV10(kernel_size=3,feature_2D_num=(441,56),use_bias=True,hidden_type='sigmoid', \
                       filters = 64,nb_layers = 34,opt = 'Adam', initializer = "he_normal", \
                       loss_function = "binary_crossentropy", weight_p=1.0,weight_n=1.0, \
                       region_size = 3,config_1d = 1,att_outdim=0,insert_pos = 'none',L=None):
    
    input2d = Input(shape=(None,None,feature_2D_num[0]), name='input2d')
    input1d = Input(shape=(None,None,feature_2D_num[1]), name='input1d')
    
    x1d = InstanceNormalization()(input1d)
    x1d = Lambda(lambda x: x[:,0,:,1::2])(x1d)
    
    x1d = Conv1D(filters,3,padding='same')(x1d)
    x1d = InstanceNormalization()(x1d)
    x1d = Activation('relu')(x1d)
    
    x1dr = Bidirectional(CuDNNGRU(filters//4,return_sequences = True))(x1d)
    x1dr = MultiHeadAttention(4,filters,0.1)(x1dr,x1dr,x1dr)
    
    x1d = Add()([x1d,x1dr])
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
#    DNCON4_2D_att = MultiHeadAttention2D(region_size,d_model=DNCON4_2D_conv.shape[3].value)([DNCON4_2D_conv,DNCON4_2D_conv,DNCON4_2D_conv])
#    DNCON4_2D_conv = Add()([DNCON4_2D_conv,DNCON4_2D_att])
#    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
    final_out = Conv2D(filters=1, kernel_size=(1, 1), strides=(1,1),use_bias=True,
                             kernel_initializer='he_normal', padding="same",
                             dilation_rate=(1,1))(DNCON4_2D_conv)
    final_out = Activation("sigmoid")(final_out)
        
    model = Model(inputs=[input2d,input1d], outputs=final_out)
    model.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)

    return model

def ContactTransformerV11(kernel_size=3,feature_2D_num=(441,56),use_bias=True,hidden_type='sigmoid', \
                       filters = 64,nb_layers = 34,opt = 'Adam', initializer = "he_normal", \
                       loss_function = "binary_crossentropy", weight_p=1.0,weight_n=1.0, \
                       region_size = 3,config_1d = 1,att_outdim=0,insert_pos = 'none',L=None):
    
    input2d = Input(shape=(None,None,feature_2D_num[0]), name='input2d')
    input1d = Input(shape=(None,None,feature_2D_num[1]), name='input1d')
    
    x1d = InstanceNormalization()(input1d)
    x1d = Lambda(lambda x: x[:,0,:,1::2])(x1d)
    
    x1d = Conv1D(filters,3,padding='same')(x1d)
    x1d = InstanceNormalization()(x1d)
    x1d = Activation('relu')(x1d)
    
#    x1dr = Bidirectional(CuDNNGRU(filters//4,return_sequences = True))(x1d)
    x1dr = MultiHeadAttention(4,filters,0.1)(x1d,x1d,x1d)
    
    x1d = Add()([x1d,x1dr])
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
#    DNCON4_2D_att = MultiHeadAttention2D(region_size,d_model=DNCON4_2D_conv.shape[3].value)([DNCON4_2D_conv,DNCON4_2D_conv,DNCON4_2D_conv])
#    DNCON4_2D_conv = Add()([DNCON4_2D_conv,DNCON4_2D_att])
#    DNCON4_2D_conv = InstanceNormalization()(DNCON4_2D_conv)
    final_out = Conv2D(filters=1, kernel_size=(1, 1), strides=(1,1),use_bias=True,
                             kernel_initializer='he_normal', padding="same",
                             dilation_rate=(1,1))(DNCON4_2D_conv)
    final_out = Activation("sigmoid")(final_out)
        
    model = Model(inputs=[input2d,input1d], outputs=final_out)
    model.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)

    return model


if  __name__ == '__main__':
    x2d = np.random.random((20,10,10,441))
    x1d = np.random.random((20,10,10,56))
    y = np.random.random((20,10,10,1))
    y_sym = np.zeros((20,1))
    
#    m1 = ContactTransformerV9(region_size=5)
#    m1.summary()
#    history = m1.fit([x2d,x1d],[y,y_sym],epochs = 15)
#    y = m1.predict([x2d,x1d])

#    m2 = ContactTransformerLite(config_1d=1)
#    m2.summary()
#    history = m2.fit([x2d,x1d],y,epochs = 15)
    
#    m2 = ContactTransformerV4(feature_2D_num=(441,56),att_outdim=16,att_config=1,insert_pos='last_conv')
#    m2.summary()
##    aaa=m2.to_json()
##    bbb=m2.to_json()
#    history = m2.fit([x2d,x1d],y,epochs = 15)    
#    y = m2.predict([x2d,x1d])