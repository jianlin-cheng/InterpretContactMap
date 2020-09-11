# -*- coding: utf-8 -*-
"""
Variable length transformer without embeddings
"""

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer, Input, Dropout, Lambda, Activation, Add, Dense, Conv1D
from keras.layers import TimeDistributed
from keras.models import Model
from keras.initializers import Ones,Zeros
from keras.layers import BatchNormalization, ZeroPadding1D, MaxPooling1D


class LayerNormalization(Layer):
	def __init__(self, eps=1e-6, **kwargs):
		self.eps = eps
		super(LayerNormalization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
		self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
		super(LayerNormalization, self).build(input_shape)
	def call(self, x):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta
	def compute_output_shape(self, input_shape):
		return input_shape
    
class ScaledDotProductAttention():
	def __init__(self, attn_dropout=0.1):
		self.dropout = Dropout(attn_dropout)
	def __call__(self, q, k, v, mask=None):   # mask_k or mask_qk
		temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
		attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/temper)([q, k])  # shape=(batch, q, k)
		if mask is not None:
			mmask = Lambda(lambda x:(-1e+9)*(1.-K.cast(x, 'float32')))(mask)
			attn = Add()([attn, mmask])
		attn = Activation('softmax')(attn)
		attn = self.dropout(attn)
		output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
		return output, attn

class MultiHeadAttention():
	# mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, dropout):
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout
        
        self.qs_layer = Dense(n_head*d_k, use_bias=False)
        self.ks_layer = Dense(n_head*d_k, use_bias=False)
        self.vs_layer = Dense(n_head*d_v, use_bias=False)

        self.attention = ScaledDotProductAttention()
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
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

        if mask is not None:
            mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
        head, attn = self.attention(qs, ks, vs, mask=mask)  

        def reshape2(x):
            s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
            x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
            x = tf.transpose(x, [1, 2, 0, 3])
            x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
            return x
        head = Lambda(reshape2)(head)

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        return outputs, attn

class PositionwiseFeedForward():
	def __init__(self, d_hid, d_inner_hid, dropout=0.1):
		self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
		self.w_2 = Conv1D(d_hid, 1)
		self.layer_norm = LayerNormalization()
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
		self.norm_layer = LayerNormalization()
	def __call__(self, enc_input, mask=None):
		output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
		output = self.norm_layer(Add()([enc_input, output]))
		output = self.pos_ffn_layer(output)
		return output, slf_attn
    
def GetPosEncodingMatrix(max_len, d_emb):
	pos_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
		if pos != 0 else np.zeros(d_emb) 
			for pos in range(max_len)
			])
	pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
	pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
	return pos_enc

class SelfAttention():
	def __init__(self, d_model, d_inner_hid, n_head, layers=6, dropout=0.1):
		self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]
	def __call__(self, src_emb, src_seq, return_att=False, active_layers=999):
		if return_att: atts = []
#		mask = Lambda(lambda x:K.cast(K.greater(x, 0), 'float32'))(src_seq)
		mask = None
		x = src_emb		
		for enc_layer in self.layers[:active_layers]:
			x, att = enc_layer(x, mask)
			if return_att: atts.append(att)
		return (x, atts) if return_att else x

class DecoderLayer():
	def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
		self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
		self.enc_att_layer  = MultiHeadAttention(n_head, d_model, dropout=dropout)
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
		self.norm_layer1 = LayerNormalization()
		self.norm_layer2 = LayerNormalization()
	def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None, dec_last_state=None):
		if dec_last_state is None: dec_last_state = dec_input
		output, slf_attn = self.self_att_layer(dec_input, dec_last_state, dec_last_state, mask=self_mask)
		x = self.norm_layer1(Add()([dec_input, output]))
		output, enc_attn = self.enc_att_layer(x, enc_output, enc_output, mask=enc_mask)
		x = self.norm_layer2(Add()([x, output]))
		output = self.pos_ffn_layer(x)
		return output, slf_attn, enc_attn

def GetPadMask(q, k):
	'''
	shape: [B, Q, K]
	'''
	ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
	mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
	mask = K.batch_dot(ones, mask, axes=[2,1])
	return mask

def GetSubMask(s):
	'''
	shape: [B, Q, K], lower triangle because the i-th row should have i 1s.
	'''
	len_s = tf.shape(s)[1]
	bs = tf.shape(s)[:1]
	mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
	return mask
    
class Decoder():
	def __init__(self, d_model, d_inner_hid, n_head, layers=6, dropout=0.1):
		self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]
	def __call__(self, tgt_emb, tgt_seq, src_seq, enc_output, return_att=False, active_layers=999):
		x = tgt_emb
		self_pad_mask = Lambda(lambda x:GetPadMask(x, x))(tgt_seq)
		self_sub_mask = Lambda(GetSubMask)(tgt_seq)
		self_mask = Lambda(lambda x:K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
		enc_mask = Lambda(lambda x:GetPadMask(x[0], x[1]))([tgt_seq, src_seq])
		if return_att: self_atts, enc_atts = [], []
		for dec_layer in self.layers[:active_layers]:
			x, self_att, enc_att = dec_layer(x, enc_output, self_mask, enc_mask)
			if return_att: 
				self_atts.append(self_att)
				enc_atts.append(enc_att)
		return (x, self_atts, enc_atts) if return_att else x

def add_layer(x):
    return Lambda(lambda x:x[0]+x[1], output_shape=lambda x:x[0])

class Transformer:
	def __init__(self, d_model=256, d_inner_hid=512, n_head=4, layers=2, dropout=0.1):
		self.d_model = d_model
		self.layers = layers
		d_emb = d_model
		d_k = d_v = d_model // n_head
		assert d_k * n_head == d_model and d_v == d_k

		self.emb_dropout = Dropout(dropout)
		self.encoder = SelfAttention(d_model, d_inner_hid, n_head, layers, dropout)
		self.decoder = Decoder(d_model, d_inner_hid, n_head, layers, dropout)
		self.target_layer = TimeDistributed(Dense(d_emb, use_bias=False))

	def compile(self, optimizer='adam', active_layers=999):
		src_seq_input = Input(shape=(None,self.d_model))
		tgt_seq_input = Input(shape=(None,self.d_model))

		src_seq = src_seq_input
		tgt_seq  = Lambda(lambda x:x[:,:-1])(tgt_seq_input) 

		src_emb = src_seq
		tgt_emb = tgt_seq

		if self.pos_emb: 
			src_emb = add_layer([src_emb, self.pos_emb(src_seq)])
			tgt_emb = add_layer([tgt_emb, self.pos_emb(tgt_seq)])
		src_emb = self.emb_dropout(src_emb)

		enc_output = self.encoder(src_emb, src_seq, active_layers=active_layers)
		dec_output = self.decoder(tgt_emb, tgt_seq, src_seq, enc_output, active_layers=active_layers)	
		final_output = self.target_layer(dec_output)

		self.model = Model([src_seq_input, tgt_seq_input], final_output)
		self.model.compile(optimizer, 'mse')


def basic_1d(
    filters,
    stage=0,
    block=0,
    kernel_size=3,
    numerical_name=False,
    stride=None,
):
    """
    A one-dimensional basic block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    Usage:
        >>> import keras_resnet.blocks
        >>> keras_resnet.blocks.basic_1d(64)
    """
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


def bottleneck_1d(
    filters,
    stage=0,
    block=0,
    kernel_size=3,
    numerical_name=False,
    stride=None,
    freeze_bn=False
):
    """
    A one-dimensional bottleneck block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    Usage:
        >>> import keras_resnet.blocks
        >>> keras_resnet.blocks.bottleneck_1d(64)
    """
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
    """
    Constructs a `keras.models.Model` object using the given block count.
    :param inputs: input tensor (e.g. an instance of `Input`)
    :param blocks: the network’s residual architecture
    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_1d`)
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :param numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
        >>> import keras_resnet.blocks
        >>> import keras_resnet.models
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = Input(shape)
        >>> blocks = [2, 2, 2, 2]
        >>> block = keras_resnet.blocks.basic_1d

    """
    def __init__(self,blocks,block,features,numerical_names=None):

        if numerical_names is None:
            numerical_names = [True] * len(blocks)
            
        self.features = features            
        self.blocks = blocks
        self.block = block
        self.numerical_names = numerical_names


    def __call__(self,inputs):
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

            self.features *= 2

        return x
 
    
if __name__ == '__main__':
    from keras.layers import GlobalAveragePooling1D
    from keras.callbacks import TensorBoard
    
    # resnet
    x = np.random.random((100,20,8))
    y = np.sum(x,axis = -1)
    y = np.sum(y,axis = -1)
    
    input_layer = Input((None,8))
    blocks = [2, 2, 2, 2]
    block = basic_1d
    output = ResNet1D(blocks,block,8)(input_layer)
    output = GlobalAveragePooling1D()(output)
    output = Dense(1)(output)
    model = Model(input_layer,output)
    model.compile('Adam','mse')
    model.summary()
    tensorboard_callback = TensorBoard(log_dir='/home/chen/tf_logs/resnet/')
    history = model.fit(x,y,epochs=5,callbacks=[tensorboard_callback])
    
    
    # self attention
    x = np.random.random((100,20,64))
    y = x*5+np.random.random((100,20,64))
    
    input1d = Input(shape=(None,64))
    l0 = LayerNormalization()(input1d)
    l1 = SelfAttention(64,64,4)(l0,l0)
    model = Model(input1d,l1)
    model.compile('Adam','mse')
    tensorboard_callback = TensorBoard(log_dir='/home/chen/tf_logs/attention/')
    history = model.fit(x,y,epochs = 5,callbacks=[tensorboard_callback])
