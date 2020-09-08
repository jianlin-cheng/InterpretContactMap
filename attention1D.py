from keras.layers import Input, Dense, Permute, Lambda, Softmax,Embedding, Conv3D, Dropout, Multiply, Layer, Conv2D
from keras.models import Model
from keras import backend as K
from keras.initializers import Constant

import tensorflow as tf
import numpy as np

n = 1
L = 200
depth =64
dk = 8
dv = 1
region_size = 3

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
                                 name = 'attention_score_'+str(idx))(attention_score)
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




x = Input((L,L,depth))
out = MultiHeadAttention2D(region_size=region_size)([x,x,x])
out1 = Conv2D(1,(3,3))(out)
model = Model(x,out1)
model.compile('Adam','mse')


X = np.random.random((10,L,L,depth))
y=model.predict(X)


m1 = Model(x,model.get_layer('attention_score_0').get_output_at(0))
attention_score_0 = m1.predict(X[[0],:,:,:])



kmax = 0 #0 for np kmaxpooling

x = Input((L,depth))
#PosEncodingLayer

Q = Dense(dk,use_bias=False)(x)
if kmax > 0:
    Qt = Permute((2,1))(Q)
    Qt_kmax = Lambda(lambda x:tf.nn.top_k(x,kmax)[0])(Qt)
    Q = Permute((2,1))(Qt_kmax)

K = Dense(dk,use_bias=False)(x)
V = Dense(dv,use_bias=False)(x)
Kt = Permute((2,1))(K)
QKt = Lambda(lambda x:tf.matmul(x[0],x[1])/np.sqrt(dk))([Q,Kt])
attention_score = Softmax(2,name='attention_score')(QKt)
attention_output = Lambda(lambda x:tf.matmul(x[0],x[1]))([attention_score,V])

model = Model(x,attention_output)
model.compile('Adam','mse')

X = np.random.random((n,L,depth))
Y =np.sum(X*10,2,keepdims=True) + np.random.random((n,L,dv))
history = model.fit(X,Y,epochs = 150)

#viz
from matplotlib import pyplot as plt
def heatmap2d(arr: np.ndarray,fname = None):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
        
att_viz = Model(x,model.get_layer('attention_score').get_output_at(0))
attention_score_mat = att_viz.predict(X[[0],:,:])
heatmap2d(attention_score_mat[0,:,:])