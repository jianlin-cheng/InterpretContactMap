import tensorflow as tf
import keras.backend as K

from keras.layers import Dense, Lambda, Concatenate, Activation
from keras.layers import GlobalAveragePooling2D, Permute, Softmax, Multiply

def _kmaxpooling2D(l,depth,kmax):
    x4d = Permute((3,1,2))(l)
    shape = tf.shape( x4d )
    x3d = tf.reshape( x4d, [shape[0],depth, shape[2] * shape[3] ] )
    kmax_layer = tf.nn.top_k(x3d, k=kmax, sorted=True)[0]
    return kmax_layer

def _repeat_axis4(a,repeat_time):
    return K.repeat_elements(a,repeat_time,4)

def ksum_axis3(l):
    return K.sum(l,3)

def squeeze_dim2(l):
    return K.squeeze(l,2)

def sum_weighted_imputs(l):
    return K.sum(l,3,True)

def stack_layers(l_list):
    return K.stack(l_list,axis=-1)

def get_attention_input(input_layer,kmax):
    contact_feature_num_2D = input_layer.shape.as_list()[3]
    def kmaxpooling2D(l):
        return _kmaxpooling2D(l,contact_feature_num_2D,kmax)
    kmax_layer1 = Lambda(kmaxpooling2D,name='Kmaxpooling2D')(input_layer)
    glo_ave_layer = GlobalAveragePooling2D()(input_layer)
    glo_ave_layer1 = Lambda(K.expand_dims,name='expand_dim_1')(glo_ave_layer)
    maxave_layer = Concatenate(axis=-1)([glo_ave_layer1,kmax_layer1])
    return maxave_layer

def tf_reshape_layer(l):
    shape_maxave = tf.shape( l )
    shape_maxave2 = l.shape.as_list()
    return tf.reshape(l,[shape_maxave[0],shape_maxave2[1], \
                                          shape_maxave2[2] * shape_maxave2[3] ] )
    
def GetPosEncodingMatrix(max_len, d_emb):
	pos_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
		if pos != 0 else np.zeros(d_emb) 
			for pos in range(max_len)
			], dtype=np.float32)
	pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
	pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
	return pos_enc


# channel-wise attention, weighted sum version
# the weights are learnt from k-max values in each 2D feature.
# NO dense models to capture any hard information on the index of features
# NEED fusion with positional embeddings!
def att_channel_config0(input_layer,kmax=7,att_outdim=32):
    maxave_layer = get_attention_input(input_layer,kmax)
    maxave_layer2 = Activation('tanh')(maxave_layer)
    score1 = Dense(att_outdim, use_bias=False, name='attention_score_vec')(maxave_layer2)
    attweight_layer = Softmax(1,name='attention_weights')(score1)
    weighted_sum_list = []
    for i in range(att_outdim):
        att_w = Lambda(lambda x: x[:, :, i], name='attention_weight_'+str(i))(attweight_layer)
        weighted_features = Multiply()([input_layer,att_w])
        weighted_sum = Lambda(sum_weighted_imputs)(weighted_features)
        weighted_sum_list.append(weighted_sum)
    att_out_all = Concatenate(axis=3)(weighted_sum_list)
    att_out_all = Activation('tanh')(att_out_all)
    
    return att_out_all
  

# channel-wise attention, weighted sum version
# the weights are learnt from k-max values in each 2D feature.
# dense layer along the feature indices axis is used to capture any hard information 
# on the index of features
def att_channel_config1(input_layer,kmax=7,att_outdim=16):
    maxave_layer = get_attention_input(input_layer,kmax)
    maxave_layer = Activation('tanh')(maxave_layer)
    maxave_layer3 = Dense(att_outdim, activation='relu')(maxave_layer)    
    nfeatures = maxave_layer3.shape.as_list()[1]
    a = Permute((2, 1))(maxave_layer3)
    a = Dense(nfeatures, activation='softmax')(a)    
    a_probs = Permute((2, 1), name='attention_weights')(a)
    weighted_sum_list = []
    for i in range(att_outdim):
        att_w = Lambda(lambda x: x[:, :, i], name='attention_weight_'+str(i))(a_probs)
        weighted_features = Multiply()([input_layer,att_w])
        weighted_sum = Lambda(sum_weighted_imputs)(weighted_features)
        weighted_sum_list.append(weighted_sum)
    if att_outdim > 1:
        att_out_all = Concatenate(axis=3)(weighted_sum_list)
    else:
        att_out_all = weighted_sum_list[0]
    att_out_all = Activation('tanh')(att_out_all)
    return att_out_all


# channel-wise attention, weighted sum version
# the weights are learnt from k-max values in each 2D feature.
# Dense models use DIFFERENT weights for different channels
# no more dense layers on the index of features
def att_channel_config2(input_layer,kmax=7,att_outdim=16):
    maxave_layer = get_attention_input(input_layer,kmax)
    maxave_layer2 = Activation('tanh')(maxave_layer)
        
    att_in_layers_list = []
    for i in range(input_layer.shape.as_list()[3]):
        att_input1 = Lambda(lambda x: x[:, i, :], name='attention_input_'+str(i))(maxave_layer2)
        att_input2 = Dense(att_outdim, activation='relu')(att_input1)
        att_in_layers_list.append(att_input2)
    att_out_all = Lambda(stack_layers)(att_in_layers_list)
    attweight_layer = Permute((2,1))(att_out_all)
    attweight_layer2 = Softmax(1,name='attention_weights')(attweight_layer)
    
    weighted_sum_list = []
    for i in range(att_outdim):
        att_w = Lambda(lambda x: x[:, :, i], name='attention_weight_'+str(i))(attweight_layer2)
        weighted_features = Multiply()([input_layer,att_w])
        weighted_sum = Lambda(sum_weighted_imputs)(weighted_features)
        weighted_sum_list.append(weighted_sum)
    if att_outdim > 1:
        att_out_all = Concatenate(axis=3)(weighted_sum_list)
    else:
        att_out_all = weighted_sum_list[0]
    att_out_all = Activation('tanh')(att_out_all)
    return att_out_all

# channel-wise attention, weighted sum version
# the weights are learnt from k-max values in each 2D feature.
# AFS(AAAI2019) implementation, dense models use SAME weights for different channels
def att_channel_config3(input_layer,kmax=7,att_outdim = 16):
    ###################helpers###################
    contact_feature_num_2D = input_layer.shape.as_list()[3]
    def kmaxpooling2D(l):
        return _kmaxpooling2D(l,contact_feature_num_2D,kmax)
    #############################################
    maxave_layer = get_attention_input(input_layer,kmax)
    maxave_layer2 = Activation('tanh')(maxave_layer)
        
    att_weights_layers_list = []
    for i in range(input_layer.shape.as_list()[3]):
        att_input1 = Lambda(lambda x: x[:, i, :], name='attention_input_'+str(i))(maxave_layer2)
        att_input2 = Dense(att_outdim)(att_input1)
        att_input3 = Dense(2, activation='softmax')(att_input2)
        att_input4 = Lambda(lambda x: x[:, 1], name='attention_net_'+str(i))(att_input3)
        att_weights_layers_list.append(att_input4)
    att_weights = Lambda(stack_layers, name='attention_weights')(att_weights_layers_list)
    att_out = Multiply()([input_layer,att_weights])
#    att_out = Activation('tanh')(att_out)        
    return att_out

# channel-wise attention, weighted sum version
# the weights are learnt from k-max values in each 2D feature.
# NO dense models to capture any hard information on the index of features
# One-hot encoding as positional embeddings
def att_channel_config4(input_layer,kmax=7,att_outdim=16):
    feature_num = input_layer.shape.as_list()[3]
    maxave_layer = get_attention_input(input_layer,kmax)
    maxave_layer2 = Activation('tanh')(maxave_layer)
    
    
    add_layer = Lambda(lambda x:x[0]+x[1], output_shape=lambda x:x[0])
    
    positional_embedding = GetPosEncodingMatrix(feature_num, att_outdim)
    positional_embedding = np.eye(feature_num,dtype=np.float32)
    
    positional_embedding2 = tf.constant(positional_embedding)
    
    add_layer([maxave_layer2,positional_embedding2])
    
    
    maxave_layer3 = Permute((2,1))(maxave_layer2)
    
    

    maxave_layer4 = Lambda(tf_reshape_layer)(maxave_layer3)
    score1 = Dense(att_outdim, use_bias=False, name='attention_score_vec')(maxave_layer4)
    attweight_layer = Softmax(1,name='attention_weights')(score1)
    
    weighted_sum_list = []
    for i in range(att_outdim):
        att_w = Lambda(lambda x: x[:, :, i], name='attention_weight_'+str(i))(attweight_layer)
        weighted_features = Multiply()([input_layer,att_w])
        weighted_sum = Lambda(sum_weighted_imputs)(weighted_features)
        weighted_sum_list.append(weighted_sum)
        
    if att_outdim > 1:
        att_out_all = Concatenate(axis=3)(weighted_sum_list)
    else:
        att_out_all = weighted_sum_list[0]
    att_out_all = Activation('tanh')(att_out_all)
    
    return att_out_all
    

if __name__ == "__main__":
    # test_code
    from keras.layers import Input
    from keras.models import Model
    from keras.callbacks import TensorBoard
    from keras.optimizers import Adam
    import numpy as np
    import matplotlib.pyplot as plt
    
  
#    tbCallBack = TensorBoard(log_dir='/home/chen/Dropbox/MU/workspace/dncon4/tbl/',\
#                             histogram_freq=0,write_graph=True, write_images=True)
    tbCallBack = TensorBoard(log_dir='C:/Users/chen/Documents/Dropbox/MU/workspace/dncon4/tbl/',\
                             histogram_freq=0,write_graph=True, write_images=True)    
    att_outdim=3;contact_feature_num_2D=20
    
    np.random.seed(10)
    x1 = np.random.random((640,25,25,contact_feature_num_2D))
    x2 = x1.copy()
    x2[:,:,:,-1] = x2[:,:,:,-1] * 1.5
    y1 = np.mean(x2,axis=3,keepdims = True)
#    y1 = np.tanh(y1)
    
    x1_val = np.random.random((64,25,25,contact_feature_num_2D))
    x2_val = x1_val.copy()
    x2_val[:,:,:,-1] = x2_val[:,:,:,-1] * 1.5
    y1_val = np.mean(x2_val,axis=3,keepdims = True)
#    y1_val = np.tanh(y1_val)
    
    contact_input_shape=(None,None,contact_feature_num_2D)
    input_layer = Input(shape=contact_input_shape)
    
    #############################################
#    att_layer = att_channel_config0(input_layer,kmax=7,att_outdim=4);epochs=30
#    att_layer = att_channel_config1(input_layer,kmax=7,att_outdim=3);epochs=30
    att_layer = att_channel_config2(input_layer,kmax=7,att_outdim=3);epochs=30
#    att_layer = att_channel_config3(input_layer,kmax=7,attention_net_hidden_nodes = 2);epochs=90
#    att_layer = att_channel_config4(input_layer,kmax=7,att_outdim=1);epochs=30
    #############################################
    
    output_layer = Lambda(sum_weighted_imputs)(att_layer)
    m1 = Model(input_layer,output_layer)
    m1.compile(Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000),'mse')
    m1.fit(x1,y1, epochs=epochs,callbacks=[tbCallBack],validation_data = [x1_val,y1_val])
    
    a1 = m1.get_layer('attention_weights').get_output_at(0)
    m2 = Model(input_layer,a1)
    p1 = m2.predict(x1_val)
    
    avg_p1 = np.mean(p1,axis=0)
    plt.imshow(avg_p1)
#    plt.imshow(p1)
    
#    y_pred1 = m1.predict(x1_val)[0]
#    mse1 = np.mean(np.square(np.absolute(y_pred1-y1_val[0,:,:,:])))
#    y_pred1-y1_val[0,:,:,:]
#    
#    y_pred2 = np.matmul(x1_val[0,:,:,:],p1[None,0,:].transpose())
#    y_pred1-y_pred2




