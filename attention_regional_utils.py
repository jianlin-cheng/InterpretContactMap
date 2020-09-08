import keras.backend as K

from keras.layers import Dense, Conv2D, Input, Softmax, Multiply, Lambda
from keras.models import Model
from keras.optimizers import Adam

def get_stretch_weight(kernel_size,depth=1):
    w = np.zeros((kernel_size,kernel_size,depth,depth*kernel_size*kernel_size))
    for k in range(depth):
        for i in range(kernel_size):
            for j in range(kernel_size):
                w[i,j,k,k*kernel_size*kernel_size+kernel_size*i+j] = 1
    return np.asarray(w)

def sum_axis3(l):
    return K.sum(l,3,True)

def regional_attention_model(kernel_size,depth = 1, \
                             opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000),\
                             loss = 'binary_crossentropy'):
    input_layer = Input((None,None,depth),name='attention_input')
    stretch_layer = Conv2D(kernel_size*kernel_size*depth, (kernel_size,kernel_size),  \
                           padding='same',activation=None, use_bias=False, \
                           bias_constraint=None, trainable=False, name = 'strech_layer')
    out_tensor = stretch_layer(input_layer)
    out_tensor1 = Dense(32)(out_tensor)
    out_tensor2 = Dense(kernel_size*kernel_size*depth)(out_tensor1)
    out_tensor3 = Softmax(3,name='attention_weights')(out_tensor2)
    out_tensor4 = Multiply()([out_tensor3,out_tensor])
    out_tensor5 = Lambda(sum_axis3)(out_tensor4)
    
    m1 = Model(input_layer,out_tensor5)
    m1.compile(opt,loss)
    
    m1.get_layer('strech_layer').set_weights([get_stretch_weight(kernel_size,depth=depth)])
    return m1

if __name__ == "__main__":
    # test_code
    from keras.callbacks import TensorBoard
    from keras.optimizers import Adam
    import numpy as np
    import matplotlib.pyplot as plt
    
#    tbCallBack = TensorBoard(log_dir='/home/chen/Dropbox/MU/workspace/dncon4/tbl/',\
#                             histogram_freq=0,write_graph=True, write_images=True)
    tbCallBack = TensorBoard(log_dir='C:/Users/chen/Documents/Dropbox/MU/workspace/dncon4/tbl/',\
                             histogram_freq=0,write_graph=True, write_images=True)    
 
    kernel_size = 3; depth = 3
    
    np.random.seed(10)
    x1 = np.random.random((640,64,64,1))*4
    y1 = x1.copy()
    y1 = y1 + np.random.random((640,64,64,1))
    y1[:,0,:,:] = y1[:,0,:,:] + 1.5*y1[:,1,:,:]
    
    x2 = np.random.random((640,64,64,depth))*4
    y2 = x2.copy()
    y2 = y2 + np.random.random((640,64,64,depth))
    y2[:,0,:,:] = y2[:,0,:,:] + 1.5*y2[:,1,:,:]   
    y3 = np.mean(y2,axis=3,keepdims =True)
    
    
    m1 = regional_attention_model(kernel_size,loss = 'mse')
    history1 = m1.fit(x1,y1,epochs = 60)
    
    a1 = m1.get_layer('attention_weights').get_output_at(0)
    input_layer = m1.get_layer('attention_input').get_input_at(0)
    m1a = Model(input_layer,a1)
    p1 = m1a.predict(x1)
    plt.imshow(p1[0,0,7,:].reshape((kernel_size,kernel_size)))
    plt.imshow(p1[0,1,7,:].reshape((kernel_size,kernel_size)))

    m2 = regional_attention_model(kernel_size,depth=depth,loss = 'mse')
    history2 = m2.fit(x2,y3,epochs = 60)
    a2 = m2.get_layer('attention_weights').get_output_at(0)
    input_layer = m2.get_layer('attention_input').get_input_at(0)
    m2a = Model(input_layer,a2)
    p2 = m2a.predict(x2) 
    plt.imshow(p2[0,0,:,:])
    plt.imshow(p2[0,1,:,:])
#    plt.imshow(p2[0,1,7,:].reshape((kernel_size,kernel_size)))

