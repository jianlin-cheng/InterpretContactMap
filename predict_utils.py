import numpy as np
import pandas as pd


import keras.backend as K
import tensorflow as tf

from keras.models import model_from_json,load_model, Sequential, Model
from keras.utils import CustomObjectScope

from Model_construct import *
from contact_transformer import *
from DNCON_lib import *
from keras.optimizers import Adam



def load_model_from_config(model_config,model_func):
    model_parameters = pd.read_csv(model_config,header =0, index_col =0)
    win_array = int(model_parameters.iloc[0][0])
    feature_2D_num = list(map(int, model_parameters.iloc[1][0][1:-1].split(',')))
    use_bias = model_parameters.iloc[2][0]=='True'
    hidden_type = model_parameters.iloc[3][0]
    nb_filters = int(model_parameters.iloc[4][0])
    nb_layers = int(model_parameters.iloc[5][0])
    initializer = model_parameters.iloc[7][0]
    loss_function = model_parameters.iloc[8][0]
    weight_p = float(model_parameters.iloc[9][0])
    weight_n = float(model_parameters.iloc[10][0])
    att_config = int(model_parameters.iloc[11][0])
    # if att_config<1:
    #     att_config = int(att_config)
    kmax = int(model_parameters.iloc[12][0])
    att_outdim = int(model_parameters.iloc[13][0])
    insert_pos = model_parameters.iloc[14][0]
    
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
    DNCON4 = model_func(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,\
                               nb_layers,opt,initializer,loss_function,weight_p,weight_n,att_config, \
                               kmax,att_outdim,insert_pos)
    return DNCON4


def load_plm(plm_data):
    plm_rawdata = np.fromfile(plm_data, dtype=np.float32)
    L = int(math.sqrt(plm_rawdata.shape[0]/21/21))
    inputs_plm = plm_rawdata.reshape(1,441,L,L)
    inputs_plm = np.moveaxis(inputs_plm,1,-1)
    return inputs_plm

def load_pssm(pssm_data):
    f = open(pssm_data, "r")
    all_other_data = f.readlines()
    pssm_idx = all_other_data.index('# PSSM\n')
    plm_rawdata = [line.strip().split() for line in all_other_data[(pssm_idx+1):(pssm_idx+21)]]
    L = len(plm_rawdata[0])
    inputs_pssm = np.zeros((1,L,L,40))
    for i in range(20):
        for j in range(L):
            inputs_pssm[:,:,j,2*i] = inputs_pssm[:,j,:,2*i+1] = [float(x) for x in plm_rawdata[i]]
    return inputs_pssm

def predict_cmap(model_type,input_data):
    if model_type == 'sequence_attention':
        model_func = ContactTransformerV5
        model_dir = 'models/sequence_attention/'
    elif model_type == 'regional_attention':
        model_func = ContactTransformerV7
        model_dir = 'models/regional_attention/'   
        
    with CustomObjectScope({'InstanceNormalization': InstanceNormalization, 'tf':tf}):
        DNCON4 = load_model_from_config(model_dir+'model_config.txt',model_func)
        
    DNCON4.load_weights(model_dir+'model_weights.h5')
    pred_out = DNCON4.predict(input_data, batch_size= 1)
    L = pred_out.shape[1]
    CMAP = pred_out.reshape((L, L))
    CMAP_final = (CMAP+CMAP.T)/2
    return CMAP_final