import tensorflow as tf
import numpy as np
import pandas as pd
import math
from keras.models import model_from_json, Model
from keras.utils import CustomObjectScope
from Model_construct import InstanceNormalization
from keras.optimizers import Adam

def import_data2D(data_file,feature_dim,length=None):
    # import 2d data from numpy file to array with shape (1,L,L,feature_dim)
    rawdata = np.fromfile(data_file, dtype=np.float32)
    if not length is None:
        if rawdata.shape[0] != length*length*feature_dim:
            raise ValueError('feature_dim and length provided is not compatible with stored data.')
    else:
        length = int(math.sqrt(rawdata.shape[0]/feature_dim))
    rawdata2 = rawdata.reshape(1,feature_dim,length,length)
    out = np.transpose(rawdata2, (0,2,3,1))
    return out
    
def import_data1D(data_file,start_idx,feature_dim,length):
    # import 1d data from line start_idx in text file to array with shape (length,feature_dim)
    out = np.zeros((length,feature_dim))
    f = open(data_file, "r")
    lines = f.readlines()
    f.close()
    for i in range(feature_dim):
        out[:,i] = np.array(lines[start_idx+i].strip().split(' ')).astype(np.float)
    return out 

def tile1Dto2D(data1d):
    # tile 1d data (length,feature_dim) to 2d with shape (1,L,L,feature_dim*2)
    # for last dim, odd indices are the same for all rows
    #               even indices are the same for all cols
    data2d = np.zeros((1,data1d.shape[0],data1d.shape[0],data1d.shape[1]*2))
    for i in range(data1d.shape[1]):
        data2d[0,0,:,2*i+1] = data1d[:,i]
        data2d[0,:,0,2*i] = data1d[:,i]
    return data2d

def get_precision(pred_mat,true_mat,top_n,dist=24):
    l = true_mat.shape[0]
    scores = []
    true_labels = []
    for i in range(l):
        for j in range(i+dist,l):
            scores.append(pred_mat[i,j])
            true_labels.append(true_mat[i,j])
            
    scores2 = np.array(scores)
    true_labels2 = np.array(true_labels)
    a = scores2.argsort()[-top_n:][::-1]
    return np.sum(true_labels2[a])/top_n

def load_model_json(json_file,weight_file):
    with CustomObjectScope({'InstanceNormalization': InstanceNormalization, 'tf':tf}):
        json_string = open(json_file).read()
        model = model_from_json(json_string)
        model.load_weights(weight_file)
    return model

def extract_attention_score(model,X,attention_layer_name,input_num = 1):
    #extract attention score from model
    if input_num == 1:
        input_layer = model.get_input_at(0)
    else:
        input_layer = [model.get_input_at(i) for i in range(input_num-1)][0]
    att_viz = Model(input_layer,model.get_layer(attention_layer_name).get_output_at(0))
    attention_score_mat = att_viz.predict(X)
    return attention_score_mat[0,:,:,:,:]

def get_average_sum_scores(att_weights_mat,pred_mat,true_mat,pred_threshold = 0.5,dist = 24):
    # get sum of attention weights and average of attention weights from all true positives
    # input (l,l,region_len*region_len), output two matrices of (l,l)
    region_size = int(np.sqrt(att_weights_mat.shape[2]))
    protein_len = att_weights_mat.shape[0]
    center_idx = int(np.ceil(region_size/2))
    sum_mat = np.zeros((protein_len,protein_len))
    avg_mat = np.zeros((protein_len,protein_len)) 
    count_mat = np.zeros((protein_len,protein_len))
    
    for i in range(protein_len):
        for j in range(i+dist,protein_len):
            if pred_mat[i,j] > pred_threshold and true_mat[i,j]==1:
                att_map = np.reshape(att_weights_mat[i,j],(region_size,region_size))
                for map_x in range(region_size):
                    for map_y in range(region_size):
                        pos_x = i+map_x+1 - center_idx
                        pos_y = j+map_y+1 - center_idx
                        if pos_x>=0 and pos_x<protein_len and pos_y>=0 and pos_y<protein_len and true_mat[i,j]>0:
                            sum_mat[pos_x,pos_y] += att_map[map_x,map_y]
                            count_mat[pos_x,pos_y] += 1
    for i in range(protein_len):
        for j in range(i+1,protein_len):
            if sum_mat[i,j] > 0 and count_mat[i,j]>0:
                avg_mat[i,j] = sum_mat[i,j]/count_mat[i,j]
            elif sum_mat[i,j] > 0 and count_mat[i,j]==0:
                print('error '+str(i)+','+str(j))
    return sum_mat, avg_mat

def largest_indices(ary, n,order='descend'):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    if order == 'descend':
        indices = np.argpartition(flat, -n)[-n:]
    else:
        indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def evaluate_permute(model,X,pred_mat,true_mat,sum_mat,d=1):
    tn_flag = np.logical_and(pred_mat<0.5,true_mat==0)
    tp_flag =  np.logical_and(pred_mat>=0.5,true_mat==1)
    l = true_mat.shape[0]
    
    # pick TP predictions with highest k attention scores
    sum_mat_tp = tp_flag * sum_mat
    k = np.sum(sum_mat_tp>0)
    indices_high_tp = largest_indices(sum_mat_tp, k,order='descend')
    pred_permute_high_tp = get_permute_prediction(model,X,indices_high_tp,d)
    prec_permute_high_tp = get_precision(pred_permute_high_tp,true_mat,int(l//5))
    
    # pick TP predictions with lowest k attention scores
    sum_mat_tp = tp_flag * (np.max(sum_mat)-sum_mat)
    indices_low_tp = largest_indices(sum_mat_tp, k,order='descend')
    pred_permute_low_tp = get_permute_prediction(model,X,indices_low_tp,d)
    prec_permute_low_tp = get_precision(pred_permute_low_tp,true_mat,int(l//5))
    
    # pick TN predictions with highest k attention scores
    sum_mat_tn = tn_flag * sum_mat
    indices_high_tn = largest_indices(sum_mat_tn, k,order='descend')
    pred_permute_high_tn = get_permute_prediction(model,X,indices_high_tn,d)
    prec_permute_high_tn = get_precision(pred_permute_high_tn,true_mat,int(l//5))
    
    # pick TP predictions with lowest k attention scores
    sum_mat_tn = tn_flag * (np.max(sum_mat)-sum_mat)
    indices_low_tn = largest_indices(sum_mat_tn, k,order='descend')
    pred_permute_low_tn = get_permute_prediction(model,X,indices_low_tn,d)
    prec_permute_low_tn = get_precision(pred_permute_low_tn,true_mat,int(l//5))
    
    return np.array([prec_permute_high_tp,prec_permute_low_tp,
                     prec_permute_high_tn,prec_permute_low_tn])
    
def get_permute_prediction(model,X,permute_ind,d=1):
    x_permute_0 = X[0].copy()
    x_permute_1 = X[1].copy()
    
    for idx in range(len(permute_ind[0])):
        for i in range(1-d,d):
            for j in range(1-d,d):
                center_i = permute_ind[0][idx]
                center_j = permute_ind[1][idx]
                
                xi = center_i+i
                xj = center_j+j
                if xi <0:
                    xi = 0
                elif xi >= x_permute_0.shape[1]:
                    xi = x_permute_0.shape[1]-1
                if xj <0:
                    xj = 0
                elif xj >= x_permute_0.shape[1]:
                    xj = x_permute_0.shape[1]-1
                    
                x_permute_0[0,xi,xj,:] = 0
                x_permute_1[0,xi,xj,:] = 0
    pred_permute = model.predict([x_permute_0,x_permute_1])
    pred_permute = pred_permute[0,:,:,0]
    return pred_permute

def load_model_from_config(config_file,model_func,weight_file):
    model_parameters = pd.read_csv(config_file,header =0, index_col =0)
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
    kmax = int(model_parameters.iloc[12][0])
    att_outdim = int(model_parameters.iloc[13][0])
    insert_pos = model_parameters.iloc[14][0]
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
    model = model_func(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,\
                               nb_layers,opt,initializer,loss_function,weight_p,weight_n,att_config, \
                               kmax,att_outdim,insert_pos)
    model.load_weights(weight_file)
    return model