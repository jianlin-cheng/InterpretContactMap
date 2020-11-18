import math
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.optimizers import Adam
from keras import backend as K
from keras.utils import CustomObjectScope, Sequence
from model_utils import InstanceNormalization,ContactTransformerV5, ContactTransformerV7
from model_utils import extract_sequence_weights, extract_regional_weights


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
    kmax = int(model_parameters.iloc[12][0])
    att_outdim = int(model_parameters.iloc[13][0])
    insert_pos = model_parameters.iloc[14][0]
    n_head = int(model_parameters.iloc[15][0])
    att_dim = int(model_parameters.iloc[16][0])
    K.clear_session()
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
    model = model_func(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,\
                               nb_layers,opt,initializer,loss_function,weight_p,weight_n,att_config, \
                               kmax,att_outdim,insert_pos,n_head=n_head,att_dim=att_dim)
    return model


def load_plm(plm_data):
    plm_rawdata = np.fromfile(plm_data, dtype=np.float32)
    L = int(math.sqrt(plm_rawdata.shape[0]/21/21))
    inputs_plm = plm_rawdata.reshape(1,441,L,L)
    inputs_plm = np.moveaxis(inputs_plm,1,-1)
    return inputs_plm

# Load 1D features and tile them to 2D: (L,n)->(L,L,2n)
def load_features1D(feature_data,name,dim=20):
    f = open(feature_data, "r")
    all_other_data = f.readlines()
    f.close() 
    feature_idx = all_other_data.index('# '+name+'\n')
    d = [line.strip().split() for line in all_other_data[(feature_idx+1):(feature_idx+dim+1)]]
    return np.array(d)

def tile_2d(d):
    L = d.shape[1]
    dim = d.shape[0]
    inputs_feature = np.zeros((1,L,L,dim*2))
    for i in range(dim):
        for j in range(L):
            inputs_feature[:,:,j,2*i] = inputs_feature[:,j,:,2*i+1] = [float(x) for x in d[i]]
    return inputs_feature

def init_model(model_type):
    if model_type == 'sequence':
        model_func = ContactTransformerV5
        model_dir = 'models/sequence_attention/'
    elif model_type == 'regional':
        model_func = ContactTransformerV7
        model_dir = 'models/regional_attention/'    
    with CustomObjectScope({'InstanceNormalization': InstanceNormalization, 'tf':tf}):
        model = load_model_from_config(model_dir+'model_config.txt',model_func)
    return model,model_dir

def predict_cmap(input_data,model_type,weights=False):
    model,model_dir = init_model(model_type)
    model.load_weights(model_dir+'model_weights.h5')
    pred_out = model.predict(input_data, batch_size= 1)
    L = pred_out.shape[1]
    CMAP = pred_out.reshape((L, L))
    CMAP_final = (CMAP+CMAP.T)/2
    score = None
    if weights:
        score = extract_attention_score(model,input_data, model_type)
    return CMAP_final, score


def load_sample_config(sample_list_file):
    f = open(sample_list_file, "r")
    all_sample_data = f.readlines()
    f.close() 
    sample_list_train = all_sample_data[1].strip().split(' ')
    sample_list_val = all_sample_data[3].strip().split(' ')
    return sample_list_train,sample_list_val
    
    
class data_generator(Sequence):
    def __init__(self,plm_data_path,pssm_data_path,label_path,sample_list):
        self.plm_data_path = plm_data_path
        self.pssm_data_path = pssm_data_path
        self.label_path = label_path
        self.sample_list = sample_list
        self.on_epoch_end()

    def __len__(self):
        #Denotes the number of batches per epoch
        return len(self.sample_list)

    def __getitem__(self, index):
        # Generate data
        X, y = self.__data_generation(index)
        return X, y

    def on_epoch_end(self):
      self.indexes = np.arange(len(self.sample_list))
      np.random.shuffle(self.indexes)
        
    def __data_generation(self,index):
        current_sample = self.sample_list[self.indexes[index]]
        inputs_plm = load_plm(self.plm_data_path+current_sample+'.plm')
        inputs_pssm = tile_2d(np.load(self.pssm_data_path+current_sample+'.pssm.npy'))
        X = [inputs_plm,inputs_pssm]
        Y = np.load(self.label_path+current_sample+'.npy')
        return X, Y

def extract_attention_score(model,X,model_type):
    if model_type == 'sequence':
        return extract_sequence_weights(X,model)
    elif model_type == 'regional':
        return extract_regional_weights(X, model)
    else:
        raise ValueError()
