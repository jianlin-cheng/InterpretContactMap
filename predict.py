import sys
import os
import time

#This may wrong sometime
sys.path.insert(0, sys.path[0])
from Model_construct import *
from DNCON_lib import *

import numpy as np
from keras.models import model_from_json,load_model, Sequential, Model
from keras.utils import CustomObjectScope
from random import randint
import keras.backend as K
import tensorflow as tf
from contact_transformer import contact_transformer


CV_dir=sys.argv[1]
model_type = sys.argv[2]

GLOABL_Path = sys.argv[3]
out_dir = sys.argv[4]



print("Find gloabl path :", GLOABL_Path)
###CASP13 FM
path_of_lists = GLOABL_Path + '/data/CASP13/lists-test-train/'
path_of_index = GLOABL_Path + '/data/CASP13/pdb_index/'
path_of_X= GLOABL_Path + '/data/CASP13/feats_fixed/'  # new pipline
path_of_Y= GLOABL_Path + '/data/CASP13/feats_fixed/'  

reject_fea_path = GLOABL_Path + '/architecture_distance/lib/'
reject_fea_file = ['feature_to_use_2d.txt','feature_to_use_1d.txt' ]
#if feature list set other then will use this reject fea file
model_name = 'sequence_attention'# 'DNCON4_2dDIARES' 
feature_list = 'other'# ['combine', 'combine_all2d', 'other', 'ensemble']  # combine will output three map and it combine, other just output one pred
data_list_choose = 'test'# ['train', 'test', 'train_sub', 'all']
Maximum_length = 1000  # casp12 700
dist_string = "80"
loss_function = 'binary_crossentropy'
only_predict_flag = False # if do not have lable set True
if_use_binsize = False #False True
if_test_all_weights = False # if set True, this will test all in epoch weights acc on dataset

import pandas as pd
from keras.optimizers import Adam
from Model_construct import sequence_attention

#CV_dir = '/storage/htc/bdm/ccm3x/DNCON4/architecture_distance/outputs/DilatedResNet_arch/new_sturct/filter64_layers34_inter150_optnadam_ftsize3_batchsize1_27.0_sequence_attention'
def load_sequence_model(CV_dir):
    model_file = CV_dir+'/model-train-sequence_attention.json.txt'
    model_parameters = pd.read_csv(model_file,header =0, index_col =0)
    
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
    DNCON4 = sequence_attention(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,\
                               nb_layers,opt,initializer,loss_function,weight_p,weight_n,att_config, \
                               kmax,att_outdim,insert_pos)
    return DNCON4

#This may can only used on local machine
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
if memory_gpu == []:
    print("System is out of GPU memory, Run on CPU")
    os.environ['CUDA_VISIBLE_DEVICES']="0"
else:
    if np.max(memory_gpu) <= 2000:
        print("System is out of GPU memory, Run on CPU")
        os.environ['CUDA_VISIBLE_DEVICES']="7"
        os.system('rm tmp')
        # sys.exit(1)
    else:
        os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
        os.system('rm tmp')

def chkdirs(fn):
    dn = os.path.dirname(fn)
    if not os.path.exists(dn): os.makedirs(dn)


print("\n######################################\n佛祖保佑，永不迨机，永无bug，精度九十九\n######################################\n")

#predict_method has three value : bin_class, mul_class, real_dist
predict_method = 'bin_class'
if loss_function == 'weighted_BCE':
    predict_method = 'bin_class'
    path_of_Y_train = path_of_Y + '/bin_class/'
    path_of_Y_evalu = path_of_Y + '/bin_class/'
    
    if weight_p <= 1:
        weight_n = 1.0 - weight_p
        
    loss_function = _weighted_binary_crossentropy(weight_p, weight_n)
elif loss_function == 'weighted_CCE':
    predict_method = 'mul_class'
    loss_function = _weighted_categorical_crossentropy(weight_p)
elif loss_function == 'weighted_MSE':
    predict_method = 'real_dist'
    path_of_Y_train = path_of_Y + '/real_dist/'
    path_of_Y_evalu = path_of_Y + '/bin_class/'
    loss_function = _weighted_mean_squared_error(1)
elif loss_function == 'binary_crossentropy':
    predict_method = 'bin_class'
    path_of_Y_train = path_of_Y + '/bin_class/'
    path_of_Y_evalu = path_of_Y + '/bin_class/'
    loss_function = loss_function
else:
    predict_method = 'real_dist'
    path_of_Y_train = path_of_Y + '/real_dist/'
    path_of_Y_evalu = path_of_Y + '/bin_class/'
    loss_function = loss_function

tr_l = {}
trs_l = {}
te_l = {}
if 'train' == data_list_choose:
    tr_l = build_dataset_dictionaries_train(path_of_lists)
    
if 'test' == data_list_choose:
    te_l = build_dataset_dictionaries_test(path_of_lists)
    
if 'train_sub' == data_list_choose:
    trs_l = build_dataset_dictionaries_other(path_of_lists, 'train_sub.lst')
    
if 'all' == data_list_choose:
    tr_l = build_dataset_dictionaries_train(path_of_lists)
    te_l = build_dataset_dictionaries_test(path_of_lists)
    
all_l = te_l.copy()        
all_l.update(trs_l)       
all_l.update(tr_l)

print('Total Number to predict = ',str(len(all_l)))

##### running validation
selected_list = subset_pdb_dict(all_l,   0, Maximum_length, 5000, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset

model_out= "%s/model-train-%s.json" % (CV_dir, model_name)
model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir, model_name)
model_weight_epochs = "%s/model_weights/" % (CV_dir)
model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir, model_name)
model_weight_top10 = "%s/model_weights_top10/" % (CV_dir)

iter_num = 0
if if_test_all_weights == False:
    iter_num = 1
else:
    iter_num = len(os.listdir(model_weight_epochs))
    pred_target_acc_dir = "%s/target_acc_epoch/"%(CV_dir)
    chkdirs(pred_target_acc_dir)


for index in range(iter_num):
    with CustomObjectScope({'InstanceNormalization': InstanceNormalization, 'tf':tf}):
        DNCON4 = contact_transformer()
        OTHER = []
        for feafile_num in range(len(reject_fea_file)):
            OTHER.append(reject_fea_path + reject_fea_file[feafile_num])
    step_num = 0


    ####Predict the trainig data set
    for key in selected_list:
        value = selected_list[key]
        p1 = {key: value}
        if if_use_binsize:
            Maximum_length = Maximum_length
        else:
            Maximum_length = value
        if len(p1) < 1:
            continue
        print("start predict %s %d" %(key, value))

        if 'other' in feature_list:
            if isinstance(reject_fea_file, str) == True:
                selected_list_2D_other = get_x_2D_from_this_list(p1, path_of_X, Maximum_length,dist_string, OTHER, value)
                print(selected_list_2D_other.shape)
                if type(selected_list_2D_other) == bool:
                    continue
                DNCON4_prediction_other = DNCON4.predict([selected_list_2D_other], batch_size= 1)  
            elif len(reject_fea_file)>=2:
                pred_temp = []
                bool_flag = False
                for fea_num in range(len(OTHER)):
                    temp = get_x_2D_from_this_list(p1, path_of_X, Maximum_length, dist_string, reject_fea_file[fea_num], value)
                    print("selected_list_2D.shape: ",temp.shape)
                    if type(temp) == bool:
                        bool_flag= True
                    pred_temp.append(temp)
                if bool_flag == True:
                    continue
                else:
                    DNCON4_prediction_other = DNCON4.predict(pred_temp, batch_size= 1)
              

        np.save(out_dir+'/'+key+'.npy',DNCON4_prediction_other)

        # if just for generate predict map, stop here is fine 
