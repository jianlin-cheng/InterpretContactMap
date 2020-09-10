
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:47:26 2017

@author: Zhiye
"""
import sys
import os
import time

#This may wrong sometime
sys.path.insert(0, sys.path[0])
from Model_construct import *
from contact_transformer import *
from DNCON_lib import *
import pandas as pd
from keras.optimizers import Adam
from Model_construct import sequence_attention
import numpy as np
from keras.models import model_from_json,load_model, Sequential, Model
from keras.utils import CustomObjectScope
from random import randint
import keras.backend as K
import tensorflow as tf

# CV_dir = '/storage/htc/bdm/ccm3x/DNCON4/analysis/models/test/'
CV_dir=sys.argv[1]

GLOABL_Path = sys.path[0].split('DNCON4')[0]+'DNCON4/'
print("Find gloabl path :", GLOABL_Path)
###CASP13 FM
path_of_lists = GLOABL_Path + '/data/CASP13/lists-test-train/'
path_of_index = GLOABL_Path + '/data/CASP13/pdb_index/'
path_of_X= GLOABL_Path + '/data/CASP13/feats_fixed/'  # new pipline
path_of_Y= GLOABL_Path + '/data/CASP13/feats_fixed/'  

reject_fea_path = GLOABL_Path + '/architecture_distance/lib/'
#if feature list set other then will use this reject fea file
feature_list = 'other'# ['combine', 'combine_all2d', 'other', 'ensemble']  # combine will output three map and it combine, other just output one pred
data_list_choose = 'test'# ['train', 'test', 'train_sub', 'all']
Maximum_length = 1000  # casp12 700
dist_string = "80"
loss_function = 'binary_crossentropy'
only_predict_flag = False # if do not have lable set True
if_use_binsize = False #False True


model_dict = {'sequence_attention':sequence_attention,
              'baselineModel':baselineModel,
              'ContactTransformerV7':ContactTransformerV7,
              'ContactTransformerV5':ContactTransformerV5,
              'ContactTransformerV6':ContactTransformerV6,
              'baselineModelSym':baselineModelSym}

model_name = CV_dir.split('/')[-2].split('_')[-1]
print(model_name)
model_func = model_dict[model_name]
# if model_name == 'sequence_attention':
#     reject_fea_file = ['feature_to_use_baseline.txt','feature_to_use_1d.txt' ]
# else:
#     reject_fea_file = 'feature_to_use_baseline.txt'
reject_fea_file = 'feature_to_use_baseline.txt'
#CV_dir = '/storage/htc/bdm/ccm3x/DNCON4/architecture_distance/outputs/DilatedResNet_arch/new_sturct/filter64_layers34_inter150_optnadam_ftsize3_batchsize1_27.0_sequence_attention'
def load_sequence_model(CV_dir):
    model_file = CV_dir+'/model-train-'+model_name+'.json.txt'
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
iter_num = len(os.listdir(model_weight_epochs))
pred_target_acc_dir = "%s/target_acc_epoch/"%(CV_dir)
chkdirs(pred_target_acc_dir)
    

def evaluate_performance(weight_file):
    weights = os.listdir(model_weight_epochs)
    pred_history_out = "%s/predict.acc_history" % (CV_dir) 
    with open(pred_history_out, "a") as myfile:
        myfile.write(time.strftime('%Y-%m-%d %H:%M:%S\n',time.localtime(time.time())))
    with CustomObjectScope({'InstanceNormalization': InstanceNormalization, 'tf':tf}):
        if os.path.exists(model_out):
            json_string = open(model_out).read()
            DNCON4 = model_from_json(json_string)
        else:
            DNCON4 = load_sequence_model(CV_dir)
#    weight_file = '/storage/htc/bdm/ccm3x/DNCON4/architecture_distance/outputs/DilatedResNet_arch/new_sturct/filter64_layers34_inter150_optnadam_ftsize3_batchsize1_25.0_ContactTransformerV7/model_weights/model-train-weight-ContactTransformerV7-epoch21.h5'
    DNCON4.load_weights(weight_file)

    model_predict= "%s/pred_map/"%(CV_dir)
    chkdirs(model_predict)
    if 'combine' == feature_list:
        model_predict_cov= "%s/pred_map/cov/"%(CV_dir)
        model_predict_plm= "%s/pred_map/plm/"%(CV_dir)
        model_predict_pre= "%s/pred_map/pre/"%(CV_dir)
        model_predict_sum= "%s/pred_map/sum/"%(CV_dir)
        chkdirs(model_predict_cov)
        chkdirs(model_predict_plm)
        chkdirs(model_predict_pre)
        chkdirs(model_predict_sum)
        COV = reject_fea_path + 'feature_to_use_cov.txt'
        PLM = reject_fea_path + 'feature_to_use_plm.txt'
        PRE = reject_fea_path + 'feature_to_use_pre.txt'
    elif 'combine_all2d' == feature_list:
        model_predict_cov= "%s/pred_map/cov/"%(CV_dir)
        model_predict_plm= "%s/pred_map/plm/"%(CV_dir)
        model_predict_pre= "%s/pred_map/pre/"%(CV_dir)
        model_predict_sum= "%s/pred_map/sum/"%(CV_dir)
        chkdirs(model_predict_cov)
        chkdirs(model_predict_plm)
        chkdirs(model_predict_pre)
        chkdirs(model_predict_sum)
        COV = reject_fea_path + 'feature_to_use_cov_other2d_com.txt'
        PLM = reject_fea_path + 'feature_to_use_plm_other2d_com.txt'
        PRE = reject_fea_path + 'feature_to_use_pre_other2d_com.txt'
    elif 'other' == feature_list:
        if isinstance(reject_fea_file, str) == True:
            OTHER = reject_fea_path + reject_fea_file
        elif len(reject_fea_file) >= 2:
            OTHER = []
            for feafile_num in range(len(reject_fea_file)):
                OTHER.append(reject_fea_path + reject_fea_file[feafile_num])
    elif 'ensemble' == feature_list: 
        model_predict_ensemble= "%s/ensemble/"%(model_predict) 
        model_predict_enscov= "%s/ensemble/cov/"%(model_predict) 
        model_predict_ensplm= "%s/ensemble/plm/"%(model_predict) 
        chkdirs(model_predict_ensemble) 
        chkdirs(model_predict_enscov) 
        chkdirs(model_predict_ensplm) 
        COV = reject_fea_path + 'feature_to_use_cov.txt'
        PLM = reject_fea_path + 'feature_to_use_plm.txt'
    else:
        print("Make sure you input the right paramters %s\n" % (feature_list))
        sys.exit(1)

    step_num = 0
    ####The init of acc parameters
    out_avg_pc_l5_cov = 0.0
    out_avg_pc_l2_cov = 0.0
    out_avg_pc_1l_cov = 0.0
    out_avg_acc_l5_cov = 0.0
    out_avg_acc_l2_cov = 0.0
    out_avg_acc_1l_cov = 0.0
    out_avg_pc_l5_plm = 0.0
    out_avg_pc_l2_plm = 0.0
    out_avg_pc_1l_plm = 0.0
    out_avg_acc_l5_plm = 0.0
    out_avg_acc_l2_plm = 0.0
    out_avg_acc_1l_plm = 0.0
    out_avg_pc_l5_sum = 0.0
    out_avg_pc_l2_sum = 0.0
    out_avg_pc_1l_sum = 0.0
    out_avg_acc_l5_sum = 0.0
    out_avg_acc_l2_sum = 0.0
    out_avg_acc_1l_sum = 0.0
    out_avg_pc_l5_other = 0.0
    out_avg_pc_l2_other = 0.0
    out_avg_pc_1l_other = 0.0
    out_avg_acc_l5_other = 0.0
    out_avg_acc_l2_other = 0.0
    out_avg_acc_1l_other = 0.0
    out_gloable_mse = 0.0
    out_weighted_mse = 0.0
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

        #This part is for COV feature 
        if 'combine' in feature_list:
            selected_list_2D_cov = get_x_2D_from_this_list(p1, path_of_X, Maximum_length,dist_string, COV, value)
            selected_list_2D_plm = get_x_2D_from_this_list(p1, path_of_X, Maximum_length,dist_string, PLM, value)
            # selected_list_2D_pre = get_x_2D_from_this_list(p1, path_of_X, Maximum_length,dist_string, PRE, value)
            if (type(selected_list_2D_cov) == bool or type(selected_list_2D_plm) == bool):
                print("seq %s do not have feature" %(key))
                continue
            DNCON4_prediction_cov = DNCON4.predict([selected_list_2D_cov], batch_size= 1)   
            DNCON4_prediction_plm = DNCON4.predict([selected_list_2D_plm], batch_size= 1)  
            # DNCON4_prediction_pre = DNCON4.predict([selected_list_2D_pre], batch_size= 1)  
            DNCON4_prediction_pre = 0

            CMAP = DNCON4_prediction_cov.reshape(Maximum_length, Maximum_length)
            Map_UpTrans = np.triu(CMAP, 1).T
            Map_UandL = np.triu(CMAP)
            real_cmap_cov = Map_UandL + Map_UpTrans
            CMAP = DNCON4_prediction_plm.reshape(Maximum_length, Maximum_length)
            Map_UpTrans = np.triu(CMAP, 1).T
            Map_UandL = np.triu(CMAP)
            real_cmap_plm = Map_UandL + Map_UpTrans
            # CMAP = DNCON4_prediction_pre.reshape(Maximum_length, Maximum_length)
            # Map_UpTrans = np.triu(CMAP, 1).T
            # Map_UandL = np.triu(CMAP)
            # real_cmap_pre = Map_UandL + Map_UpTrans

            # real_cmap_sum = (real_cmap_cov * 0.2 + real_cmap_plm * 0.6 + real_cmap_pre * 0.2)/2    
            real_cmap_sum = (real_cmap_cov * 0.35 + real_cmap_plm * 0.65)/2    
            pred_cmap = np.concatenate((real_cmap_cov.reshape(value,value,1), real_cmap_plm.reshape(value,value,1), real_cmap_sum.reshape(value,value,1)), axis=-1)

            cov_cmap_file = "%s/%s.txt" % (model_predict_cov,key)
            plm_cmap_file = "%s/%s.txt" % (model_predict_plm,key)
            # pre_cmap_file = "%s/%s.txt" % (model_predict_pre,key)
            sum_cmap_file = "%s/%s.txt" % (model_predict_sum,key)
            cmap_file = "%s/%s.npy" % (model_predict,key)
            np.savetxt(cov_cmap_file, real_cmap_cov, fmt='%.4f')
            np.savetxt(plm_cmap_file, real_cmap_plm, fmt='%.4f')
            # np.savetxt(pre_cmap_file, real_cmap_pre, fmt='%.4f')
            np.savetxt(sum_cmap_file, real_cmap_sum, fmt='%.4f')
            np.save(cmap_file, pred_cmap)

        if 'other' in feature_list:
            if isinstance(reject_fea_file, str) == True:
                selected_list_2D_other = get_x_2D_from_this_list(p1, path_of_X, Maximum_length,dist_string, OTHER, value)
                print(selected_list_2D_other.shape)
                if type(selected_list_2D_other) == bool:
                    continue
                print('start predict')
                DNCON4_prediction_other = DNCON4.predict([selected_list_2D_other[:,:,:,40:],
                                                          selected_list_2D_other[:,:,:,range(40)]], batch_size= 1)  
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
            
            if model_name == 'baselineModelSym':
                DNCON4_prediction_other = DNCON4_prediction_other[0]
            CMAP = DNCON4_prediction_other.reshape(Maximum_length, Maximum_length)
            
#            Map_UpTrans = np.triu(CMAP, 1).T
#            Map_UandL = np.triu(CMAP)
#            real_cmap_other = Map_UandL + Map_UpTrans
            real_cmap_other = (CMAP+CMAP.T)/2
            other_cmap_file = "%s/%s.txt" % (model_predict, key)
            np.savetxt(other_cmap_file, real_cmap_other, fmt='%.4f')
        
        # Predict different epoch map
        if 'ensemble' in feature_list:
            selected_list_2D_cov = get_x_2D_from_this_list(p1, path_of_X, Maximum_length,dist_string, COV, value)
            selected_list_2D_plm = get_x_2D_from_this_list(p1, path_of_X, Maximum_length,dist_string, PLM, value)
            
            weights = os.listdir(model_weight_top10)
            ensemble_cov_cmap = np.zeros((Maximum_length, Maximum_length))
            ensemble_plm_cmap = np.zeros((Maximum_length, Maximum_length))
            ensemble_sum_cmap = np.zeros((Maximum_length, Maximum_length))
            weight_num = 0
            for weight in weights:
                model_predict_epoch= "%s/%d/"%(model_predict_ensemble, weight_num)
                chkdirs(model_predict_epoch)
                model_weight_out = model_weight_top10 + '/' + weight
                weight_num += 1
                DNCON4.load_weights(weight_file)

                DNCON4_prediction_cov = DNCON4.predict([selected_list_2D_cov], batch_size= 1)   
                DNCON4_prediction_plm = DNCON4.predict([selected_list_2D_plm], batch_size= 1)  

                CMAP = DNCON4_prediction_cov.reshape(Maximum_length, Maximum_length)
                Map_UpTrans = np.triu(CMAP, 1).T
                Map_UandL = np.triu(CMAP)
                real_cmap_cov = Map_UandL + Map_UpTrans
                CMAP = DNCON4_prediction_plm.reshape(Maximum_length, Maximum_length)
                Map_UpTrans = np.triu(CMAP, 1).T
                Map_UandL = np.triu(CMAP)
                real_cmap_plm = Map_UandL + Map_UpTrans

                real_cmap_sum = (real_cmap_cov * 0.35 + real_cmap_plm * 0.65)/2

                ensemble_cov_cmap += real_cmap_cov
                ensemble_plm_cmap += real_cmap_plm
                ensemble_sum_cmap += real_cmap_sum
                sum_cmap_file = "%s/%s.txt" % (model_predict_epoch,key)
                np.savetxt(sum_cmap_file, real_cmap_sum, fmt='%.4f')

            ensemble_cov_cmap /= weight_num
            ensemble_plm_cmap /= weight_num
            ensemble_sum_cmap /= weight_num
            sum_cmap_file = "%s/%s.txt" % (model_predict_ensemble,key)
            np.savetxt(sum_cmap_file, ensemble_sum_cmap, fmt='%.4f')
            cov_cmap_file = "%s/%s.txt" % (model_predict_enscov,key)
            np.savetxt(cov_cmap_file, ensemble_cov_cmap, fmt='%.4f')
            plm_cmap_file = "%s/%s.txt" % (model_predict_ensplm,key)
            np.savetxt(plm_cmap_file, ensemble_plm_cmap, fmt='%.4f')

        # if just for generate predict map, stop here is fine 
        if only_predict_flag == True:
            continue

        print('Loading label sets..')
        # casp target

        if list(key)[0] == 'T' and 'CASP12' not in path_of_lists:
            selected_list_label, sub_map_index, sub_map_gap = get_y_from_this_list_casp(p1, path_of_Y_evalu, path_of_index, 0, Maximum_length, dist_string)# dist_string 80
        else:
            selected_list_label = get_y_from_this_list(p1, path_of_Y_evalu, 0, Maximum_length, dist_string)# dist_string 80
        if 'combine' in feature_list:
            # normal target full length
            if list(key)[0] != 'T' or 'CASP12' in path_of_lists:        
                DNCON4_prediction_cov = real_cmap_cov.reshape(len(p1), Maximum_length * Maximum_length)
                DNCON4_prediction_plm = real_cmap_plm.reshape(len(p1), Maximum_length * Maximum_length)
                DNCON4_prediction_sum = real_cmap_sum.reshape(len(p1), Maximum_length * Maximum_length)

                (a, b, c,avg_pc_l5_cov,avg_pc_l2_cov,avg_pc_1l_cov,avg_acc_l5_cov,avg_acc_l2_cov,avg_acc_1l_cov) = evaluate_prediction_4(p1, DNCON4_prediction_cov, selected_list_label, 24)
                val_acc_history_content_cov = "%s\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,'COV',avg_pc_l5_cov,avg_pc_l2_cov,avg_pc_1l_cov,avg_acc_l5_cov,avg_acc_l2_cov,avg_acc_1l_cov)
                
                (a, b, c,avg_pc_l5_plm,avg_pc_l2_plm,avg_pc_1l_plm,avg_acc_l5_plm,avg_acc_l2_plm,avg_acc_1l_plm) = evaluate_prediction_4(p1, DNCON4_prediction_plm, selected_list_label, 24)
                val_acc_history_content_plm = "%s\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,'PLM',avg_pc_l5_plm,avg_pc_l2_plm,avg_pc_1l_plm,avg_acc_l5_plm,avg_acc_l2_plm,avg_acc_1l_plm)
                
                (a, b, c,avg_pc_l5_sum,avg_pc_l2_sum,avg_pc_1l_sum,avg_acc_l5_sum,avg_acc_l2_sum,avg_acc_1l_sum) = evaluate_prediction_4(p1, DNCON4_prediction_sum, selected_list_label, 24)
                val_acc_history_content_sum = "%s\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,'SUM',avg_pc_l5_sum,avg_pc_l2_sum,avg_pc_1l_sum,avg_acc_l5_sum,avg_acc_l2_sum,avg_acc_1l_sum)
                
                step_num += 1
            else:
                # selected_list_label = selected_list_label.reshape(selected_list_label.shape[0], Maximum_length, Maximum_length)
                print("traget:%s" %(key), "sub_map_index:", sub_map_index, "sub_map_gap:", sub_map_gap)
                # break
                sub_selected_list_label = []
                DNCON4_prediction_cov = []
                DNCON4_prediction_plm = []
                DNCON4_prediction_sum = []
                for i in range(len(selected_list_label)):
                    label_sub = selected_list_label[i].reshape(Maximum_length,Maximum_length)
                    print(label_sub.shape)
                    left = sub_map_index[i][0]
                    right = sub_map_index[i][1] + 1
                    label_sub = label_sub[left:right,:]
                    label_sub = label_sub[:,left:right]

                    # real_cmap in shape L * L
                    real_cmap_cov_copy = np.copy(real_cmap_cov)
                    real_cmap_plm_copy = np.copy(real_cmap_plm)
                    real_cmap_sum_copy = np.copy(real_cmap_sum)
                    real_cmap_cov_copy = real_cmap_cov_copy[left:right,:]
                    real_cmap_cov_copy = real_cmap_cov_copy[:,left:right]
                    real_cmap_plm_copy = real_cmap_plm_copy[left:right,:]
                    real_cmap_plm_copy = real_cmap_plm_copy[:,left:right]
                    real_cmap_sum_copy = real_cmap_sum_copy[left:right,:]
                    real_cmap_sum_copy = real_cmap_sum_copy[:,left:right]
                    if np.sum(sub_map_gap[i]) < 1:
                        pass
                    else:
                        for j in range(len(sub_map_gap[i])):
                            sub_map_gap[i][j] = [x - sub_map_index[i][0] for x in sub_map_gap[i][j]]
                            gap_arrange = np.arange(sub_map_gap[i][j][0],sub_map_gap[i][j][1] + 1)
                            # print(gap_arrange)
                            label_sub = np.delete(label_sub, gap_arrange, axis=0)
                            label_sub = np.delete(label_sub, gap_arrange, axis=1)
                            real_cmap_cov_copy = np.delete(real_cmap_cov_copy, gap_arrange, axis=0)
                            real_cmap_cov_copy = np.delete(real_cmap_cov_copy, gap_arrange, axis=1)
                            real_cmap_plm_copy = np.delete(real_cmap_plm_copy, gap_arrange, axis=0)
                            real_cmap_plm_copy = np.delete(real_cmap_plm_copy, gap_arrange, axis=1)
                            real_cmap_sum_copy = np.delete(real_cmap_sum_copy, gap_arrange, axis=0)
                            real_cmap_sum_copy = np.delete(real_cmap_sum_copy, gap_arrange, axis=1)
                    l = label_sub.shape[0]

                    # sub_label_cmap_file = "%s/true_%s_%d.txt" % (model_predict_sub, key, i)
                    # sub_pred_cmap_file = "%s/pred_plm_%s_%d.txt" % (model_predict_sub, key, i)
                    # np.savetxt(sub_label_cmap_file, selected_list_label[i].reshape(Maximum_length,Maximum_length), fmt='%.4f')
                    # np.savetxt(sub_pred_cmap_file, real_cmap_plm_copy, fmt='%.4f')

                    label_sub = label_sub.reshape((-1, l*l))
                    sub_selected_list_label.append(label_sub)
                    real_cmap_cov_copy = real_cmap_cov_copy.reshape((-1, l*l))
                    DNCON4_prediction_cov.append(real_cmap_cov_copy)
                    real_cmap_plm_copy = real_cmap_plm_copy.reshape((-1, l*l))
                    DNCON4_prediction_plm.append(real_cmap_plm_copy)
                    real_cmap_sum_copy = real_cmap_sum_copy.reshape((-1, l*l))
                    DNCON4_prediction_sum.append(real_cmap_sum_copy)

                # DNCON4_prediction_cov = real_cmap_cov.reshape(len(p1), Maximum_length * Maximum_length)
                # DNCON4_prediction_plm = real_cmap_plm.reshape(len(p1), Maximum_length * Maximum_length)
                # DNCON4_prediction_sum = real_cmap_sum.reshape(len(p1), Maximum_length * Maximum_length)
                val_acc_history_content_cov = []
                val_acc_history_content_plm = []
                val_acc_history_content_sum = []
                avg_acc_l5_cov = []
                avg_acc_l5_plm = []
                avg_acc_l5_sum = []
                for i in range(len(sub_selected_list_label)):
                    selected_list_label_local  = sub_selected_list_label[i].reshape(1,-1) 
                    DNCON4_prediction_cov_local  = DNCON4_prediction_cov[i].reshape(1,-1) 
                    DNCON4_prediction_plm_local  = DNCON4_prediction_plm[i].reshape(1,-1) 
                    DNCON4_prediction_sum_local  = DNCON4_prediction_sum[i].reshape(1,-1) 

                    (a, b, c,avg_pc_l5_cov,avg_pc_l2_cov,avg_pc_1l_cov,avg_acc_l5_cov_,avg_acc_l2_cov,avg_acc_1l_cov) = evaluate_prediction_4(p1, DNCON4_prediction_cov_local, selected_list_label_local, 24)
                    val_content_cov = "%s\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key+'_D'+str(i),value,'COV',avg_pc_l5_cov,avg_pc_l2_cov,avg_pc_1l_cov,avg_acc_l5_cov_,avg_acc_l2_cov,avg_acc_1l_cov)
                    val_acc_history_content_cov.append(val_content_cov)
                    avg_acc_l5_cov.append(avg_acc_l5_cov_)
                    
                    (a, b, c,avg_pc_l5_plm,avg_pc_l2_plm,avg_pc_1l_plm,avg_acc_l5_plm_,avg_acc_l2_plm,avg_acc_1l_plm) = evaluate_prediction_4(p1, DNCON4_prediction_plm_local, selected_list_label_local, 24)
                    val_content_plm = "%s\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key+'_D'+str(i),value,'PLM',avg_pc_l5_plm,avg_pc_l2_plm,avg_pc_1l_plm,avg_acc_l5_plm_,avg_acc_l2_plm,avg_acc_1l_plm)
                    val_acc_history_content_plm.append(val_content_plm)
                    avg_acc_l5_plm.append(avg_acc_l5_plm_)
                    
                    (a, b, c,avg_pc_l5_sum,avg_pc_l2_sum,avg_pc_1l_sum,avg_acc_l5_sum_,avg_acc_l2_sum,avg_acc_1l_sum) = evaluate_prediction_4(p1, DNCON4_prediction_sum_local, selected_list_label_local, 24)
                    val_content_sum = "%s\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key+'_D'+str(i),value,'SUM',avg_pc_l5_sum,avg_pc_l2_sum,avg_pc_1l_sum,avg_acc_l5_sum_,avg_acc_l2_sum,avg_acc_1l_sum)
                    val_acc_history_content_sum.append(val_content_sum)
                    avg_acc_l5_sum.append(avg_acc_l5_sum_)
                    step_num += 1
            
            if type(val_acc_history_content_cov) == list:
                for i in range(len(val_acc_history_content_cov)):
                    print('The pred accuracy is ',val_acc_history_content_cov[i])  
                    print('The pred accuracy is ',val_acc_history_content_plm[i]) 
                    print('The pred accuracy is ',val_acc_history_content_sum[i])
            else:
                print('The pred accuracy is ',val_acc_history_content_cov)  
                print('The pred accuracy is ',val_acc_history_content_plm) 
                print('The pred accuracy is ',val_acc_history_content_sum)

            with open(pred_history_out, "a") as myfile:
                # print(type(val_acc_history_content_cov))
                if type(val_acc_history_content_cov) == list:
                    for i in range(len(val_acc_history_content_cov)):
                        myfile.write(val_acc_history_content_cov[i])
                        myfile.write(val_acc_history_content_plm[i])
                        myfile.write(val_acc_history_content_sum[i])
                else:
                    myfile.write(val_acc_history_content_cov)
                    myfile.write(val_acc_history_content_plm)
                    myfile.write(val_acc_history_content_sum)

            out_avg_pc_l5_cov += avg_pc_l5_cov 
            out_avg_pc_l2_cov += avg_pc_l2_cov 
            out_avg_pc_1l_cov += avg_pc_1l_cov 
            out_avg_acc_l5_cov += np.sum(avg_acc_l5_cov)
            out_avg_acc_l2_cov += avg_acc_l2_cov 
            out_avg_acc_1l_cov += avg_acc_1l_cov 
            out_avg_pc_l5_plm += avg_pc_l5_plm 
            out_avg_pc_l2_plm += avg_pc_l2_plm 
            out_avg_pc_1l_plm += avg_pc_1l_plm 
            out_avg_acc_l5_plm += np.sum(avg_acc_l5_plm) 
            out_avg_acc_l2_plm += avg_acc_l2_plm 
            out_avg_acc_1l_plm += avg_acc_1l_plm 
            out_avg_pc_l5_sum += avg_pc_l5_sum 
            out_avg_pc_l2_sum += avg_pc_l2_sum 
            out_avg_pc_1l_sum += avg_pc_1l_sum 
            out_avg_acc_l5_sum += np.sum(avg_acc_l5_sum) 
            out_avg_acc_l2_sum += avg_acc_l2_sum 
            out_avg_acc_1l_sum += avg_acc_1l_sum 

        if 'other' in feature_list:

            if list(key)[0] != 'T' or 'CASP12' in path_of_lists:  
                DNCON4_prediction_other = real_cmap_other.reshape(len(p1), Maximum_length * Maximum_length)
                (a, b, c,avg_pc_l5_other,avg_pc_l2_other,avg_pc_1l_other,avg_acc_l5_other,avg_acc_l2_other,avg_acc_1l_other) = evaluate_prediction_4(p1, DNCON4_prediction_other, selected_list_label, 24)
                val_acc_history_content_other = "%s\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,'OTHER',avg_pc_l5_other,avg_pc_l2_other,avg_pc_1l_other,avg_acc_l5_other,avg_acc_l2_other,avg_acc_1l_other)
                step_num += 1
            else:
                print("traget:%s" %(key), "sub_map_index:", sub_map_index, "sub_map_gap:", sub_map_gap)
                sub_selected_list_label = []
                DNCON4_prediction_other = []
                for i in range(len(selected_list_label)):
                    label_sub = selected_list_label[i].reshape(Maximum_length,Maximum_length)
                    left = sub_map_index[i][0]
                    right = sub_map_index[i][1] + 1

                    label_sub = label_sub[left:right,:]
                    label_sub = label_sub[:,left:right]
                    # real_cmap in shape L * L
                    real_cmap_other_copy = np.copy(real_cmap_other)
                    real_cmap_other_copy = real_cmap_other_copy[left:right,:]
                    real_cmap_other_copy = real_cmap_other_copy[:,left:right]
                    if np.sum(sub_map_gap[i]) < 1:
                        pass
                    else:
                        for j in range(len(sub_map_gap[i])):
                            sub_map_gap[i][j] = [x - sub_map_index[i][0] for x in sub_map_gap[i][j]]
                            gap_arrange = np.arange(sub_map_gap[i][j][0],sub_map_gap[i][j][1] + 1)
                            label_sub = np.delete(label_sub, gap_arrange, axis=0)
                            label_sub = np.delete(label_sub, gap_arrange, axis=1)
                            real_cmap_other_copy = np.delete(real_cmap_other_copy, gap_arrange, axis=0)
                            real_cmap_other_copy = np.delete(real_cmap_other_copy, gap_arrange, axis=1)
                    l = label_sub.shape[0]
                    # sub_label_cmap_file = "%s/true_%s_%d.txt" % (model_predict_sub, key, i)
                    # sub_pred_cmap_file = "%s/pred_%s_%d.txt" % (model_predict_sub, key, i)
                    # np.savetxt(sub_label_cmap_file, label_sub, fmt='%.4f')
                    # np.savetxt(sub_pred_cmap_file, real_cmap_other_copy, fmt='%.4f')

                    label_sub = label_sub.reshape((-1, l*l))
                    sub_selected_list_label.append(label_sub)
                    real_cmap_other_copy = real_cmap_other_copy.reshape((-1, l*l))
                    DNCON4_prediction_other.append(real_cmap_other_copy)

                val_acc_history_content_other = []            
                avg_acc_l5_other = []
                for i in range(len(sub_selected_list_label)): 
                    selected_list_label_local  = sub_selected_list_label[i].reshape(1,-1) 
                    DNCON4_prediction_other_local  = DNCON4_prediction_other[i].reshape(1,-1)   
                    (a, b, c,avg_pc_l5_other,avg_pc_l2_other,avg_pc_1l_other,avg_acc_l5_other_,avg_acc_l2_other,avg_acc_1l_other) = evaluate_prediction_4(p1, DNCON4_prediction_other_local, selected_list_label_local, 24)
                    val_content_other = "%s\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,'OTHER',avg_pc_l5_other,avg_pc_l2_other,avg_pc_1l_other,avg_acc_l5_other_,avg_acc_l2_other,avg_acc_1l_other)
                    val_acc_history_content_other.append(val_content_other)
                    avg_acc_l5_other.append(avg_acc_l5_other_)

                    step_num += 1
            
            if type(val_acc_history_content_other) == list:
                for i in range(len(val_acc_history_content_other)):
                    print('The pred accuracy is ',val_acc_history_content_other[i])  
            else:
                print('The pred accuracy is ',val_acc_history_content_other)  

            with open(pred_history_out, "a") as myfile:
                if type(val_acc_history_content_other) == list:
                    for i in range(len(val_acc_history_content_other)):
                        myfile.write(val_acc_history_content_other[i])
                else:
                    myfile.write(val_acc_history_content_other)

            out_avg_pc_l5_other += avg_pc_l5_other 
            out_avg_pc_l2_other += avg_pc_l2_other 
            out_avg_pc_1l_other += avg_pc_1l_other 
            out_avg_acc_l5_other += np.sum(avg_acc_l5_other)
            out_avg_acc_l2_other += avg_acc_l2_other 
            out_avg_acc_1l_other += avg_acc_1l_other 

    #     global_mse = 0.0
    #     weighted_mse = 0.0
    #     if predict_method == 'real_dist':
    #         selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
    #         global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_prediction, selected_list_label_dist)
    #         # to binary
    #         DNCON4_prediction = DNCON4_prediction * (DNCON4_prediction <= 8) 

    # #### The add of acc parameters
    #     out_gloable_mse += global_mse
    #     out_weighted_mse += weighted_mse 

    if only_predict_flag == True:
        print ("Predict map filepath: %s"%(model_predict))
        print ("END, Have Fun!\n")
    else:
        print ('step_num=', step_num)
        #### The out avg acc parameters
        if 'combine' in feature_list:
            all_num = step_num
            out_gloable_mse /= all_num
            out_weighted_mse /= all_num
            out_avg_pc_l5_cov /= all_num
            out_avg_pc_l2_cov /= all_num
            out_avg_pc_1l_cov /= all_num
            out_avg_acc_l5_cov /= all_num
            out_avg_acc_l2_cov /= all_num
            out_avg_acc_1l_cov /= all_num
            out_avg_pc_l5_plm /= all_num
            out_avg_pc_l2_plm /= all_num
            out_avg_pc_1l_plm /= all_num
            out_avg_acc_l5_plm /= all_num
            out_avg_acc_l2_plm /= all_num
            out_avg_acc_1l_plm /= all_num
            out_avg_pc_l5_sum /= all_num
            out_avg_pc_l2_sum /= all_num
            out_avg_pc_1l_sum /= all_num
            out_avg_acc_l5_sum /= all_num
            out_avg_acc_l2_sum /= all_num
            out_avg_acc_1l_sum /= all_num


            val_acc_history_content_cov = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (out_avg_pc_l5_cov,out_avg_pc_l2_cov,out_avg_pc_1l_cov,out_avg_acc_l5_cov,out_avg_acc_l2_cov,out_avg_acc_1l_cov)
            val_acc_history_content_plm = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (out_avg_pc_l5_plm,out_avg_pc_l2_plm,out_avg_pc_1l_plm,out_avg_acc_l5_plm,out_avg_acc_l2_plm,out_avg_acc_1l_plm)
            val_acc_history_content_sum = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (out_avg_pc_l5_sum,out_avg_pc_l2_sum,out_avg_pc_1l_sum,out_avg_acc_l5_sum,out_avg_acc_l2_sum,out_avg_acc_1l_sum)
            

            print('The validation accuracy is ',val_acc_history_content_cov)
            print('The validation accuracy is ',val_acc_history_content_plm)
            print('The validation accuracy is ',val_acc_history_content_sum)
            print ("Predict map filepath: %s"%(model_predict))
            print ("END, Have Fun!\n")

        if 'other' in feature_list:
            all_num = step_num
            out_gloable_mse /= all_num
            out_weighted_mse /= all_num
            out_avg_pc_l5_other /= all_num
            out_avg_pc_l2_other /= all_num
            out_avg_pc_1l_other /= all_num
            out_avg_acc_l5_other /= all_num
            out_avg_acc_l2_other /= all_num
            out_avg_acc_1l_other /= all_num

            val_acc_history_content_other = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (out_avg_pc_l5_other,out_avg_pc_l2_other,out_avg_pc_1l_other,out_avg_acc_l5_other,out_avg_acc_l2_other,out_avg_acc_1l_other)

            print('The validation accuracy is ',val_acc_history_content_other)
            print ("Predict map filepath: %s"%(model_predict))
            print ("END, Have Fun!\n")
        
    return out_avg_acc_l5_other,out_avg_acc_l2_other,out_avg_acc_1l_other

result_df = pd.DataFrame(columns=['Model','Precision_L5','Precision_L2','Precision_L1'])
weights_dir = CV_dir+'/model_weights/'
idx = 0
best_weight_file = CV_dir + '/model-train-weight-'+model_name+'-best-val.h5'
out_avg_acc_l5_other,out_avg_acc_l2_other,out_avg_acc_1l_other = evaluate_performance(best_weight_file)
result_df.loc[idx] = ['Best',out_avg_acc_l5_other,out_avg_acc_l2_other,out_avg_acc_1l_other]
print('Best model in validation:'+str(out_avg_acc_l5_other)+' '+str(out_avg_acc_l2_other)+' '+str(out_avg_acc_1l_other))
#idx = 1
#for i in os.listdir(weights_dir):
#    weight_file = weights_dir+i
#    print('use weights:    '+weight_file)
#    out_avg_acc_l5_other,out_avg_acc_l2_other,out_avg_acc_1l_other = evaluate_performance(weight_file)
#    result_df.loc[idx] = [i,out_avg_acc_l5_other,out_avg_acc_l2_other,out_avg_acc_1l_other]
#    idx += 1
result_df.to_csv(CV_dir+'/performance_all.csv',index=False)

