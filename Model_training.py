# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:47:26 2017

@author: Zhiye
"""
import os

from Model_construct import *

from DNCON_lib import *

from Model_construct import _weighted_binary_crossentropy, _weighted_categorical_crossentropy, _weighted_mean_squared_error
from contact_transformer import ContactTransformerV5, ContactTransformerV6,ContactTransformerV7, baselineModel,baselineModelSym,ContactTransformerV10,ContactTransformerV11

import numpy as np
import time
import shutil
import sys
import os
import platform
import gc
from collections import defaultdict
import pickle
from six.moves import range

import keras.backend as K
import tensorflow as tf
from keras.models import model_from_json,load_model, Sequential, Model
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Adagrad, Adamax, Nadam
from keras.utils import multi_gpu_model, Sequence
from keras.callbacks import ReduceLROnPlateau
from random import randint



def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)

class SequenceData(Sequence):
    """docstring for SequenceData"""
    def __init__(self, path_of_lists, path_of_X, path_of_Y, min_seq_sep,dist_string, batch_size, reject_fea_file='None', 
    dataset_select='train', if_use_binsize=False, predict_method='bin_class'):
        self.path_of_lists = path_of_lists
        self.path_of_X = path_of_X
        self.path_of_Y = path_of_Y
        self.min_seq_sep = min_seq_sep
        self.dist_string = dist_string
        self.batch_size = batch_size
        self.reject_fea_file = reject_fea_file
        self.dataset_select = dataset_select
        self.if_use_binsize = if_use_binsize
        self.predict_method = predict_method

        self.accept_list = []
        if self.reject_fea_file != 'None':
            with open(self.reject_fea_file) as f:
                for line in f:
                    if line.startswith('#'):
                        feature_name = line.strip()
                        feature_name = feature_name[0:]
                        self.accept_list.append(feature_name)

    def __len__(self):
        if (self.dataset_select == 'train'):
            self.dataset_list = build_dataset_dictionaries_train(self.path_of_lists)
        elif (self.dataset_select == 'vali'):
            self.dataset_list = build_dataset_dictionaries_test(self.path_of_lists)
        else:
            self.dataset_list = build_dataset_dictionaries_train(self.path_of_lists)
        training_dict = subset_pdb_dict(self.dataset_list, 0, 700, 7000, 'random') #can be random ordered   
        self.training_list = list(training_dict.keys())
        self.training_lens = list(training_dict.values())
        self.all_data_num = len(training_dict)
        return self.all_data_num

    def on_epoch_end(self):
        training_dict = subset_pdb_dict(self.dataset_list, 0, 700, 7000, 'random') #can be random ordered   
        self.training_list = list(training_dict.keys())
        self.training_lens = list(training_dict.values())

    def get_batch_XY(self, index):
        batch_X=[]
        batch_Y=[]
        batch_list = self.training_list[index * self.batch_size:(index + 1) * self.batch_size]
        batch_list_len = self.training_lens[index * self.batch_size:(index + 1) * self.batch_size]
        if self.if_use_binsize:
            max_pdb_lens = 320
        else:
            # print(batch_list_len)
            max_pdb_lens = max(batch_list_len)
        for i in range(0, len(batch_list)):
            pdb_name = batch_list[i]
            pdb_len = batch_list_len[i]
            notxt_flag = True
            featurefile = self.path_of_X + '/X-' + pdb_name + '.txt'
            if ((len(self.accept_list) == 1 and ('# cov' not in self.accept_list and '# plm' not in self.accept_list)) or 
                  (len(self.accept_list) == 2 and ('# cov' not in self.accept_list or '# plm' not in self.accept_list)) or (len(self.accept_list) > 2)):
                notxt_flag = False
                if not os.path.isfile(featurefile):
                    print("feature file not exists: ",featurefile, " pass!")
                    continue     
            cov = self.path_of_X + '/' + pdb_name + '.cov'
            if '# cov' in self.accept_list:
                if not os.path.isfile(cov):
                    print("Cov Matrix file not exists: ",cov, " pass!")
                    continue        
            plm = self.path_of_X + '/' + pdb_name + '.plm'
            if '# plm' in self.accept_list:
                if not os.path.isfile(plm):
                    print("plm matrix file not exists: ",plm, " pass!")
                    continue   
            pre = self.path_of_X + '/' + pdb_name + '.pre'
            if '# pre' in self.accept_list:
                if not os.path.isfile(pre):
                    # print("pre matrix file not exists: ",pre, " pass!")
                    continue
            netout = self.path_of_X + '/net_out/' + pdb_name + '.npy'
            if '# netout' in self.accept_list:      
                if not os.path.isfile(netout):
                    print("netout matrix file not exists: ",netout, " pass!")
                    continue 

            if self.predict_method == 'bin_class':       
                targetfile = self.path_of_Y + '/Y' + str(self.dist_string) + '-'+ pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif self.predict_method == 'mul_class':
                targetfile = self.path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue 
            elif self.predict_method == 'real_dist':
                targetfile = self.path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            else:
                targetfile = self.path_of_Y + '/Y' + str(self.dist_string) + '-'+ pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue

            (featuredata, feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, pre, netout, self.accept_list, pdb_len, notxt_flag)
            feature_2D_all = []
            for key in sorted(feature_index_all_dict.keys()):
                featurename = feature_index_all_dict[key]
                feature = featuredata[key]
                feature = np.asarray(feature)
                if feature.shape[0] == feature.shape[1]:
                    feature_2D_all.append(feature)
                else:
                    print("Wrong dimension")
            fea_len = feature_2D_all[0].shape[0]

            F = len(feature_2D_all)
            X = np.zeros((max_pdb_lens, max_pdb_lens, F))
            for m in range(0, F):
                X[0:fea_len, 0:fea_len, m] = feature_2D_all[m]

            # X = np.memmap(cov, dtype=np.float32, mode='r', shape=(F, max_pdb_lens, max_pdb_lens))
            # X = X.transpose(1, 2, 0)

            l_max = max_pdb_lens
            if self.predict_method == 'bin_class':
                Y = getY(targetfile, self.min_seq_sep, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
            elif self.predict_method == 'mul_class':
                print("Haven't has this function! quit!\n")
                sys.exit(1)
            elif self.predict_method == 'real_dist':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)              
            
            batch_X.append(X)
            batch_Y.append(Y)
        batch_X =  np.array(batch_X)
        batch_Y =  np.array(batch_Y)
        return  batch_X, batch_Y

    def __getitem__(self, index):

        batch_X, batch_Y = self.get_batch_XY(index)
        if len(batch_X.shape) < 4 or len(batch_Y.shape) < 4:
            print("Loading data have wrong shape! Reload!")
            index += 1
            batch_X, batch_Y = self.get_batch_XY(index)

        return batch_X, batch_Y
        
# dist_string = dist_stringinterval
def generate_data_from_file(path_of_lists, path_of_X, path_of_Y, min_seq_sep,dist_string, batch_size, reject_fea_file='None', 
    child_list_index=0, list_sep_flag=False, dataset_select='train', if_use_binsize=False, predict_method='bin_class', Maximum_length = 500):
    accept_list = []
    if reject_fea_file != 'None':
        with open(reject_fea_file) as f:
            for line in f:
                if line.startswith('#'):
                    feature_name = line.strip()
                    feature_name = feature_name[0:]
                    accept_list.append(feature_name)
    if (dataset_select == 'train'):
        dataset_list = build_dataset_dictionaries_train(path_of_lists)
    elif (dataset_select == 'vali'):
        dataset_list = build_dataset_dictionaries_test(path_of_lists)
    else:
        dataset_list = build_dataset_dictionaries_train(path_of_lists)

    if (list_sep_flag == False):
        training_dict = subset_pdb_dict(dataset_list, 0, Maximum_length, 7000, 'random') #can be random ordered   
        training_list = list(training_dict.keys())
        training_lens = list(training_dict.values())
        all_data_num = len(training_dict)
        loopcount = all_data_num // int(batch_size)
        # print('crop_list_num=',all_data_num)
        # print('crop_loopcount=',loopcount)
    else:
        training_dict = subset_pdb_dict(dataset_list, 0, Maximum_length, 7000, 'ordered') #can be random ordered
        all_training_list = list(training_dict.keys())
        all_training_lens = list(training_dict.values())
        if ((child_list_index + 1) * 15 > len(training_dict)):
            print("Out of list range!\n")
            child_list_index = len(training_dict)/15 - 1
        child_batch_list = all_training_list[child_list_index * 15:(child_list_index + 1) * 15]
        child_batch_list_len = all_training_lens[child_list_index * 15:(child_list_index + 1) * 15]
        all_data_num = 15
        loopcount = all_data_num // int(batch_size)
        print('crop_list_num=',all_data_num)
        print('crop_loopcount=',loopcount)
        training_list = child_batch_list
        training_lens = child_batch_list_len
    index = 0
    while(True):
        if index >= loopcount:
            raining_dict = subset_pdb_dict(dataset_list, 0, Maximum_length, 7000, 'random') #can be random ordered   
            training_list = list(training_dict.keys())
            training_lens = list(training_dict.values())
            index = 0
        batch_list = training_list[index * batch_size:(index + 1) * batch_size]
        batch_list_len = training_lens[index * batch_size:(index + 1) * batch_size]
        index += 1
        # print(index, end='\t')
        if if_use_binsize:
            max_pdb_lens = Maximum_length
        else:
            max_pdb_lens = max(batch_list_len)

        data_all_dict = dict()
        batch_X=[]
        batch_Y=[]
        for i in range(0, len(batch_list)):
            pdb_name = batch_list[i]
            pdb_len = batch_list_len[i]
            notxt_flag = True
            featurefile =path_of_X + '/other/' + 'X-'  + pdb_name + '.txt'
            if ((len(accept_list) == 1 and ('# cov' not in accept_list and '# plm' not in accept_list and '# pre' not in accept_list and '# netout' not in accept_list)) or 
                  (len(accept_list) == 2 and ('# cov' not in accept_list or '# plm' not in accept_list or '# pre' not in accept_list or '# netout' not in accept_list)) or (len(accept_list) > 2)):
                notxt_flag = False
                if not os.path.isfile(featurefile):
                    print("feature file not exists: ",featurefile, " pass!")
                    continue     
            cov = path_of_X + '/cov/' + pdb_name + '.cov'
            if '# cov' in accept_list:
                if not os.path.isfile(cov):
                    print("Cov Matrix file not exists: ",cov, " pass!")
                    continue        
            plm = path_of_X + '/plm/' + pdb_name + '.plm'
            if '# plm' in accept_list:
                if not os.path.isfile(plm):
                    print("plm matrix file not exists: ",plm, " pass!")
                    continue   
            pre = path_of_X + '/pre/' + pdb_name + '.pre'
            if '# pre' in accept_list:
                if not os.path.isfile(pre):
                    # print("pre matrix file not exists: ",pre, " pass!")
                    continue
            netout = path_of_X + '/net_out/' + pdb_name + '.npy'
            if '# netout' in accept_list:      
                if not os.path.isfile(netout):
                    print("netout matrix file not exists: ",netout, " pass!")
                    continue 

            if predict_method == 'bin_class':       
                targetfile = path_of_Y + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'mul_class':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue 
            elif predict_method == 'real_dist':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            else:
                targetfile = path_of_Y + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue

            (featuredata, feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, pre, netout, accept_list, pdb_len, notxt_flag)
            if featuredata == False or feature_index_all_dict == False:
                print("Bad alignment, Please check!\n")
                continue
            feature_2D_all = []
            for key in sorted(feature_index_all_dict.keys()):
                featurename = feature_index_all_dict[key]
                feature = featuredata[key]
                feature = np.asarray(feature)
                if feature.shape[0] == feature.shape[1]:
                    feature_2D_all.append(feature)
                else:
                    print("Wrong dimension")
            fea_len = feature_2D_all[0].shape[0]

            F = len(feature_2D_all)
            X = np.zeros((max_pdb_lens, max_pdb_lens, F))
            for m in range(0, F):
                X[0:fea_len, 0:fea_len, m] = feature_2D_all[m]

            # X = np.memmap(cov, dtype=np.float32, mode='r', shape=(F, max_pdb_lens, max_pdb_lens))
            # X = X.transpose(1, 2, 0)

            l_max = max_pdb_lens
            if predict_method == 'bin_class':
                Y = getY(targetfile, min_seq_sep, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
            elif predict_method == 'mul_class':
                print("Haven't has this function! quit!\n")
                sys.exit(1)
            elif predict_method == 'real_dist':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
                # real_dist is different with bin class, bin out is l*l vector, real dist out is (l,l) matrix
                # Y = getY_dist(targetfile, 0, l_max)
                # if (l_max != len(Y)):
                #     print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                #     print('len(Y) = %d, lmax = %d'%(len(Y), l_max))
                #     continue
            batch_X.append(X)
            batch_Y.append(Y)
            del X
            del Y
        batch_X =  np.array(batch_X)
        batch_Y =  np.array(batch_Y)
        # print('X shape\n', batch_X.shape)
        # print('Y shape', batch_Y.shape)
        if len(batch_X.shape) < 4 or len(batch_Y.shape) < 4:
            # print('Data shape error, pass!\n')
            continue
        yield batch_X, batch_Y
# dist_string = dist_stringinterval
def generate_data_from_file_mix(path_of_lists, path_of_X, path_of_Y, min_seq_sep,dist_string, batch_size, reject_fea_file='None', 
    child_list_index=0, list_sep_flag=False, dataset_select='train', if_use_binsize=False, predict_method='bin_class', Maximum_length = 500):
    accept_list1 = []
    accept_list2 = []
    if reject_fea_file != 'None':
        with open(reject_fea_file[0]) as f:
            for line in f:
                if line.startswith('#'):
                    feature_name = line.strip()
                    feature_name = feature_name[0:]
                    accept_list1.append(feature_name)
        with open(reject_fea_file[1]) as f:
            for line in f:
                if line.startswith('#'):
                    feature_name = line.strip()
                    feature_name = feature_name[0:]
                    accept_list2.append(feature_name)
    if (dataset_select == 'train'):
        dataset_list = build_dataset_dictionaries_train(path_of_lists)
    elif (dataset_select == 'vali'):
        dataset_list = build_dataset_dictionaries_test(path_of_lists)
    else:
        dataset_list = build_dataset_dictionaries_train(path_of_lists)

    if (list_sep_flag == False):
        training_dict = subset_pdb_dict(dataset_list, 0, Maximum_length, 7000, 'random') #can be random ordered   
        training_list = list(training_dict.keys())
        training_lens = list(training_dict.values())
        all_data_num = len(training_dict)
        loopcount = all_data_num // int(batch_size)
    else:
        training_dict = subset_pdb_dict(dataset_list, 0, Maximum_length, 7000, 'ordered') #can be random ordered
        all_training_list = list(training_dict.keys())
        all_training_lens = list(training_dict.values())
        if ((child_list_index + 1) * 15 > len(training_dict)):
            print("Out of list range!\n")
            child_list_index = len(training_dict)/15 - 1
        child_batch_list = all_training_list[child_list_index * 15:(child_list_index + 1) * 15]
        child_batch_list_len = all_training_lens[child_list_index * 15:(child_list_index + 1) * 15]
        all_data_num = 15
        loopcount = all_data_num // int(batch_size)
        print('crop_list_num=',all_data_num)
        print('crop_loopcount=',loopcount)
        training_list = child_batch_list
        training_lens = child_batch_list_len
    index = 0
    while(True):
        if index >= loopcount:
            raining_dict = subset_pdb_dict(dataset_list, 0, Maximum_length, 7000, 'random') #can be random ordered   
            training_list = list(training_dict.keys())
            training_lens = list(training_dict.values())
            index = 0
        batch_list = training_list[index * batch_size:(index + 1) * batch_size]
        batch_list_len = training_lens[index * batch_size:(index + 1) * batch_size]
        index += 1
        # print(index, end='\t')
        if if_use_binsize:
            max_pdb_lens = Maximum_length
        else:
            max_pdb_lens = max(batch_list_len)

        data_all_dict = dict()  
        batch_X1=[] 
        batch_X2=[]
        batch_Y=[]
        for i in range(0, len(batch_list)):
            pdb_name = batch_list[i]
            pdb_len = batch_list_len[i]
            notxt_flag1 = True
            notxt_flag2 = True
            featurefile =path_of_X + '/other/' + 'X-'  + pdb_name + '.txt'
            if ((len(accept_list1) == 1 and ('# cov' not in accept_list1 and '# plm' not in accept_list1 and '# pre' not in accept_list1)) or 
                  (len(accept_list1) == 2 and ('# cov' not in accept_list1 or '# plm' not in accept_list1 or '# pre' not in accept_list1)) or (len(accept_list1) > 2)):
                notxt_flag1 = False
                if not os.path.isfile(featurefile):
                    print("feature file not exists: ",featurefile, " pass!")
                    continue    
            if ((len(accept_list2) == 1 and ('# cov' not in accept_list2 and '# plm' not in accept_list2 and '# pre' not in accept_list2)) or 
                  (len(accept_list2) == 2 and ('# cov' not in accept_list2 or '# plm' not in accept_list2 or '# pre' not in accept_list2)) or (len(accept_list2) > 2)):
                notxt_flag2 = False
                if not os.path.isfile(featurefile):
                    print("feature file not exists: ",featurefile, " pass!")
                    continue  
            cov = path_of_X + '/cov/' + pdb_name + '.cov'
            if '# cov' in accept_list1:
                if not os.path.isfile(cov):
                    print("Cov Matrix file not exists: ",cov, " pass!")
                    continue        
            plm = path_of_X + '/plm/' + pdb_name + '.plm'
            if '# plm' in accept_list1:
                if not os.path.isfile(plm):
                    print("plm matrix file not exists: ",plm, " pass!")
                    continue   
            pre = path_of_X + '/pre/' + pdb_name + '.pre'
            if '# pre' in accept_list1:
                if not os.path.isfile(pre):
                    # print("pre matrix file not exists: ",pre, " pass!")
                    continue
            netout = path_of_X + '/net_out/' + pdb_name + '.npy'
            if '# netout' in accept_list1:      
                if not os.path.isfile(netout):
                    print("netout matrix file not exists: ",netout, " pass!")
                    continue 

            if predict_method == 'bin_class':       
                targetfile = path_of_Y + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'mul_class':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue 
            elif predict_method == 'real_dist':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            else:
                targetfile = path_of_Y + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue
            (featuredata, feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, pre, netout, accept_list1, pdb_len, notxt_flag1)
            if featuredata == False or feature_index_all_dict == False:
                print("Bad alignment, Please check!\n")
                continue
            feature_2D_all = []
            for key in sorted(feature_index_all_dict.keys()):
                featurename = feature_index_all_dict[key]
                feature = featuredata[key]
                feature = np.asarray(feature)
                if feature.shape[0] == feature.shape[1]:
                    feature_2D_all.append(feature)
                else:
                    print("Wrong dimension")
            fea_len = feature_2D_all[0].shape[0]
            F = len(feature_2D_all)
            X1 = np.zeros((max_pdb_lens, max_pdb_lens, F))
            for m in range(0, F):
                X1[0:fea_len, 0:fea_len, m] = feature_2D_all[m]

            (featuredata, feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, pre, netout, accept_list2, pdb_len, notxt_flag2)
            if featuredata == False or feature_index_all_dict == False:
                print("Bad alignment, Please check!\n")
                continue
            feature_2D_all = []
            for key in sorted(feature_index_all_dict.keys()):
                featurename = feature_index_all_dict[key]
                feature = featuredata[key]
                feature = np.asarray(feature)
                if feature.shape[0] == feature.shape[1]:
                    feature_2D_all.append(feature)
                else:
                    print("Wrong dimension")
            fea_len = feature_2D_all[0].shape[0]
            F = len(feature_2D_all)
            X2 = np.zeros((max_pdb_lens, max_pdb_lens, F))
            for m in range(0, F):
                X2[0:fea_len, 0:fea_len, m] = feature_2D_all[m]

            l_max = max_pdb_lens
            if predict_method == 'bin_class':
                Y = getY(targetfile, min_seq_sep, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
            elif predict_method == 'mul_class':
                print("Haven't has this function! quit!\n")
                sys.exit(1)
            elif predict_method == 'real_dist':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
            batch_X1.append(X1)
            batch_X2.append(X2)
            batch_Y.append(Y)
            del X1, X2, Y
        batch_X1 =  np.array(batch_X1)
        batch_X2 =  np.array(batch_X2)
        batch_Y =  np.array(batch_Y)
        if len(batch_X1.shape) < 4 or len(batch_X2.shape) < 4 or len(batch_Y.shape) < 4:
            # print('Data shape error, pass!\n')
            continue
        yield [batch_X1, batch_X2], batch_Y

def generate_data_from_file_mix_sym(path_of_lists, path_of_X, path_of_Y, min_seq_sep,dist_string, batch_size, reject_fea_file='None', 
    child_list_index=0, list_sep_flag=False, dataset_select='train', if_use_binsize=False, predict_method='bin_class', Maximum_length = 500):
    accept_list1 = []
    accept_list2 = []
    if reject_fea_file != 'None':
        with open(reject_fea_file[0]) as f:
            for line in f:
                if line.startswith('#'):
                    feature_name = line.strip()
                    feature_name = feature_name[0:]
                    accept_list1.append(feature_name)
        with open(reject_fea_file[1]) as f:
            for line in f:
                if line.startswith('#'):
                    feature_name = line.strip()
                    feature_name = feature_name[0:]
                    accept_list2.append(feature_name)
    if (dataset_select == 'train'):
        dataset_list = build_dataset_dictionaries_train(path_of_lists)
    elif (dataset_select == 'vali'):
        dataset_list = build_dataset_dictionaries_test(path_of_lists)
    else:
        dataset_list = build_dataset_dictionaries_train(path_of_lists)

    if (list_sep_flag == False):
        training_dict = subset_pdb_dict(dataset_list, 0, Maximum_length, 7000, 'random') #can be random ordered   
        training_list = list(training_dict.keys())
        training_lens = list(training_dict.values())
        all_data_num = len(training_dict)
        loopcount = all_data_num // int(batch_size)
    else:
        training_dict = subset_pdb_dict(dataset_list, 0, Maximum_length, 7000, 'ordered') #can be random ordered
        all_training_list = list(training_dict.keys())
        all_training_lens = list(training_dict.values())
        if ((child_list_index + 1) * 15 > len(training_dict)):
            print("Out of list range!\n")
            child_list_index = len(training_dict)/15 - 1
        child_batch_list = all_training_list[child_list_index * 15:(child_list_index + 1) * 15]
        child_batch_list_len = all_training_lens[child_list_index * 15:(child_list_index + 1) * 15]
        all_data_num = 15
        loopcount = all_data_num // int(batch_size)
        print('crop_list_num=',all_data_num)
        print('crop_loopcount=',loopcount)
        training_list = child_batch_list
        training_lens = child_batch_list_len
    index = 0
    while(True):
        if index >= loopcount:
            raining_dict = subset_pdb_dict(dataset_list, 0, Maximum_length, 7000, 'random') #can be random ordered   
            training_list = list(training_dict.keys())
            training_lens = list(training_dict.values())
            index = 0
        batch_list = training_list[index * batch_size:(index + 1) * batch_size]
        batch_list_len = training_lens[index * batch_size:(index + 1) * batch_size]
        index += 1
        # print(index, end='\t')
        if if_use_binsize:
            max_pdb_lens = Maximum_length
        else:
            max_pdb_lens = max(batch_list_len)

        data_all_dict = dict()  
        batch_X1=[] 
        batch_X2=[]
        batch_Y=[]
        for i in range(0, len(batch_list)):
            pdb_name = batch_list[i]
            pdb_len = batch_list_len[i]
            notxt_flag1 = True
            notxt_flag2 = True
            featurefile =path_of_X + '/other/' + 'X-'  + pdb_name + '.txt'
            if ((len(accept_list1) == 1 and ('# cov' not in accept_list1 and '# plm' not in accept_list1 and '# pre' not in accept_list1)) or 
                  (len(accept_list1) == 2 and ('# cov' not in accept_list1 or '# plm' not in accept_list1 or '# pre' not in accept_list1)) or (len(accept_list1) > 2)):
                notxt_flag1 = False
                if not os.path.isfile(featurefile):
                    print("feature file not exists: ",featurefile, " pass!")
                    continue    
            if ((len(accept_list2) == 1 and ('# cov' not in accept_list2 and '# plm' not in accept_list2 and '# pre' not in accept_list2)) or 
                  (len(accept_list2) == 2 and ('# cov' not in accept_list2 or '# plm' not in accept_list2 or '# pre' not in accept_list2)) or (len(accept_list2) > 2)):
                notxt_flag2 = False
                if not os.path.isfile(featurefile):
                    print("feature file not exists: ",featurefile, " pass!")
                    continue  
            cov = path_of_X + '/cov/' + pdb_name + '.cov'
            if '# cov' in accept_list1:
                if not os.path.isfile(cov):
                    print("Cov Matrix file not exists: ",cov, " pass!")
                    continue        
            plm = path_of_X + '/plm/' + pdb_name + '.plm'
            if '# plm' in accept_list1:
                if not os.path.isfile(plm):
                    print("plm matrix file not exists: ",plm, " pass!")
                    continue   
            pre = path_of_X + '/pre/' + pdb_name + '.pre'
            if '# pre' in accept_list1:
                if not os.path.isfile(pre):
                    # print("pre matrix file not exists: ",pre, " pass!")
                    continue
            netout = path_of_X + '/net_out/' + pdb_name + '.npy'
            if '# netout' in accept_list1:      
                if not os.path.isfile(netout):
                    print("netout matrix file not exists: ",netout, " pass!")
                    continue 

            if predict_method == 'bin_class':       
                targetfile = path_of_Y + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'mul_class':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue 
            elif predict_method == 'real_dist':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            else:
                targetfile = path_of_Y + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue
            (featuredata, feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, pre, netout, accept_list1, pdb_len, notxt_flag1)
            if featuredata == False or feature_index_all_dict == False:
                print("Bad alignment, Please check!\n")
                continue
            feature_2D_all = []
            for key in sorted(feature_index_all_dict.keys()):
                featurename = feature_index_all_dict[key]
                feature = featuredata[key]
                feature = np.asarray(feature)
                if feature.shape[0] == feature.shape[1]:
                    feature_2D_all.append(feature)
                else:
                    print("Wrong dimension")
            fea_len = feature_2D_all[0].shape[0]
            F = len(feature_2D_all)
            X1 = np.zeros((max_pdb_lens, max_pdb_lens, F))
            for m in range(0, F):
                X1[0:fea_len, 0:fea_len, m] = feature_2D_all[m]

            (featuredata, feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, pre, netout, accept_list2, pdb_len, notxt_flag2)
            if featuredata == False or feature_index_all_dict == False:
                print("Bad alignment, Please check!\n")
                continue
            feature_2D_all = []
            for key in sorted(feature_index_all_dict.keys()):
                featurename = feature_index_all_dict[key]
                feature = featuredata[key]
                feature = np.asarray(feature)
                if feature.shape[0] == feature.shape[1]:
                    feature_2D_all.append(feature)
                else:
                    print("Wrong dimension")
            fea_len = feature_2D_all[0].shape[0]
            F = len(feature_2D_all)
            X2 = np.zeros((max_pdb_lens, max_pdb_lens, F))
            for m in range(0, F):
                X2[0:fea_len, 0:fea_len, m] = feature_2D_all[m]

            l_max = max_pdb_lens
            if predict_method == 'bin_class':
                Y = getY(targetfile, min_seq_sep, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
            elif predict_method == 'mul_class':
                print("Haven't has this function! quit!\n")
                sys.exit(1)
            elif predict_method == 'real_dist':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
            batch_X1.append(X1)
            batch_X2.append(X2)
            batch_Y.append(Y)
            del X1, X2, Y
        batch_X1 =  np.array(batch_X1)
        batch_X2 =  np.array(batch_X2)
        batch_Y =  np.array(batch_Y)
        if len(batch_X1.shape) < 4 or len(batch_X2.shape) < 4 or len(batch_Y.shape) < 4:
            # print('Data shape error, pass!\n')
            continue
        yield [batch_X1, batch_X2], [batch_Y,np.zeros((batch_Y.shape[0],1))]
        
def generate_data_from_file_mix_general(path_of_lists, path_of_X, path_of_Y, min_seq_sep,dist_string, batch_size, reject_fea_file='None', 
    child_list_index=0, list_sep_flag=False, dataset_select='train', if_use_binsize=False, predict_method='bin_class', Maximum_length = 500):
    if len(reject_fea_file) < 2:
        print("please check the parameters: reject_fea_file number!\n")
        sys.exit(1)
    else:
        accept_list = []
        for num in range(len(reject_fea_file)):
            tmp_list = []
            with open(reject_fea_file[num]) as f:
                for line in f:
                    if line.startswith('#'):
                        feature_name = line.strip()
                        feature_name = feature_name[0:]
                        tmp_list.append(feature_name)
            accept_list.append(tmp_list)
        # print(accept_list)

    if (dataset_select == 'train'):
        dataset_list = build_dataset_dictionaries_train(path_of_lists)
    elif (dataset_select == 'vali'):
        dataset_list = build_dataset_dictionaries_test(path_of_lists)
    else:
        dataset_list = build_dataset_dictionaries_train(path_of_lists)
    if (list_sep_flag == False):
        training_dict = subset_pdb_dict(dataset_list, 0, Maximum_length, 7000, 'random') #can be random ordered   
        training_list = list(training_dict.keys())
        training_lens = list(training_dict.values())
        all_data_num = len(training_dict)
        loopcount = all_data_num // int(batch_size)
    else:
        training_dict = subset_pdb_dict(dataset_list, 0, Maximum_length, 7000, 'ordered') #can be random ordered
        all_training_list = list(training_dict.keys())
        all_training_lens = list(training_dict.values())
        if ((child_list_index + 1) * 15 > len(training_dict)):
            print("Out of list range!\n")
            child_list_index = len(training_dict)/15 - 1
        child_batch_list = all_training_list[child_list_index * 15:(child_list_index + 1) * 15]
        child_batch_list_len = all_training_lens[child_list_index * 15:(child_list_index + 1) * 15]
        all_data_num = 15
        loopcount = all_data_num // int(batch_size)
        print('crop_list_num=',all_data_num)
        print('crop_loopcount=',loopcount)
        training_list = child_batch_list
        training_lens = child_batch_list_len
    index = 0
    while(True):
        if index >= loopcount:
            raining_dict = subset_pdb_dict(dataset_list, 0, Maximum_length, 7000, 'random') #can be random ordered   
            training_list = list(training_dict.keys())
            training_lens = list(training_dict.values())
            index = 0
        batch_list = training_list[index * batch_size:(index + 1) * batch_size]
        batch_list_len = training_lens[index * batch_size:(index + 1) * batch_size]
        index += 1
        # print(index, end='\t')
        if if_use_binsize:
            max_pdb_lens = Maximum_length
        else:
            max_pdb_lens = max(batch_list_len)

        data_all_dict = dict()
        batch_X = []  
        batch_Y=[]
        Y_flag = 0
        for num in range(len(accept_list)):
            batch_X_tmp=[] 
            for i in range(0, len(batch_list)):
                pdb_name = batch_list[i]
                pdb_len = batch_list_len[i]
                notxt_flag = True
                featurefile =path_of_X + '/other/' + 'X-'  + pdb_name + '.txt'
                if ((len(accept_list[num]) == 1 and ('# cov' not in accept_list[num] and '# plm' not in accept_list[num] and '# pre' not in accept_list[num])) or 
                      (len(accept_list[num]) == 2 and ('# cov' not in accept_list[num] or '# plm' not in accept_list[num] or '# pre' not in accept_list[num])) or (len(accept_list[num]) > 2)):
                    notxt_flag = False
                    if not os.path.isfile(featurefile):
                        print("feature file not exists: ",featurefile, " pass!")
                        continue    
                cov = path_of_X + '/cov/' + pdb_name + '.cov'
                if '# cov' in accept_list[num]:
                    if not os.path.isfile(cov):
                        print("Cov Matrix file not exists: ",cov, " pass!")
                        continue        
                plm = path_of_X + '/plm/' + pdb_name + '.plm'
                if '# plm' in accept_list[num]:
                    if not os.path.isfile(plm):
                        print("plm matrix file not exists: ",plm, " pass!")
                        continue   
                pre = path_of_X + '/pre/' + pdb_name + '.pre'
                if '# pre' in accept_list[num]:
                    if not os.path.isfile(pre):
                        # print("pre matrix file not exists: ",pre, " pass!")
                        continue
                netout = path_of_X + '/net_out/' + pdb_name + '.npy'
                if '# netout' in accept_list[num]:      
                    if not os.path.isfile(netout):
                        print("netout matrix file not exists: ",netout, " pass!")
                        continue 

                if predict_method == 'bin_class':       
                    targetfile = path_of_Y + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
                    if not os.path.isfile(targetfile):
                            print("target file not exists: ",targetfile, " pass!")
                            continue  
                elif predict_method == 'mul_class':
                    targetfile = path_of_Y + pdb_name + '.txt'
                    if not os.path.isfile(targetfile):
                            print("target file not exists: ",targetfile, " pass!")
                            continue 
                elif predict_method == 'real_dist':
                    targetfile = path_of_Y + pdb_name + '.txt'
                    if not os.path.isfile(targetfile):
                            print("target file not exists: ",targetfile, " pass!")
                            continue  
                else:
                    targetfile = path_of_Y + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
                    if not os.path.isfile(targetfile):
                            print("target file not exists: ",targetfile, " pass!")
                            continue
                (featuredata, feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, pre, netout, accept_list[num], pdb_len, notxt_flag)
                if featuredata == False or feature_index_all_dict == False:
                    print("Bad alignment, Please check!\n")
                    continue
                feature_2D_all = []
                for key in sorted(feature_index_all_dict.keys()):
                    featurename = feature_index_all_dict[key]
                    feature = featuredata[key]
                    feature = np.asarray(feature)
                    if feature.shape[0] == feature.shape[1]:
                        feature_2D_all.append(feature)
                    else:
                        print("Wrong dimension")
                fea_len = feature_2D_all[0].shape[0]
                F = len(feature_2D_all)
                X1 = np.zeros((max_pdb_lens, max_pdb_lens, F))
                for m in range(0, F):
                    X1[0:fea_len, 0:fea_len, m] = feature_2D_all[m]

                if Y_flag == 0:
                    Y_flag = 1
                    l_max = max_pdb_lens
                    if predict_method == 'bin_class':
                        Y = getY(targetfile, min_seq_sep, l_max)
                        if (l_max * l_max != len(Y)):
                            print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                            continue
                        Y = Y.reshape(l_max, l_max, 1)
                    elif predict_method == 'mul_class':
                        print("Haven't has this function! quit!\n")
                        sys.exit(1)
                    elif predict_method == 'real_dist':
                        Y = getY(targetfile, 0, l_max)
                        if (l_max * l_max != len(Y)):
                            print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                            continue
                        Y = Y.reshape(l_max, l_max, 1)
                    batch_Y.append(Y)
                batch_X_tmp.append(X1)
            batch_X_tmp =  np.array(batch_X_tmp)
            # print(batch_X_tmp.shape)
            batch_X.append(batch_X_tmp)
            del batch_X_tmp, X1
        # print(len(batch_X))
        # batch_X =  np.array(batch_X)
        batch_Y =  np.array(batch_Y)
        # print(batch_Y.shape)
        # default cov, plm, pre, other
        if len(batch_X) == 2:
            if len(batch_X[0].shape) < 4 or len(batch_X[1].shape) < 4 or len(batch_Y.shape) < 4:
                print('Data shape error, pass!\n')
                continue
            yield [batch_X[0], batch_X[1]], batch_Y
        elif len(batch_X) == 3:
            if len(batch_X[0].shape) < 4 or len(batch_X[1].shape) < 4 or len(batch_X[2].shape) < 4 or len(batch_Y.shape) < 4:
                print('Data shape error, pass!\n')
                continue
            yield [batch_X[0], batch_X[1], batch_X[2]], batch_Y
        elif len(batch_X) == 4:
            if len(batch_X[0].shape) < 4 or len(batch_X[1].shape) or len(batch_X[2].shape) or len(batch_X[3].shape) < 4 or len(batch_Y.shape) < 4:
                print('Data shape error, pass!\n')
                continue
            yield [batch_X[0], batch_X[1], batch_X[2], batch_X[3]], batch_Y
        else:
            print("Too much accept_list, please check!\n")
            sys.exit(1)

def DNCON4_1d2dconv_train_win_filter_layer_opt_fast_2D_generator(feature_num,CV_dir,feature_dir,model_prefix,
    epoch_outside,epoch_inside,epoch_rerun,interval_len,seq_end,win_array,use_bias,hidden_type,nb_filters,nb_layers,opt,
    lib_dir, batch_size_train,path_of_lists, path_of_Y, path_of_X, Maximum_length,dist_string, reject_fea_file='None',
    initializer = "he_normal", loss_function = "binary_crossentropy", weight_p=1.0, weight_n=1.0,  list_sep_flag=False,  if_use_binsize = False,
    att_config = 0 ,kmax = 7 ,att_outdim = 16,insert_pos = 'none'): 

    start=0
    end=seq_end
    import numpy as np
    Train_data_keys = dict()
    Train_targets_keys = dict()
    print("\n######################################\n佛祖保佑，永不迨机，永无bug，精度九十九\n######################################\n")
    feature_2D_num=feature_num # the number of features for each residue
 
    print("Load feature number", feature_2D_num)
    ### Define the model 
    model_out= "%s/model-train-%s.json" % (CV_dir,model_prefix)
    model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir,model_prefix)
    model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir,model_prefix)

    # opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)#1
    # opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
    # opt = SGD(lr=0.001, momentum=0.9, decay=0.00, nesterov=False)
    # opt = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06, decay=0.0)
    # opt = Adagrad(lr=0.01, epsilon=1e-06)
    # opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    if model_prefix == 'DNCON4_2dCONV':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#0.00
        DNCON4 = DeepConv_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    elif model_prefix == 'DNCON4_2dRES':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)#0.001  decay=0.0
        DNCON4 = DeepResnet_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    elif model_prefix == 'DNCON4_2dUNET':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)#0.001  decay=0.0
        DNCON4 = DeepUnet_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    elif model_prefix == 'DNCON4_2dWEIGHT':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)#0.001  decay=0.0
        DNCON4 = WeightLearn_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    elif model_prefix == 'DNCON4_2dDIARES':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
        DNCON4 = DilatedRes_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    elif model_prefix == 'DNCON4_2dDIARES_MIX':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
        DNCON4 = DilatedRes_with_paras_2D_Mix(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    elif model_prefix == 'DNCON4_2dDIARES_MIXG':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
        DNCON4 = DilatedRes_with_paras_2D_Mix_general(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    elif model_prefix == 'DNCON4_2dINCEPRES':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
        DNCON4 = IncepResV2_with_paras_2D(win_array, feature_2D_num, nb_filters, nb_layers, opt, initializer, loss_function, weight_p, weight_n)
    elif model_prefix == 'DNCON4_2dGRES':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
        DNCON4 = GoogleRes_with_paras_2D(win_array, feature_2D_num, nb_filters, nb_layers, opt, initializer, loss_function, weight_p, weight_n)
    elif model_prefix == 'DNCON4_2dCONRES':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
        DNCON4 = ConRes_with_paras_2D(win_array, feature_2D_num, nb_filters, nb_layers, opt, initializer, loss_function, weight_p, weight_n)
    elif model_prefix == 'DNCON4_2dDPN':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)#0.001  decay=0.0
        DNCON4 = DualpathNet_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    elif model_prefix == 'channel_attention':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
        DNCON4 = channel_attention(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,\
                                   nb_layers,opt,initializer,loss_function,weight_p,weight_n, \
                                   att_config,kmax,att_outdim,insert_pos)
    elif model_prefix == 'regional_attention_ori':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
        DNCON4 = regional_attention(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,\
                                   nb_layers,opt,initializer,loss_function,weight_p,weight_n, \
                                   att_config,att_outdim,insert_pos)   #att_config is region_size  
    elif model_prefix == 'regional_attention':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
        DNCON4 = regional_attention3D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,\
                                   nb_layers,opt,initializer,loss_function,weight_p,weight_n, \
                                   att_config,kmax,att_outdim,insert_pos)   #att_config is region_size 
    elif model_prefix == 'sequence_attention':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
        DNCON4 = sequence_attention(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,\
                                   nb_layers,opt,initializer,loss_function,weight_p,weight_n,att_config, \
                                   kmax,att_outdim,insert_pos)   
    elif model_prefix == 'baselineModel':
        DNCON4 = baselineModel(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,\
                               nb_layers,opt,initializer,loss_function,weight_p,weight_n,att_config, \
                               kmax,att_outdim,insert_pos)
    else:
        print('Undefined model.')
        sys.exit()

#    model_json = DNCON4.to_json()
    print("Saved model to disk")
    import pandas as pd
    model_parameters = pd.DataFrame((win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,
     initializer,loss_function,weight_p,weight_n,att_config,kmax,att_outdim,insert_pos))
    model_parameters.to_csv(model_out+'.txt')
#    with open(model_out, "w") as json_file:
#        json_file.write(model_json)
    print("Saved model to "+model_out+'.txt')
    rerun_flag=0
    # with tf.device("/cpu:0"):
    #     DNCON4 = multi_gpu_model(DNCON4, gpus=2)
    train_acc_history_out = "%s/training.acc_history" % (CV_dir)
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    best_val_acc_out = "%s/best_validation.acc_history" % (CV_dir)
    if os.path.exists(model_weight_out_best):
        print("######## Loading existing weights ",model_weight_out_best)
        DNCON4.load_weights(model_weight_out_best)
        rerun_flag = 1
    else:
        print("######## Setting initial weights")   
    
        chkdirs(train_acc_history_out)     
        with open(train_acc_history_out, "a") as myfile:
          myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\n")
          
        chkdirs(val_acc_history_out)     
        with open(val_acc_history_out, "a") as myfile:
          myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\n")
        
        chkdirs(best_val_acc_out)     
        with open(best_val_acc_out, "a") as myfile:
          myfile.write("Seq_Name\tSeq_Length\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\n")

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

    DNCON4.compile(loss=loss_function, metrics=['acc'], optimizer=opt)

    model_weight_epochs = "%s/model_weights/"%(CV_dir)
    model_weights_top = "%s/model_weights_top/"%(CV_dir)
    model_predict= "%s/predict_map/"%(CV_dir)
    model_val_acc= "%s/val_acc_inepoch/"%(CV_dir)
    chkdirs(model_weight_epochs)
    chkdirs(model_weights_top)
    chkdirs(model_predict)
    chkdirs(model_val_acc)

    tr_l = build_dataset_dictionaries_train(path_of_lists)
    tr_l_dict = subset_pdb_dict(tr_l, 0, Maximum_length, 7000, 'ordered')
    te_l = build_dataset_dictionaries_test(path_of_lists)
    all_l = te_l.copy()
    train_data_num = len(tr_l_dict)
    child_list_num = int(train_data_num/15)# 15 is the inter
    print('Total Number of Training dataset = ',str(len(tr_l_dict)))

    # callbacks=[reduce_lr]
    train_avg_acc_l5_best = 0 
    val_avg_acc_l5_best = 0
    min_seq_sep = 5
    lr_decay = False
    train_loss_last = 1e32
    train_loss_list = []
    evalu_loss_list = []
    # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
    #     def __init__(self, path_of_lists, path_of_X, path_of_Y, min_seq_sep,dist_string, batch_size, reject_fea_file='None', 
    # dataset_select='train', if_use_binsize=False, predict_method='bin_class'):
    # train_data_sequence = SequenceData(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, reject_fea_file, if_use_binsize=if_use_binsize, predict_method=predict_method)
    for epoch in range(epoch_rerun,epoch_outside):
        if (epoch >=30 and lr_decay == False):
            print("Setting lr_decay as true")
            lr_decay = True
            opt = SGD(lr=0.01, momentum=0.9, decay=0.00, nesterov=False)#0.001
            DNCON4.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)
            # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose = 1, min_delta=0.005, min_lr=0.00005)
        # class_weight = {0:1.,1:60.}
        if isinstance(reject_fea_file, str) == True:
            print("\n############ Running epoch ", epoch)
            if epoch == 0 and rerun_flag == 0:
                first_inepoch = 1
                history = DNCON4.fit_generator(generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, reject_fea_file, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length), 
                steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = first_inepoch, max_queue_size=20, workers=2, use_multiprocessing=False)         
                train_loss_list.append(history.history['loss'][first_inepoch-1])
            else: 
                history = DNCON4.fit_generator(generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, reject_fea_file, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length), 
                    steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = 1, max_queue_size=20, workers=2, use_multiprocessing=False)         
                train_loss_list.append(history.history['loss'][0])
        elif len(reject_fea_file) == 2:
            print("\n############ Running epoch ", epoch)
            if epoch == 0 and rerun_flag == 0:
                first_inepoch = 1
                if model_prefix == 'baselineModelSym':
                    history = DNCON4.fit_generator(generate_data_from_file_mix_sym(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, reject_fea_file, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length), 
                    steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = first_inepoch, max_queue_size=20, workers=2, use_multiprocessing=False)         
                else:
                    history = DNCON4.fit_generator(generate_data_from_file_mix(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, reject_fea_file, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length), 
                    steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = first_inepoch, max_queue_size=20, workers=2, use_multiprocessing=False)         
                train_loss_list.append(history.history['loss'][first_inepoch-1])
            else: 
                if model_prefix == 'baselineModelSym':
                    history = DNCON4.fit_generator(generate_data_from_file_mix_sym(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, reject_fea_file, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length), 
                    steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = 1, max_queue_size=20, workers=2, use_multiprocessing=False)         
                else:               
                    history = DNCON4.fit_generator(generate_data_from_file_mix(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, reject_fea_file, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length), 
                    steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = 1, max_queue_size=20, workers=2, use_multiprocessing=False)         
                train_loss_list.append(history.history['loss'][0])
        elif len(reject_fea_file) > 2:
            print("\n############ Running epoch ", epoch)
            if epoch == 0 and rerun_flag == 0:
                first_inepoch = 1
                history = DNCON4.fit_generator(generate_data_from_file_mix_general(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, reject_fea_file, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length), 
                steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = first_inepoch, max_queue_size=20, workers=2, use_multiprocessing=False)         
                train_loss_list.append(history.history['loss'][first_inepoch-1])
            else: 
                history = DNCON4.fit_generator(generate_data_from_file_mix_general(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, reject_fea_file, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length), 
                    steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = 1, max_queue_size=20, workers=2, use_multiprocessing=False)         
                train_loss_list.append(history.history['loss'][0])

        DNCON4.save_weights(model_weight_out)

        # DNCON4.save(model_and_weights)
        
        ### save models
        model_weight_out_inepoch = "%s/model-train-weight-%s-epoch%i.h5" % (model_weight_epochs,model_prefix,epoch)
        DNCON4.save_weights(model_weight_out_inepoch)
        ##### running validation

        print("Now evaluate for epoch ",epoch)
        val_acc_out_inepoch = "%s/validation_epoch%i.acc_history" % (model_val_acc, epoch) 
        sys.stdout.flush()
        selected_list = subset_pdb_dict(te_l,   0, Maximum_length, 7000, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
        print('Loading data sets ..',end='')

        testdata_len_range=50
        step_num = 0
        out_avg_pc_l5 = 0.0
        out_avg_pc_l2 = 0.0
        out_avg_pc_1l = 0.0
        out_avg_acc_l5 = 0.0
        out_avg_acc_l2 = 0.0
        out_avg_acc_1l = 0.0
        out_gloable_mse = 0.0
        out_weighted_mse = 0.0
        for key in selected_list:
            value = selected_list[key]
            p1 = {key: value}
            if if_use_binsize:
                length = Maximum_length
            else:
                length = value
            print(len(p1))
            if len(p1) < 1:
                continue
            print("start predict")
            if len(reject_fea_file)!=2:
                selected_list_2D = get_x_2D_from_this_list(p1, path_of_X, length,dist_string, reject_fea_file, value)
                if type(selected_list_2D) == bool:
                    continue
                print("selected_list_2D.shape: ",selected_list_2D.shape)
                print('Loading label sets..')
                selected_list_label = get_y_from_this_list(p1, path_of_Y_evalu, 0, length, dist_string)# dist_string 80
                DNCON4.load_weights(model_weight_out)

                DNCON4_prediction = DNCON4.predict([selected_list_2D], batch_size= 1)

            elif len(reject_fea_file)==2:
                temp1 = get_x_2D_from_this_list(p1, path_of_X, length,dist_string, reject_fea_file[0], value)
                temp2 = get_x_2D_from_this_list(p1, path_of_X, length,dist_string, reject_fea_file[1], value)
                if type(temp1) == bool or type(temp2) == bool:
                    continue
                print("selected_list_2D.shape: ",temp1.shape)
                print("selected_list_2D.shape: ",temp2.shape)
                print('Loading label sets..')
                selected_list_label = get_y_from_this_list(p1, path_of_Y_evalu, 0, length, dist_string)# dist_string 80
                DNCON4.load_weights(model_weight_out)

                DNCON4_prediction = DNCON4.predict([temp1, temp2], batch_size= 1)
                if model_prefix == 'baselineModelSym':
                    DNCON4_prediction = DNCON4_prediction[0]
            elif len(reject_fea_file)==3:
                temp1 = get_x_2D_from_this_list(p1, path_of_X, length,dist_string, reject_fea_file[0], value)
                temp2 = get_x_2D_from_this_list(p1, path_of_X, length,dist_string, reject_fea_file[1], value)
                temp3 = get_x_2D_from_this_list(p1, path_of_X, length,dist_string, reject_fea_file[2], value)
                if type(temp1) == bool or type(temp2) == bool or type(temp2) == bool:
                    continue
                print("selected_list_2D.shape: ",temp1.shape)
                print("selected_list_2D.shape: ",temp2.shape)
                print("selected_list_2D.shape: ",temp3.shape)
                print('Loading label sets..')
                selected_list_label = get_y_from_this_list(p1, path_of_Y_evalu, 0, length, dist_string)# dist_string 80
                DNCON4.load_weights(model_weight_out)

                DNCON4_prediction = DNCON4.predict([temp1, temp2, temp3], batch_size= 1)
            elif len(reject_fea_file)>=4:
                pred_temp = []
                bool_flag = False
                for fea_num in range(len(reject_fea_file)):
                    temp = get_x_2D_from_this_list(p1, path_of_X, length,dist_string, reject_fea_file[fea_num], value)
                    print("selected_list_2D.shape: ",temp.shape)
                    if type(temp) == bool:
                        bool_flag= True
                    pred_temp.append(temp)
                if bool_flag == True:
                    continue
                else:
                    print('Loading label sets..')
                    selected_list_label = get_y_from_this_list(p1, path_of_Y_evalu, 0, length, dist_string)# dist_string 80
                    DNCON4.load_weights(model_weight_out)
                    DNCON4_prediction = DNCON4.predict(pred_temp, batch_size= 1)

            CMAP = DNCON4_prediction.reshape(length, length)
            Map_UpTrans = np.triu(CMAP, 1).T
            Map_UandL = np.triu(CMAP)
            real_cmap = Map_UandL + Map_UpTrans

            DNCON4_prediction = real_cmap.reshape(len(p1), length*length)

            global_mse = 0.0
            weighted_mse = 0.0
            if predict_method == 'real_dist':
                selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, length, dist_string, lable_type = 'real')# dist_string 80
                global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_prediction, selected_list_label_dist)
                # to binary
                DNCON4_prediction = DNCON4_prediction * (DNCON4_prediction <= 8) 

            (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l) = evaluate_prediction_4(p1, DNCON4_prediction, selected_list_label, 24)
            val_acc_history_content = "%s\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l)
            print('The best validation accuracy is ',val_acc_history_content)
            with open(val_acc_out_inepoch, "a") as myfile:
                myfile.write(val_acc_history_content)

            out_gloable_mse += global_mse
            out_weighted_mse += weighted_mse 
            out_avg_pc_l5 += avg_pc_l5 * len(p1)
            out_avg_pc_l2 += avg_pc_l2 * len(p1)
            out_avg_pc_1l += avg_pc_1l * len(p1)
            out_avg_acc_l5 += avg_acc_l5 * len(p1)
            out_avg_acc_l2 += avg_acc_l2 * len(p1)
            out_avg_acc_1l += avg_acc_1l * len(p1)
            
            step_num += 1
        print ('step_num=', step_num)
        all_num = len(selected_list)
        out_gloable_mse /= all_num
        out_weighted_mse /= all_num
        out_avg_pc_l5 /= all_num
        out_avg_pc_l2 /= all_num
        out_avg_pc_1l /= all_num
        out_avg_acc_l5 /= all_num
        out_avg_acc_l2 /= all_num
        out_avg_acc_1l /= all_num
        val_acc_history_content = "%i\t%i\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (interval_len,epoch,epoch_inside,out_avg_pc_l5,out_avg_pc_l2,out_avg_pc_1l,
            out_avg_acc_l5,out_avg_acc_l2,out_avg_acc_1l, out_gloable_mse, train_loss_list[-1])
        with open(val_acc_history_out, "a") as myfile:
                    myfile.write(val_acc_history_content)  

        print('The validation accuracy is ',val_acc_history_content)
        if out_avg_acc_l5 >= val_avg_acc_l5_best:
            val_avg_acc_l5_best = out_avg_acc_l5 
            score_imed = "Accuracy L5 of Val: %.4f\t\n" % (val_avg_acc_l5_best)
            print("Saved best weight to disk, ", score_imed)
            DNCON4.save_weights(model_weight_out_best)

        train_loss = history.history['loss'][0]
        if (lr_decay and epoch > 30):
            current_lr = K.get_value(DNCON4.optimizer.lr)
            print("Current learning rate is {} ...".format(current_lr))
            if (epoch % 20 == 0):
                K.set_value(DNCON4.optimizer.lr, current_lr * 0.1)
                print("Decreasing learning rate to {} ...".format(current_lr * 0.1))
                # DNCON4.load_weights(model_weight_out_best)
            # if (train_loss < train_loss_last and current_lr < 0.01):
            #     K.set_value(DNCON4.optimizer.lr, current_lr * 1.1)
            #     print("Increasing learning rate to {} ...".format(current_lr * 1.1))
            # else:
            #     K.set_value(DNCON4.optimizer.lr, current_lr * 0.8)
            #     print("Decreasing learning rate to {} ...".format(current_lr * 0.8))
        train_loss_last = train_loss

        if epoch == epoch_outside-1:
            for key in selected_list:
                print('saving cmap of %s\n'%(key))
                value = selected_list[key]
                p1={key:value}
                if if_use_binsize:
                    length = Maximum_length
                else:
                    length = value
                print("start predict")
                if len(reject_fea_file)!=2:
                    selected_list_2D = get_x_2D_from_this_list(p1, path_of_X, length,dist_string, reject_fea_file, value)
                    if type(selected_list_2D) == bool:
                        continue
                    print("selected_list_2D.shape: ",selected_list_2D.shape)
                    print('Loading label sets..')
                    selected_list_label = get_y_from_this_list(p1, path_of_Y_evalu, 0, length, dist_string)# dist_string 80
                    DNCON4.load_weights(model_weight_out)

                    DNCON4_prediction = DNCON4.predict([selected_list_2D], batch_size= 1)
                elif len(reject_fea_file)==2:
                    temp1 = get_x_2D_from_this_list(p1, path_of_X, length,dist_string, reject_fea_file[0], value)
                    temp2 = get_x_2D_from_this_list(p1, path_of_X, length,dist_string, reject_fea_file[1], value)
                    if type(temp1) == bool or type(temp2) == bool:
                        continue
                    print("selected_list_2D.shape: ",temp1.shape)
                    print("selected_list_2D.shape: ",temp2.shape)
                    print('Loading label sets..')
                    selected_list_label = get_y_from_this_list(p1, path_of_Y_evalu, 0, length, dist_string)# dist_string 80
                    DNCON4.load_weights(model_weight_out)

                    DNCON4_prediction = DNCON4.predict([temp1, temp2], batch_size= 1)
                    if model_prefix == 'baselineModelSym':
                        DNCON4_prediction = DNCON4_prediction[0]
                CMAP = DNCON4_prediction.reshape(length, length)
                Map_UpTrans = np.triu(CMAP, 1).T
                Map_UandL = np.triu(CMAP)
                real_cmap = Map_UandL + Map_UpTrans

                global_mse = 0.0
                weighted_mse = 0.0
                if predict_method == 'real_dist':
                    selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, length, dist_string, lable_type = 'real')# dist_string 80
                    global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_prediction, selected_list_label_dist)
                    # to binary
                    DNCON4_prediction = DNCON4_prediction * (DNCON4_prediction <= 8) 

                DNCON4_pred = DNCON4_prediction.reshape(len(p1), length*length)
                (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l) = evaluate_prediction_4(p1, DNCON4_pred, selected_list_label, 24)
                val_acc_history_content = "%s\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l)
                print('The best validation accuracy is ',val_acc_history_content)
                with open(best_val_acc_out, "a") as myfile:
                    myfile.write(val_acc_history_content)  
                DNCON4_prediction = DNCON4_prediction.reshape (length, length)
                # cmap_file = "%s/%s.txt" % (model_predict,key)
                # np.savetxt(cmap_file, DNCON4_CNN_prediction, fmt='%.4f')
                cmap_file = "%s/%s.txt" % (model_predict,key)
                np.savetxt(cmap_file, real_cmap, fmt='%.4f')
                # history_loss_file = CV_dir+"/train_loss.history"


        print("Train loss history:", train_loss_list)
        # print("Validation loss history:", evalu_loss_list)
        #clear memory
        # K.clear_session()
        # tf.reset_default_graph()
    #select top10 models
    epochs = []
    accL5s = []
    with open(val_acc_history_out) as f:
        for line in f:
            cols = line.strip().split()
            if cols[0] != '150':
                continue
            else:
                epoch = cols[1]
                accL5 = cols[6]
                epochs.append(cols[1])
                accL5s.append(cols[6])
                # print(epoch, accL5)
    accL5_sort = accL5s.copy()
    accL5_sort.sort(reverse=True)
    accL5_top = accL5_sort[0:5]
    epoch_top = []
    for index in range(len(accL5_top)):
        acc_find = accL5_top[index]
        pos_find = [i for i, v in enumerate(accL5s) if v == acc_find]
        # print(pos_find)
        for num in range(len(pos_find)):
            epoch_top.append(epochs[pos_find[num]])
    epoch_top = list(set(epoch_top))
    for index in range(len(epoch_top)):
        model_weight = "model-train-weight-%s-epoch%i.h5" % (model_prefix,int(epoch_top[index]))
        src_file = os.path.join(model_weight_epochs,model_weight)
        dst_file = os.path.join(model_weights_top,model_weight)
        shutil.copyfile(src_file,dst_file)
        print("Copy %s to model_weights_top"%epoch_top[index])
    print("Training finished, best validation acc = ",val_avg_acc_l5_best)
    return val_avg_acc_l5_best

def DNCON4_train_2D_generator(feature_num,CV_dir,feature_dir, model_prefix, epoch_outside,epoch_inside,epoch_rerun,interval_len,win_array,
    use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train,path_of_lists, path_of_Y, path_of_X, Maximum_length,dist_string, reject_fea_path = 'None',
    initializer = "he_normal", loss_function = "weighted_BCE", weight_p=1.0, weight_n=1.0,  list_sep_flag=False,  if_use_binsize = False): 

    print("\n######################################\n佛祖保佑，永不迨机，永无bug，精度九十九\n######################################\n")
    feature_2D_num=feature_num # the number of features for each residue
 
    print("Load feature number", feature_2D_num)
    ### Define the model 

    res_model_out= "%s/model-train-%s.json" % (CV_dir,model_prefix)
    res_model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir,model_prefix)
    res_model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir,model_prefix)

    if model_prefix == 'DNCON4_2dINCEP':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#0.00
        DNCON4_RES = DeepInception_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    elif model_prefix == 'DNCON4_2dDIARES':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#0.001  decay=0.0
        DNCON4_RES = DilatedRes_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    elif model_prefix == 'DNCON4_2dGRES':
        opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)
        DNCON4 = GoogleRes_with_paras_2D(win_array, feature_2D_num, nb_filters, nb_layers, opt, initializer, loss_function, weight_p, weight_n)
    elif model_prefix == 'DNCON4_2dRES':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)#0.0
        DNCON4_RES = DeepResnet_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
        # DNCON4_RES_test = DeepResnet_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p, weight_n, model_save = True)


    model_json = DNCON4_RES.to_json()
    print("Saved model to disk")
    with open(res_model_out, "w") as json_file:
        json_file.write(model_json)

        
    rerun_flag=0
    train_acc_history_out = "%s/training_%s.acc_history" % (CV_dir, model_prefix)
    res_val_acc_history_out = "%s/validation_%s.acc_history" % (CV_dir, model_prefix)
    res_best_val_acc_out = "%s/best_validation_%s.acc_history" % (CV_dir, model_prefix)
    if os.path.exists(res_model_weight_out_best):
        print("######## Loading existing weights ",res_model_weight_out_best)
        DNCON4_RES.load_weights(res_model_weight_out_best)
        rerun_flag = 1
    else:
        print("######## Setting initial weights")   
    
        chkdirs(train_acc_history_out)     
        with open(train_acc_history_out, "a") as myfile:
          myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\n")
          
        chkdirs(res_val_acc_history_out)     
        with open(res_val_acc_history_out, "a") as myfile:
          myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\n")

        chkdirs(res_best_val_acc_out)     
        with open(res_best_val_acc_out, "a") as myfile:
          myfile.write("Seq_Name\tSeq_Length\tFeature\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\n")

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

    DNCON4_RES.compile(loss=loss_function, metrics=['acc'], optimizer=opt)

    res_model_weight_epochs = "%s/model_weights/"%(CV_dir)
    res_model_predict= "%s/predict_map/"%(CV_dir)
    res_model_val_acc= "%s/val_acc_inepoch/"%(CV_dir)
    chkdirs(res_model_weight_epochs)
    chkdirs(res_model_predict)
    # chkdirs(model_predict_casp13)
    chkdirs(res_model_val_acc)

    tr_l = build_dataset_dictionaries_train(path_of_lists)
    tr_l_dict = subset_pdb_dict(tr_l, 0, Maximum_length, 7000, 'ordered')
    te_l = build_dataset_dictionaries_test(path_of_lists)
    all_l = te_l.copy()
    train_data_num = len(tr_l_dict)
    child_list_num = int(train_data_num/15)# 15 is the inter
    print('Total Number of Training dataset = ',str(len(tr_l_dict)))

    # callbacks=[reduce_lr]
    train_avg_acc_l5_best = 0 
    val_avg_acc_l5_best = 0
    min_seq_sep = 5
    lr_decay = False
    train_loss_last = 1e32
    train_loss_list = []
    evalu_loss_list = []
    COV = reject_fea_path + 'feature_to_use_cov_other2d_com.txt'
    PLM = reject_fea_path + 'feature_to_use_plm_other2d_com.txt'
    PRE = reject_fea_path + 'feature_to_use_pre_other2d_com.txt'
    OTHER = reject_fea_path + 'feature_to_use_other.txt'
    # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
    for epoch in range(epoch_rerun,epoch_outside):
        if (epoch >=30 and lr_decay == False):
            print("Setting lr_decay as true")
            lr_decay = True
            # opt = SGD(lr=0.01, momentum=0.9, decay=0.00, nesterov=False)
            # DNCON4_RES.load_weights(res_model_weight_out_best)
            # DNCON4_RES.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)

        print("\n############ Running epoch ", epoch)
        if epoch == 0 and rerun_flag == 0:
            first_inepoch = 1
            # train cov
            history = DNCON4_RES.fit_generator(generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, PLM, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length), 
                steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = first_inepoch, max_queue_size=200, workers=2, use_multiprocessing=False)           
            train_loss_list.append(history.history['loss'][first_inepoch-1])
            # train pre
            # history = DNCON4_RES.fit_generator(generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, PRE, if_use_binsize=if_use_binsize, predict_method=predict_method), steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = first_inepoch, 
            #     validation_data = generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, PLM, dataset_select='vali', if_use_binsize=if_use_binsize, predict_method=predict_method), validation_steps = len(te_l))           
            # train_loss_list.append(history.history['loss'][first_inepoch-1])
            # evalu_loss_list.append(history.history['val_loss'][first_inepoch-1])
            # train plm
            history = DNCON4_RES.fit_generator(generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, COV, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length), 
                steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = first_inepoch, max_queue_size=200, workers=2, use_multiprocessing=False)           
            train_loss_list.append(history.history['loss'][first_inepoch-1])
        else:
            # train cov
            history = DNCON4_RES.fit_generator(generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, PLM, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length), 
                steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = 1, max_queue_size=200, workers=2, use_multiprocessing=False)  
            train_loss_list.append(history.history['loss'][0])
            # train pre
            # history = DNCON4_RES.fit_generator(generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, PRE, if_use_binsize=if_use_binsize, predict_method=predict_method), steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = 1,
            #     validation_data = generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, PLM, dataset_select='vali', if_use_binsize=if_use_binsize, predict_method=predict_method), validation_steps = len(te_l))  
            # train_loss_list.append(history.history['loss'][0])
            # evalu_loss_list.append(history.history['val_loss'][0])
            # train plm
            history = DNCON4_RES.fit_generator(generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, COV, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length), 
                steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = 1, max_queue_size=200, workers=2, use_multiprocessing=False)  
            train_loss_list.append(history.history['loss'][0])
        DNCON4_RES.save_weights(res_model_weight_out)

        # DNCON4_RES.save(model_and_weights)
        
        ### save models
        res_model_weight_out_inepoch = "%s/model-train-weight-%s-epoch%i.h5" % (res_model_weight_epochs,model_prefix,epoch)
        DNCON4_RES.save_weights(res_model_weight_out_inepoch)
        ##### running validation

        print("Now evaluate for epoch ",epoch)
        val_acc_out_inepoch = "%s/validation_epoch%i.acc_history" % (res_model_val_acc, epoch) 
        sys.stdout.flush()
        selected_list = subset_pdb_dict(te_l,   0, Maximum_length, 7000, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
        print('Loading data sets ..',end='')

        step_num = 0
        #The init of acc parameters
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
        out_avg_pc_l5_pre = 0.0
        out_avg_pc_l2_pre = 0.0
        out_avg_pc_1l_pre = 0.0
        out_avg_acc_l5_pre = 0.0
        out_avg_acc_l2_pre = 0.0
        out_avg_acc_1l_pre = 0.0
        out_avg_pc_l5_sum = 0.0
        out_avg_pc_l2_sum = 0.0
        out_avg_pc_1l_sum = 0.0
        out_avg_acc_l5_sum = 0.0
        out_avg_acc_l2_sum = 0.0
        out_avg_acc_1l_sum = 0.0
        out_gloable_mse = 0.0
        out_weighted_mse = 0.0

        for key in selected_list:
            value = selected_list[key]
            p1 = {key: value}
            if if_use_binsize:
                length = 320
            else:
                length = value
            print(len(p1))
            if len(p1) < 1:
                continue
            print("start predict")
            selected_list_2D_cov = get_x_2D_from_this_list(p1, path_of_X, length,dist_string, COV, value)
            selected_list_2D_plm = get_x_2D_from_this_list(p1, path_of_X, length,dist_string, PLM, value)
            # selected_list_2D_pre = get_x_2D_from_this_list(p1, path_of_X, length,dist_string, PRE, value)
            if (type(selected_list_2D_cov) == bool) or (type(selected_list_2D_plm) == bool) :
                continue

            print("selected_list_2D.shape: ",selected_list_2D_cov.shape)
            print('Loading label sets..')
            selected_list_label = get_y_from_this_list(p1, path_of_Y_evalu, 0, length, dist_string)# dist_string 80

            DNCON4_RES.load_weights(res_model_weight_out)

            DNCON4_RES_prediction_cov = DNCON4_RES.predict([selected_list_2D_cov], batch_size= 1)
            DNCON4_RES_prediction_plm = DNCON4_RES.predict([selected_list_2D_plm], batch_size= 1)
            # DNCON4_RES_prediction_pre = DNCON4_RES.predict([selected_list_2D_pre], batch_size= 1)

            CMAP = DNCON4_RES_prediction_cov.reshape(length, length)
            Map_UpTrans = np.triu(CMAP, 1).T
            Map_UandL = np.triu(CMAP)
            real_cmap_cov = Map_UandL + Map_UpTrans

            CMAP = DNCON4_RES_prediction_plm.reshape(length, length)
            Map_UpTrans = np.triu(CMAP, 1).T
            Map_UandL = np.triu(CMAP)
            real_cmap_plm = Map_UandL + Map_UpTrans

            # CMAP = DNCON4_RES_prediction_pre.reshape(length, length)
            # Map_UpTrans = np.triu(CMAP, 1).T
            # Map_UandL = np.triu(CMAP)
            # real_cmap_pre = Map_UandL + Map_UpTrans

            # real_cmap_sum = real_cmap_cov * 0.25 + real_cmap_plm * 0.6 + real_cmap_pre * 0.20
            real_cmap_sum = real_cmap_cov * 0.35 + real_cmap_plm * 0.65

            DNCON4_RES_prediction_cov = real_cmap_cov.reshape(len(p1), length*length)
            DNCON4_RES_prediction_plm = real_cmap_plm.reshape(len(p1), length*length)
            # DNCON4_RES_prediction_pre = real_cmap_pre.reshape(len(p1), length*length)
            DNCON4_RES_prediction_sum = real_cmap_sum.reshape(len(p1), length*length)

            global_mse = 0.0
            weighted_mse = 0.0
            if predict_method == 'real_dist':
                selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, length, dist_string, lable_type = 'real')# dist_string 80
                global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_RES_prediction, selected_list_label_dist)
                # to binary
                DNCON4_RES_prediction = DNCON4_RES_prediction * (DNCON4_RES_prediction <= 8) 

            (a, b, c,avg_pc_l5_cov,avg_pc_l2_cov,avg_pc_1l_cov,avg_acc_l5_cov,avg_acc_l2_cov,avg_acc_1l_cov) = evaluate_prediction_4(p1, DNCON4_RES_prediction_cov, selected_list_label, 24)
            val_acc_history_content_cov = "%s\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,'COV',avg_pc_l5_cov,avg_pc_l2_cov,avg_pc_1l_cov,avg_acc_l5_cov,avg_acc_l2_cov,avg_acc_1l_cov)
            
            (a, b, c,avg_pc_l5_plm,avg_pc_l2_plm,avg_pc_1l_plm,avg_acc_l5_plm,avg_acc_l2_plm,avg_acc_1l_plm) = evaluate_prediction_4(p1, DNCON4_RES_prediction_plm, selected_list_label, 24)
            val_acc_history_content_plm = "%s\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,'PLM',avg_pc_l5_plm,avg_pc_l2_plm,avg_pc_1l_plm,avg_acc_l5_plm,avg_acc_l2_plm,avg_acc_1l_plm)
            
            # (a, b, c,avg_pc_l5_pre,avg_pc_l2_pre,avg_pc_1l_pre,avg_acc_l5_pre,avg_acc_l2_pre,avg_acc_1l_pre) = evaluate_prediction_4(p1, DNCON4_RES_prediction_pre, selected_list_label, 24)
            # val_acc_history_content_pre = "%s\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,'PRE',avg_pc_l5_pre,avg_pc_l2_pre,avg_pc_1l_pre,avg_acc_l5_pre,avg_acc_l2_pre,avg_acc_1l_pre)
            
            (a, b, c,avg_pc_l5_sum,avg_pc_l2_sum,avg_pc_1l_sum,avg_acc_l5_sum,avg_acc_l2_sum,avg_acc_1l_sum) = evaluate_prediction_4(p1, DNCON4_RES_prediction_sum, selected_list_label, 24)
            val_acc_history_content_sum = "%s\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,'SUM',avg_pc_l5_sum,avg_pc_l2_sum,avg_pc_1l_sum,avg_acc_l5_sum,avg_acc_l2_sum,avg_acc_1l_sum)
           
            print('The best validation accuracy is ',val_acc_history_content_cov)
            print('The best validation accuracy is ',val_acc_history_content_plm)
            # print('The best validation accuracy is ',val_acc_history_content_pre)
            print('The best validation accuracy is ',val_acc_history_content_sum)
            with open(val_acc_out_inepoch, "a") as myfile:
                myfile.write(val_acc_history_content_cov)
                myfile.write(val_acc_history_content_plm)
                # myfile.write(val_acc_history_content_pre)
                myfile.write(val_acc_history_content_sum)
            # The add of acc parameters
            out_gloable_mse += global_mse
            out_weighted_mse += weighted_mse 
            out_avg_pc_l5_cov += avg_pc_l5_cov 
            out_avg_pc_l2_cov += avg_pc_l2_cov 
            out_avg_pc_1l_cov += avg_pc_1l_cov 
            out_avg_acc_l5_cov += avg_acc_l5_cov 
            out_avg_acc_l2_cov += avg_acc_l2_cov 
            out_avg_acc_1l_cov += avg_acc_1l_cov 
            out_avg_pc_l5_plm += avg_pc_l5_plm 
            out_avg_pc_l2_plm += avg_pc_l2_plm 
            out_avg_pc_1l_plm += avg_pc_1l_plm 
            out_avg_acc_l5_plm += avg_acc_l5_plm 
            out_avg_acc_l2_plm += avg_acc_l2_plm 
            out_avg_acc_1l_plm += avg_acc_1l_plm 
            # out_avg_pc_l5_pre += avg_pc_l5_pre 
            # out_avg_pc_l2_pre += avg_pc_l2_pre 
            # out_avg_pc_1l_pre += avg_pc_1l_pre 
            # out_avg_acc_l5_pre += avg_acc_l5_pre 
            # out_avg_acc_l2_pre += avg_acc_l2_pre 
            # out_avg_acc_1l_pre += avg_acc_1l_pre 
            out_avg_pc_l5_sum += avg_pc_l5_sum 
            out_avg_pc_l2_sum += avg_pc_l2_sum 
            out_avg_pc_1l_sum += avg_pc_1l_sum 
            out_avg_acc_l5_sum += avg_acc_l5_sum 
            out_avg_acc_l2_sum += avg_acc_l2_sum 
            out_avg_acc_1l_sum += avg_acc_1l_sum 
            
            step_num += 1
        print ('step_num=', step_num)
        # The out avg acc parameters
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
        out_avg_pc_l5_pre /= all_num
        out_avg_pc_l2_pre /= all_num
        out_avg_pc_1l_pre /= all_num
        out_avg_acc_l5_pre /= all_num
        out_avg_acc_l2_pre /= all_num
        out_avg_acc_1l_pre /= all_num
        out_avg_pc_l5_sum /= all_num
        out_avg_pc_l2_sum /= all_num
        out_avg_pc_1l_sum /= all_num
        out_avg_acc_l5_sum /= all_num
        out_avg_acc_l2_sum /= all_num
        out_avg_acc_1l_sum /= all_num
        

        val_acc_history_content_cov = "%i\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (interval_len,epoch,'COV',
            out_avg_pc_l5_cov,out_avg_pc_l2_cov,out_avg_pc_1l_cov,out_avg_acc_l5_cov,out_avg_acc_l2_cov,out_avg_acc_1l_cov, out_gloable_mse, out_weighted_mse)
        val_acc_history_content_plm = "%i\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (interval_len,epoch,'PLM',
            out_avg_pc_l5_plm,out_avg_pc_l2_plm,out_avg_pc_1l_plm,out_avg_acc_l5_plm,out_avg_acc_l2_plm,out_avg_acc_1l_plm, out_gloable_mse, out_weighted_mse)
        # val_acc_history_content_pre = "%i\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (interval_len,epoch,'PRE',
        #     out_avg_pc_l5_pre,out_avg_pc_l2_pre,out_avg_pc_1l_pre,out_avg_acc_l5_pre,out_avg_acc_l2_pre,out_avg_acc_1l_pre, out_gloable_mse, out_weighted_mse)
        val_acc_history_content_sum = "%i\t%i\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (interval_len,epoch,'SUM',
            out_avg_pc_l5_sum,out_avg_pc_l2_sum,out_avg_pc_1l_sum,out_avg_acc_l5_sum,out_avg_acc_l2_sum,out_avg_acc_1l_sum, out_gloable_mse, out_weighted_mse)
        
        print('The validation accuracy is ',val_acc_history_content_cov)
        print('The validation accuracy is ',val_acc_history_content_plm)
        # print('The validation accuracy is ',val_acc_history_content_pre)
        print('The validation accuracy is ',val_acc_history_content_sum)

        with open(res_val_acc_history_out, "a") as myfile:
                    myfile.write(val_acc_history_content_cov)  
                    myfile.write(val_acc_history_content_plm) 
                    # myfile.write(val_acc_history_content_pre)   
                    myfile.write(val_acc_history_content_sum)  

        train_loss = history.history['loss'][0]

        if (lr_decay and train_loss_last != 1e32 and epoch > 25):
            current_lr = K.get_value(DNCON4_RES.optimizer.lr)
            print("Current learning rate is {} ...".format(current_lr))
            if (epoch % 20 == 0):
                K.set_value(DNCON4_RES.optimizer.lr, current_lr * 0.1)
                print("Decreasing learning rate to {} ...".format(current_lr * 0.1))
            # if (train_loss < train_loss_last and current_lr < 0.01):
            #     K.set_value(DNCON4_RES.optimizer.lr, current_lr * 1.1)
            #     print("Increasing learning rate to {} ...".format(current_lr * 1.1))
            # else:
            #     K.set_value(DNCON4_RES.optimizer.lr, current_lr * 0.6)
            #     print("Decreasing learning rate to {} ...".format(current_lr * 0.6))
        train_loss_last = train_loss

        if out_avg_acc_l5_sum >= val_avg_acc_l5_best:
            val_avg_acc_l5_best = out_avg_acc_l5_sum 
            score_imed = "Accuracy L5 of Val: %.4f\t\n" % (val_avg_acc_l5_best)
            print("Saved best weight to disk, ", score_imed)
            DNCON4_RES.save_weights(res_model_weight_out_best)

        print("Train loss history:", train_loss_list)
        print("Validation loss history:", evalu_loss_list)
        #clear memory
        # K.clear_session()
        # tf.reset_default_graph()

    print("Training finished, best validation acc = ",val_avg_acc_l5_best)
    return val_avg_acc_l5_best
