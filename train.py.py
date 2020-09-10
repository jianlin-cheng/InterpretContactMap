import sys
import os
from shutil import copyfile
import platform
from glob import glob
from Model_training import *
from DNCON_lib import *
from training_strategy import *


if (len(sys.argv) != 15) and (len(sys.argv) != 19):
  print('please input the right parameters')
  sys.exit(1)
current_os_name = platform.platform()
print('%s' % current_os_name)

# 'Linux-4.15.0-47-generic-x86_64-with-Ubuntu-18.04-bionic'
# 'Linux-3.10.0-957.5.1.el7.x86_64-x86_64-with-centos-7.6.1810-Core'
if 'Ubuntu' in current_os_name.split('-'): #on local
  GLOBAL_PATH='/mnt/data/zhiye/Python/DNCON4/architecture_distance'
  sysflag='local'
elif 'centos' in current_os_name.split('-'): #on lewis or multicom
  GLOBAL_PATH=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
  sysflag='lewis'
else:
  print('Please check current operate system!')
  sys.exit(1)

sys.path.insert(0, GLOBAL_PATH+'/lib/')
print (GLOBAL_PATH+'/lib/')



inter= 150
nb_filters= 64
nb_layers= 43
opt='nadam'
filtsize=3
in_epoch=1


feature_dir = sys.argv[1] 
outputdir = sys.argv[2] 

out_epoch=int(sys.argv[3]) 
batchsize = int(sys.argv[4])
net_name = sys.argv[5] # baselineModel or regional_attention or sequence_attention


acclog_dir = sys.argv[3]+'/acc_log/'
weight_p = 1.0

att_config = 0
kmax = 7
att_outdim = 16
insert_pos = 'tail'


CV_dir=outputdir+'/filter'+str(nb_filters)+'_layers'+str(nb_layers)+'_inter'+str(inter)+'_opt'+str(opt)+'_ftsize'+str(filtsize)+'_batchsize'+str(batchsize)+'_'+str(weight_p)+'_'+net_name

lib_dir=GLOBAL_PATH+'/lib/'

gpu_schedul_strategy(sysflag, gpu_mem_rate = 0.5, allow_growth = True)

# filetsize_array = list(map(int,filtsize.split("_")))

rerun_epoch=0
if not os.path.exists(CV_dir):
  os.makedirs(CV_dir)
else:
  h5_num = len(glob(CV_dir + '/model_weights/*.h5'))
  rerun_epoch = h5_num
  if rerun_epoch <= 0:
    rerun_epoch = 0
    print("This parameters already exists, quit")
    # sys.exit(1)
  print("####### Restart at epoch ", rerun_epoch)

def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)

def chkfiles(fn):
  if os.path.exists(fn):
    return True 
  else:
    return False


dist_string = '80'

reject_fea_file = ['feature_to_use_2d.txt','feature_to_use_1d.txt']
path_of_lists = os.path.dirname(GLOBAL_PATH)+'/data/deepcov/lists-test-train/'

  
path_of_Y         =  feature_dir 
path_of_X         = feature_dir
Maximum_length=460 # 800 will get memory error

sample_datafile=path_of_lists + '/sample.lst'
train_datafile=path_of_lists + '/train.lst'
val_datafile=path_of_lists + '/test.lst'

import time
feature_num1 = load_sample_data_2D(path_of_lists, path_of_X, path_of_Y, inter,5000,0,dist_string, reject_fea_file[0])
feature_num2 = load_sample_data_2D(path_of_lists, path_of_X, path_of_Y, inter,5000,0,dist_string, reject_fea_file[1])
feature_num=[feature_num1, feature_num2]

# testdata_all_dict_padding = load_train_test_data_padding_with_interval_2D(val_datafile, feature_dir, inter,5000,0,dist_string, reject_fea_file, sample_flag=True)  

start_time = time.time()

initializer = 'he_normal'
loss_function = 'binary_crossentropy'

best_acc=DNCON4_1d2dconv_train_win_filter_layer_opt_fast_2D_generator(feature_num,CV_dir, feature_dir,net_name, out_epoch,in_epoch,rerun_epoch,inter,
  5000, filtsize, True,'sigmoid',nb_filters,nb_layers,opt,lib_dir, batchsize,path_of_lists,path_of_Y, path_of_X,Maximum_length,dist_string, reject_fea_file,
  initializer, loss_function, weight_p, att_config=att_config,kmax=kmax,att_outdim=att_outdim,insert_pos=insert_pos )

model_prefix = "DRES"
acc_history_out = "%s/%s.acc_history" % (acclog_dir, model_prefix)
chkdirs(acc_history_out)
if chkfiles(acc_history_out):
    print ('acc_file_exist,pass!')
    pass
else:
    print ('create_acc_file!')
    with open(acc_history_out, "w") as myfile:
        myfile.write("time\t netname\t initializer\t loss_function\t weight0\t weight1\t filternum\t layernum\t kernelsize\t batchsize\t accuracy\n")

time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
acc_history_content = "%s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %.4f\n" % (time_str, model_prefix, initializer, loss_function, str(weight_p),
 str(nb_filters),str(nb_layers),str(filtsize),str(batchsize),best_acc)
with open(acc_history_out, "a") as myfile: myfile.write(acc_history_content) 
print("--- %s seconds ---" % (time.time() - start_time))
print("outputdir:", CV_dir)
