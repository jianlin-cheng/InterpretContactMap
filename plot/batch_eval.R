model_dir = '/storage/htc/bdm/ccm3x/DNCON4/architecture_distance/outputs/DilatedResNet_arch/new_sturct/'
setup_file = '~/exp_table_all_use.csv'

slurm_command = '#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH --partition Gpu
#SBATCH --cpus-per-task=1  # cores per task
#SBATCH --mem-per-cpu=24G  # memory per core (default is 1GB/core)
#SBATCH --time 0-2:00     # days-hours:minutes
#SBATCH --account=general  # investors will replace this (e.g. `rc-gpu`)

## labels and outputs
#SBATCH --job-name=gpu_test
#SBATCH --output=results-%j.out  # %j is the unique jobID
#SBATCH --gres gpu:1

module load cuda/cuda-9.0.176
module load cudnn/cudnn-7.1.4-cuda-9.0.176
export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=""
## Activate python virtual environment
source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
module load R/R-3.3.1
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}\'DNCON4\'
feature_dir=/storage/htc/bdm/DNCON4/feature/Deepcov/uniref042018_unclust302017_metaclust50_06_2018/aln_filt50/
output_dir=$gloable_dir/architecture_distance/outputs/DilatedResNet_arch/new_sturct
acclog_dir=$gloable_dir/architecture_distance/outputs/All_Validation_Acc
printf "$gloable_dir"

'

models <- paste0(grep('filter.+',list.dirs(model_dir,full.names = T,recursive = F),value = T),'/')
setup_table <- read.csv(setup_file,as.is = T,row.names = 1)
for(i in models){
    exp_idx <- as.integer(unlist(strsplit(gsub('.+/(.+)/','\\1',i),'_'))[7])
    
    if(length(list.files(i,pattern = '.+.h5'))>0){
        model_name <- setup_table$model[exp_idx]
        feature_file <- setup_table$feature[exp_idx]
        # print(c(i,model_name,feature_file))
        eval_command <- paste0('python /storage/htc/bdm/ccm3x/DNCON4/architecture_distance/lib/Model_evaluate2.py ',
                               i,' ',model_name,' ',feature_file)
        print(eval_command)
        slurm_command <- paste0(slurm_command,'\n',eval_command)
        
    }
}

writeLines(slurm_command,'~/eval.sh')

#sbatch eval.sh
################################ when evaluation finished ##############################################
eval_result = list.files('~',pattern = '.+.out')

setup_file = '~/exp_table_all_use.csv'

eval_data <- readLines(eval_result)
setup_table <- read.csv(setup_file,as.is = T,row.names = 1)
setup_table1 <- cbind.data.frame(setup_table,'epoch_finished'=0,'epoch_used'=0,
                      'Precision_L5'=0,'Precision_L2'=0,'Precision_L1'=0,stringsAsFactors =F)

for(i in 1:length(eval_data)){
    if (length(grep('The validation accuracy is.+',eval_data[i]))>0){
        print(eval_data[i])
        score_i <- as.numeric(unlist(strsplit(gsub('The validation accuracy is  ','',eval_data[i]),'\t')))
        exp_name <- gsub('.+/(.+)//pred_map/','\\1',eval_data[i+2])
        exp_historyfile <- paste0(gsub('Predict map filepath: (.+)/pred_map/','\\1',eval_data[i+2]),'validation.acc_history')
        exp_history <- read.delim(exp_historyfile,header = F,as.is = T,sep = '\t',comment.char='I')
        exp_idx <- as.integer(unlist(strsplit(exp_name,'_'))[7])
        epoch_finished <- nrow(exp_history)
        epoch_used <- which.max(exp_history$V7)
        setup_table1[exp_idx,7:11] <- c(epoch_finished,epoch_used,score_i[4:6])
    }
}

write.csv(setup_table1,'~/eval_result_exp2.csv')






