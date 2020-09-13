import argparse
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from ICM_utils import data_generator, load_sample_config, init_model
from os.path import normpath

def make_argument_parser():
    parser = argparse.ArgumentParser(description="Train a model.",formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--plm_data_path', '-plm', type=str, required=True,
                        help='Path to PLM data files')
    parser.add_argument('--pssm_data_path', '-pssm', type=str, required=True,
                        help='Path to PSSM data files')
    parser.add_argument('--label_path', '-l', type=str, required=True,
                        help='Path to label data files. Should be text files with 0/1 indicate the contacts.')
    parser.add_argument('--sample_list_file', '-s', type=str, 
                        required=True,help='Config file indicating the sample names for training and validation')
    parser.add_argument('--model_type', '-m', type=str, required=True,
                        help='Type of model, can be \"sequence\" or \"regional\"')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Path where the trained models and history are saved')
    parser.add_argument('--patience', '-pa', type=int, required=False,
                        help='Stop the training early for no improvements in validation after x epochs.',default=5)
    parser.add_argument('--epochs', '-e', type=int, required=False,
                        help='Number of epochs',default=60)

    return parser

def main():
    parser = make_argument_parser()
    args = parser.parse_args()
    
    plm_data_path = args.plm_data_path
    pssm_data_path = args.pssm_data_path
    label_path = args.label_path
    sample_list_file = args.sample_list_file
    output_dir = args.output_dir
    model_type = args.model_type
    patience = args.patience
    epochs = args.epochs

    output_model_best = normpath(output_dir + '/best_model.h5')
    output_model = normpath(output_dir + '/epoch{epoch:02d}.val_acc{val_acc:03f}.h5')
    output_history = normpath(output_dir + '/histroy.csv')

    sample_list_train, sample_list_val = load_sample_config(sample_list_file)
    datagen_train = data_generator(plm_data_path,pssm_data_path,label_path,sample_list_train)
    datagen_val = data_generator(plm_data_path,pssm_data_path,label_path,sample_list_val)
    
    checkpointer1 = ModelCheckpoint(filepath=output_model,verbose=1,
                                    save_best_only=False, monitor='val_acc',
                                    save_weights_only=True)
    checkpointer2 = ModelCheckpoint(filepath=output_model_best,verbose=1, 
                                    save_best_only=True, monitor='val_acc',
                                    save_weights_only=True)
    earlystopper = EarlyStopping(monitor='val_acc', patience=patience, verbose=1)
    csv_logger = CSVLogger(output_history,append=True)

    if model_type in ['sequence','regional']:
        model,_ = init_model(model_type)
    else:
        raise ValueError('Model type should be \"sequence\" or \"regional\"')
    model.fit_generator(datagen_train,epochs=epochs,validation_data=datagen_val,verbose=1,
                        callbacks=[checkpointer1,checkpointer2, earlystopper, csv_logger])
    
if __name__ == '__main__':
    main()
