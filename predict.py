import os
import argparse
import numpy as np

from ICM_utils import load_plm, load_features1D, init_model, predict_cmap, extract_attention_score

def make_argument_parser():

    parser = argparse.ArgumentParser(description="Predict a model.",formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--model_type', '-m', type=str, required=True,
                        help='Type of model, can be one of sequence, regional or combined')
    parser.add_argument('--plm_data', '-l', type=str, required=True,
                        help='Path to PLM data. Should be a numpy array flatten from (441,L,L), and saved as .npy format(https://numpy.org/doc/stable/reference/generated/numpy.save.html)')
    parser.add_argument('--pssm_data', '-s', type=str, required=True,
                        help='Path to PSSM data. Should be a text file start with \" # PSSM\" as the first line, and the following contents should be 20 lines each contains L values, where L is the length of the input sequence.')
    parser.add_argument('--out_file', '-o', type=str, required=True,
                        help='Path to output contact map. An L by L numeric matrix saved as TSV format.')
    parser.add_argument('--weights', '-w', action='store_true',
                        help='Should attention weights be extracted.')

    return parser


def main():
    parser = make_argument_parser()
    args = parser.parse_args()
    
    model_type = args.model_type
    plm_data = args.plm_data
    pssm_data = args.pssm_data
    out_file = args.out_file
    weights = args.weights
    
    inputs_plm = load_plm(plm_data) # load plm
    inputs_pssm = load_features1D(pssm_data,'PSSM') # load pssm
    input_data = [inputs_plm,inputs_pssm]
    
    save_path = os.path.dirname(out_file)

    if model_type == 'sequence' or model_type == 'regional':
        model,model_dir = init_model(model_type)
        model.load_weights(model_dir+'model_weights.h5')
        output_map = predict_cmap(model,input_data)
        if weights:
            score = extract_attention_score(model,input_data,model_type)
            np.save(save_path+'/'+model_type+'_weights.npy',score)
    else:
        smodel,_ = init_model('sequence')
        s_map = predict_cmap(smodel,input_data)
        if weights:
            score_s = extract_attention_score(smodel,input_data,'sequence')
            np.save(save_path+'/sequence_weights.npy',score_s)
            
        rmodel,_ = init_model('regional')
        r_map = predict_cmap(rmodel,input_data)
        if weights:
            score_r = extract_attention_score(rmodel,input_data,'regional')
            np.save(save_path+'/regional_weights.npy',score_r)
            
    output_map = (s_map + r_map)/2        
    np.savetxt(out_file, output_map, fmt='%.4f')


if __name__ == '__main__':
    main()
