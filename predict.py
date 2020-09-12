import argparse
import numpy as np

from ICM_utils import load_plm, load_features, predict_cmap

def make_argument_parser():

    parser = argparse.ArgumentParser(description="Predict a model.",formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--model_type', '-m', type=str, required=True,help='Type of model, can be one of sequence_attention, regional_attention or combined')
    parser.add_argument('--plm_data', '-l', type=str, required=True,help='Path to PLM data. Should be a numpy array flatten from (441,L,L), and saved as .npy format(https://numpy.org/doc/stable/reference/generated/numpy.save.html)')
    parser.add_argument('--pssm_data', '-s', type=str, required=True,help='Path to PSSM data. Should be a text file start with \" # PSSM\" as the first line, and the following contents should be 20 lines each contains L values, where L is the length of the input sequence.')
    parser.add_argument('--out_file', '-o', type=str, required=True,help='Path to output contact map. An L by L numeric matrix saved as TSV format.')

    return parser


def main():
    parser = make_argument_parser()
    args = parser.parse_args()
    
    model_type = args.model_type
    plm_data = args.plm_data
    pssm_data = args.pssm_data
    out_file = args.out_file

    # load plm
    inputs_plm = load_plm(plm_data)
    
    # load pssm
    inputs_pssm = load_features(pssm_data)
    
    input_data = [inputs_plm,inputs_pssm]
    
    # predict contact
    if model_type == 'sequence_attention' or model_type == 'regional_attention':
        output_map = predict_cmap(model_type,input_data)
    else:
        output_map = (predict_cmap('sequence_attention',input_data) + predict_cmap('regional_attention',input_data))/2
            
    np.savetxt(out_file, output_map, fmt='%.4f')


if __name__ == '__main__':
    main()
