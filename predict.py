import os
import argparse
import numpy as np

from ICM_utils import load_plm, tile_2d, predict_cmap

def make_argument_parser():

    parser = argparse.ArgumentParser(description="Predict a model.",formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--model_type', '-m', type=str, required=True,
                        help='Type of model, can be one of \"sequence\", \"regional\" or \"combine\"')
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
    inputs_pssm = tile_2d(np.load(pssm_data)) # load pssm
    input_data = [inputs_plm,inputs_pssm]
    save_path = os.path.dirname(out_file)

    if model_type == 'sequence' or model_type == 'regional':
        output_map, score = predict_cmap(input_data,model_type,weights)
        if weights:
            np.save(save_path+'/'+model_type+'_weights.npy',score)
    elif model_type == 'combine':
        s_map, score_s = predict_cmap(input_data,'sequence',weights)
        r_map, score_r = predict_cmap(input_data,'regional',weights)
        if weights:
            np.save(save_path+'/sequence_weights.npy',score_s)
            np.save(save_path+'/regional_weights.npy',score_r)
        output_map = (s_map + r_map)/2  
    else:
        raise ValueError('Model type should be one of \"sequence\", \"regional\" or \"combine\"')
    np.savetxt(out_file, output_map, fmt='%.4f')


if __name__ == '__main__':
    main()
