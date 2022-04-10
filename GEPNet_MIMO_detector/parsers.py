import argparse
import numpy as np

num_constell = 16
symblist = [x for x in range(1,17,2)] #max 26244-QAM
symPerQadrant = num_constell/4
list_of_symbols = [symblist[x] for x in range (int(np.sqrt(symPerQadrant)))]
energy_symbols = sum([number ** 2 for number in list_of_symbols])
norm_cons = np.sqrt(2*np.sqrt(symPerQadrant)  * ( energy_symbols) / symPerQadrant)
norm_symbs = list_of_symbols/norm_cons
constellation= np.array(np.concatenate((norm_symbs, -1*norm_symbs)), dtype=np.float32)

# Training configuration
NT1_tr = 4
NT2_tr = 8
snrdb_list_tr = {NT1_tr:np.array([8.0, 23.5]),NT2_tr:np.array([10.0, 30.5])}    # Only need to specify SNRdB min and max for training
# Testing configuration
NT1 = 4
NT2 = 8
snrdb_list_test = {NT1:np.arange(8.0, 23.5,3.0),NT2:np.arange(10.0, 40.5,5.0)}


def parsersers_():
    parser = argparse.ArgumentParser(description='GNN for detection')
    parser.add_argument('--Nr', type=int, default=8, help='Nr')
    parser.add_argument('--Nt_list', type=int, default=np.array([NT1_tr,NT2_tr]), help='Nt_list')
    parser.add_argument('--Nt_list_test', type=int, default=np.array([NT1,NT2]), help='Nt_list_test')
    
    parser.add_argument('--saved_all_models',default=False, help='saved_all_models')
    parser.add_argument('--compare', default=True, help='compare with other detectors')
    
    parser.add_argument('--samples', type=int, default=500000, help='train sample data')
    parser.add_argument('--batch_size', type=int, default=128, help='num_samplesPer_batch')
    parser.add_argument('--validation_size','-vs', type=int, default=5000, help='validation_size')
    parser.add_argument('--bs_test', type=int, default=256, help='TestingBatchSize')
    parser.add_argument('--n_epochs','-ne', type=int, default=500, help='n_epochs')
    
    parser.add_argument('--num_neuron', type=int, default=64, help='num_neuron')
    parser.add_argument('--su','-su', type=int, default=8, help='num_feature_su')
    parser.add_argument('--beta', type=float, default=0.7, help='beta_EP')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning_rate')
    parser.add_argument('--Dropout', type=float, default=0, help='dropout')
        
    parser.add_argument('--num_classes', type=int, default=constellation.shape[0], help='num_classes')
    parser.add_argument('--iter_EP_genData', type=float, default=10, help='number of iterations for conventional detector e.g., EP')
    parser.add_argument('--iter_GEPNet', type=int, default=10, help='number of GEPNet iterations ')
    parser.add_argument('--iter_GNN', type=int, default=2, help='number of GNN iterations inside a GEPNet iteration')
    return parser.parse_args()
