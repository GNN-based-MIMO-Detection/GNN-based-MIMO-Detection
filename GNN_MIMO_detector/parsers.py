import argparse
import numpy as np

num_constell = 16   #Please set the number of QAM constellations


symblist = [x for x in range(1,17,2)] #max 26244-QAM
symPerQadrant = num_constell/4
list_of_symbols = [symblist[x] for x in range (int(np.sqrt(symPerQadrant)))]
energy_symbols = sum([number ** 2 for number in list_of_symbols])
norm_cons = np.sqrt(2*np.sqrt(symPerQadrant)  * ( energy_symbols) / symPerQadrant)
norm_symbs = list_of_symbols/norm_cons
constellation= np.array(np.concatenate((norm_symbs, -1*norm_symbs)), dtype=np.float32)

def parsersers_():
    parser = argparse.ArgumentParser(description='GNN for detection')
    
    parser.add_argument('--MMSE', default=False, action='store_false') #Set true if you want to use MMSE prior, otherwise set false
    parser.add_argument('--iter_GNN', type=int, default=10, help='iter_GNN') #Number of layers in the GNN
    parser.add_argument('--num_neuron', type=int, default=128, help='num_neuron') #Number of neurons in first layer of MLPs inside of the GNN
    parser.add_argument('--num_feature_su','-su', type=int, default=8, help='num_feature_su') #Number of embedded features     
    parser.add_argument('--samples', type=int, default=300000, help='train sample data') #Total number of training samples
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size') #Batch size
    parser.add_argument('--n_epochs','-ne', type=int, default=500, help='n_epochs') #Total number of training epochs
    
    parser.add_argument('--Nt', type=int, default=16, help='Nt') #Number of tranmitters
    parser.add_argument('--Nr', type=int, default=32, help='Nr') #Number of receivers
    
    parser.add_argument('--SNR_dB_min_train', type=int, default=12, help='SNR_dB_min') #Training SNR(dB)
    parser.add_argument('--SNR_dB_max_train', type=int, default=17.5, help='SNR_dB_max') #Training SNR(dB)
    parser.add_argument('--SNR_dB_min_test', type=int, default=12, help='SNR_dB_min_test') #Testing SNR(dB)
    parser.add_argument('--SNR_dB_max_test', type=int, default=18, help='SNR_dB_max_test') #Testing SNR(dB)
    parser.add_argument('--SNR_step_test', type=int, default=1, help='SNR_step_test') #Testing SNR(dB) step
    
    parser.add_argument('--num_classes', type=int, default=constellation.shape[0], help='num_classes')
    parser.add_argument('--Dropout', type=float, default=0, help='dropout') #Set dropout value


    return parser.parse_args()
