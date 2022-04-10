import time
from datetime import datetime
import torch
import torch.nn as nn
import os
import statistics

from parsers import parsersers_,constellation,snrdb_list_test
from Data_loader import Data_loader_test
from Train_Eval_funcs import evaluate

from GNN import GNN
from GEPNet import GEPNet

dtype = torch.float64
torch.set_default_dtype(dtype)

device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    

# Initialization
args=vars(parsersers_())
compare         = args['compare']
Nr              = args['Nr']
Nt_list_train   = args['Nt_list']
Nt_list         = args['Nt_list_test']
num_classes     = args['num_classes']
total_samples   = args['samples']
bs_test         = args['bs_test']
iter_data       = round(total_samples/bs_test)
num_classes     = args['num_classes']
beta            = args['beta']
num_neuron      = args['num_neuron']
num_su          = args['su']
dropout         = args['Dropout']
iter_GEPNet     = args['iter_GEPNet']
iter_gnn        = args['iter_GNN']
iter_EP_gD      = args['iter_EP_genData']
QAM_cardinality = len(constellation)**2


dt_string = datetime.now().strftime("%d_%H:%M:%S")


# Initialize models, optimizer, and learning scheduler
GEPNet = GEPNet(iter_GEPNet, beta, Nr,num_su, num_neuron,constellation, device, dtype)
model = GNN(iter_gnn, num_neuron, num_su, num_classes, dropout).to(device)
criterion = nn.CrossEntropyLoss().to(device)
name=GEPNet.__class__.__name__ 

# Load model
model.load_state_dict(torch.load(f'models/{name}_{Nr}X{Nt_list_train}_{QAM_cardinality}QAM/model.pkl'))
model=model.to(device)
model.eval()


for Nt in (Nt_list):
    test_SER_MMSE_list=[]
    test_SER_EP_list=[]
    test_SER_list=[]
    idxs=[]
    snr_list = snrdb_list_test[Nt]
    for snr in snr_list:
        
        t = time.time()
        
        S_MMSE_list = []
        S_EP_list = []
        S_ML_list = []
        S_GEP_list = []
        for iter_data_index in range(iter_data):
        
            dataLoader= Data_loader_test (Nt,Nr,bs_test,snr,constellation,iter_EP_gD,compare)
            test_dataloader,SER_mmse,SER_EP = dataLoader.getTestData()  
            loss_val,val_acc,val_SER=evaluate(model,GEPNet,device,test_dataloader, criterion, Nt*2,dtype,constellation)
            
            S_MMSE_list.append(SER_mmse)
            S_EP_list.append(SER_EP)
            S_GEP_list.append(val_SER)
                    
        if compare: 
            p_mmse = statistics.mean(S_MMSE_list)   
            p_ep = statistics.mean(S_EP_list) 
            test_SER_MMSE_list.append(p_mmse)                                                                                                                                 
            test_SER_EP_list.append(p_ep)                                                                                                                                                                    
        p_gep = statistics.mean(S_GEP_list)
        test_SER_list.append(p_gep)
        
        idxs.append(snr)
        elapsed = time.time() - t
            
        if compare == True:
            print(f'SNR={snr}: SER_MMSE {p_mmse:.8f}; SER_EP {p_ep:.8f}; SER_GEPNet {p_gep:.8f} \n')
        else:
            print(f'SNR={snr}: SER_GEPNet {p_gep:.8f} \n')
        print("Time elapsed" ,elapsed,"\n")
        
        
    if not os.path.exists(f'Test_reports/{name}_{Nr}X{Nt}_{QAM_cardinality}QAM'):
        os.makedirs(f'Test_reports/{name}_{Nr}X{Nt}_{QAM_cardinality}QAM')
    
    if compare: 
        report={'MMSE':test_SER_MMSE_list,'EP':test_SER_EP_list,'GEPNet':test_SER_list}
        f = open(f'Test_reports/{name}_{Nr}X{Nt}_{QAM_cardinality}QAM/TestReport.txt',"w")
        f.write("SNR=" + str(idxs) + "\n" + "MMSE=" + str(test_SER_MMSE_list) + "\n" + "EP=" + str(test_SER_EP_list) + "\n" + "GEPNet=" + str(test_SER_list))
        f.close()
    else:    
        report={'GEPNet':test_SER_list}
        f = open(f'Test_reports/{name}_{Nr}X{Nt}_{QAM_cardinality}QAM/TestReport.txt',"w")
        f.write("SNR=" + str(idxs) + "\n"+ "GEPNet=" + str(test_SER_list))
        f.close()
    
    
    f = open(f'Test_reports/{name}_{Nr}X{Nt}_{QAM_cardinality}QAM/NumberofTestingData.txt', "w")
    t_s = iter_data*bs_test
    f.write(str(t_s))
    f.close()
        
    from matplotlib import pyplot as plt
    if compare: 
        plt.semilogy(idxs,test_SER_EP_list,'bs-.', label='EP')
        plt.semilogy(idxs,test_SER_MMSE_list,'r--', label='MMSE')
        plt.semilogy(idxs,test_SER_list,'go-.', label='GEPNet')
    else:
        plt.semilogy(idxs,test_SER_list,'go-.', label='GEPNet')
    plt.xlabel('SNR (dB)')
    plt.ylabel('SER')
    plt.legend(loc='best')  
    plt.show()