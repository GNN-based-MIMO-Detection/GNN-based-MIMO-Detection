import torch
import time
from datetime import datetime

from parsers import parsersers_,constellation
from Data_loader import Data_loader_test
from Train_Eval_funcs import evaluate
from GNN_model import GNN

# initialization
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)
args=vars(parsersers_())

Nr              = args['Nr']
Nt              = args['Nt']
with_mmse       = args['MMSE']
snr_db_min_tr   = args['SNR_dB_min_train']
snr_db_max_tr   = args['SNR_dB_max_train']
SNR_dB_min_test = args['SNR_dB_min_test']
SNR_dB_max_test = args['SNR_dB_max_test']
SNR_step_test   = args['SNR_step_test']

num_classes     = args['num_classes']
iter_gnn        = args['iter_GNN']
num_neuron      = args['num_neuron']
num_su          = args['num_feature_su']
dropout         = args['Dropout']
batch_size      = args['batch_size']
total_samples   = args['samples']



# Load trained model
model = GNN(iter_gnn,num_neuron,num_su,num_classes,Nt*2,with_mmse,dropout).to(device)

model_name=model.__class__.__name__ 
model.load_state_dict(torch.load(f'models/{model_name}_MMSE_{with_mmse}_{Nr*2}X{Nt*2}_SNR_{snr_db_min_tr}_{snr_db_max_tr}_dB/model.pkl'))
model=model.to(device)
model.eval()

dt_string = datetime.now().strftime("%d_%H:%M:%S")




# Create directories to save trained models and training reports
f = open(f'reports/{model_name}_MMSE_{with_mmse}_{Nr*2}X{Nt*2}_SNR_{snr_db_min_tr}_{snr_db_max_tr}_dB/logTest.txt',"w")
f.write(dt_string + "\n")
f.write(str(device) + "\n")
f.close()


test_SER_MMSE_list=[]
test_SER_list=[]
idxs=[]
for idx in range(SNR_dB_min_test,SNR_dB_max_test,SNR_step_test ):

    # Testing phase
    t = time.time()
    
    dataLoader = Data_loader_test(Nt,Nr,total_samples,batch_size, idx,constellation)
    test_dataloader = dataLoader.getTestData()  
    test_acc,test_SER=evaluate(model,device,test_dataloader,Nt*2,dtype,num_neuron,constellation)
    
    print(f'test_acc={test_acc:.8f}, test_SER={test_SER:.8f},SNR={idx:.8f} \n')
    test_SER_list.append(test_SER)
    idxs.append(idx)

    # Record elapsed time    
    elapsed = time.time() - t
    print("Time elapsed" ,elapsed,"\n")
    f = open(f'reports/{model_name}_MMSE_{with_mmse}_{Nr*2}X{Nt*2}_SNR_{snr_db_min_tr}_{snr_db_max_tr}_dB/logTest.txt',"a")
    f.write("time per SNR point: " + str(elapsed) + "\n")
    f.close()


# Write results on the test report
report={'GNN':test_SER_list}
f = open(f'reports/{model_name}_MMSE_{with_mmse}_{Nr*2}X{Nt*2}_SNR_{snr_db_min_tr}_{snr_db_max_tr}_dB/TestReport.txt',"w")
f.write("SNR=" + str(idxs)  + "\n" + "GNN=" + str(test_SER_list))
f.close()


# Plot SER vs SNR
from matplotlib import pyplot as plt
plt.semilogy(idxs,test_SER_list,'go-', label='GNN')
plt.xlabel('SNR (dB)')
plt.ylabel('SER')
plt.legend(loc='best')
plt.show()

    

