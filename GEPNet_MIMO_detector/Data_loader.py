import torch
from Gen_dat import genData
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CF_Dataset(Dataset):
    def __init__(self, x, edge, Tx, noise_info, H, Rx):
        self.x = x
        self.edge = edge
        self.Tx = Tx
        self.noise_info =noise_info
        self.H =H
        self.Rx =Rx
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx,:], self.edge[idx,:] , self.Tx[idx,:], self.noise_info[idx,:],  self.H[idx,:],  self.Rx[idx,:]
    

class Data_loader (object) :
    
    def __init__(self, Nt, Nr, batch_size, SNR_dB_min, SNR_dB_max, constellation):
    
        dtype = torch.float64

        params = {
            'constellation': constellation,
            'Nt': Nt,  
            'Nr': Nr,  
            'batch_size': batch_size,
            'SNR_dB_min': SNR_dB_min,
            'SNR_dB_max': SNR_dB_max,
            'iter_EP_gD': 0,
            'compare': False
            }
        
        GD=genData(params)
        H_real, tX, rX, init_feats, edge_weight, noise_info, SER_mmse,SER_EP = GD.dataTrain()

        H_train=torch.from_numpy(H_real).to(dtype)
        Rx_train=torch.from_numpy(rX).to(dtype)
        x_train=torch.from_numpy(init_feats).to(dtype)
        edge_train = torch.from_numpy(edge_weight).to(dtype)
        Tx_train = torch.from_numpy(tX).to(torch.int64) 
        
        train_dataset = CF_Dataset(x_train, edge_train, Tx_train, noise_info, H_train, Rx_train)
        self.train_data = DataLoader(train_dataset, batch_size=batch_size)
        
    def getTrainData(self):
        return self.train_data
    
    


class Data_loader_test (object) :
    
    def __init__(self, Nt, Nr, bs_test, SNR_dB_test, constellation,iter_EP_gD,compare):
    
        dtype = torch.float64
        
        params = {
            'constellation': constellation,
            'batch_size':bs_test,
            'Nt': Nt,
            'Nr': Nr,
            'SNR_dB_min': SNR_dB_test,
            'SNR_dB_max': SNR_dB_test,
            'iter_EP_gD': iter_EP_gD,
            'compare': compare
            }
        GD=genData(params)
        H_real, tX, rX, init_feats, edge_weight, noise_info, SER_mmse,SER_EP = GD.dataTrain()
        self.SER_mmse = SER_mmse
        self.SER_EP = SER_EP
        
        x_test=torch.from_numpy(init_feats).to(dtype)
        Tx_test = torch.Tensor(tX).to(torch.int64)
        edge_weight_test=torch.from_numpy(edge_weight).to(dtype)
        H_test=torch.from_numpy(H_real).to(dtype)
        Rx_test=torch.from_numpy(rX).to(dtype)
        
        test_dataset =CF_Dataset(x_test,edge_weight_test,Tx_test,noise_info,H_test,Rx_test)

        self.test_data =DataLoader(test_dataset, batch_size=bs_test)

    
    def getTestData(self):
        return self.test_data,self.SER_mmse,self.SER_EP