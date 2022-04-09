from Gen_dat import genData
import torch
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CF_Dataset(Dataset):
    def __init__(self, x_t, e_w, y_t, n_i, x_MMSE, var_MMSE):
        self.xt = x_t
        self.ew = e_w
        self.yt = y_t
        self.ni =n_i
        self.x_hat_MMSE = x_MMSE
        self.var_MMSE = var_MMSE
        

    def __len__(self):
        return len(self.xt)

    def __getitem__(self, idx):
        return self.xt[idx,:], self.ew[idx,:] , self.yt[idx,:], self.ni[idx,:], self.x_hat_MMSE[idx, :], self.var_MMSE[idx, :]
    

        

class Data_loader (object) :
    
    def __init__(self, Nt, Nr, samples, batch_size, SNR_dB_min, SNR_dB_max, constellation):
    
        dtype = torch.float64

        params = {
            'constellation': constellation,
            'Nt': Nt,  # Number of transmit antennas
            'Nr': Nr,  # Number of transmit antennas
            'samples': samples,
            'SNR_dB_min': SNR_dB_min,
            'SNR_dB_max': SNR_dB_max,
            }
        print(params)
        
        # Generate Data
        GD=genData(params)
        H_real, tX, rX, init_feats, edge_weight, noise_info, xhat_MMSE, var_MMSE = GD.dataTrain()
        
        # Portion of validation data
        validation_sizes = 0.2

        # Split training and validitaion data
        (init_feats_train, init_feats_val, 
         edge_weight_train, edge_weight_val,
         tX_train, tX_val, 
         xhat_MMSE_train, xhat_MMSE_val, 
         var_MMSE_train, var_MMSE_val,noise_info_train,noise_info_val) = train_test_split(init_feats,edge_weight,tX,xhat_MMSE,var_MMSE,noise_info,test_size=validation_sizes)
        
        # Convert numpy to torch
        x_train=torch.from_numpy(init_feats_train).to(dtype)
        x_val=torch.from_numpy(init_feats_val).to(dtype)
        Tx_train = torch.Tensor(tX_train).to(torch.int64) 
        Tx_val = torch.Tensor(tX_val).to(torch.int64)
        Xhat_MMSE_train = torch.from_numpy(xhat_MMSE_train).to(dtype)
        Xhat_MMSE_val = torch.from_numpy(xhat_MMSE_val).to(dtype)
        var_MMSE_train = torch.from_numpy(var_MMSE_train).to(dtype)
        var_MMSE_val = torch.from_numpy(var_MMSE_val).to(dtype)
        noise_info_train = torch.from_numpy(noise_info_train).to(dtype)
        noise_info_val = torch.from_numpy(noise_info_val).to(dtype)
    
        # Create training and validation dataset
        train_dataset = CF_Dataset(x_train,edge_weight_train,Tx_train,noise_info_train, Xhat_MMSE_train, var_MMSE_train)
        self.train_data = DataLoader(train_dataset, batch_size=batch_size)
        
        val_dataset = CF_Dataset(x_val,edge_weight_val,Tx_val,noise_info_val, Xhat_MMSE_val, var_MMSE_val)
        self.val_data = DataLoader(val_dataset, batch_size=batch_size)
        

    def getTrainData(self):
        return self.train_data
    
    def getValData(self):
        return self.val_data
    



class Data_loader_test (object) :
    
    def __init__(self, Nt, Nr, samples, batch_size, snr, constellation):
    
        dtype = torch.float64
        
        params = {
            'constellation': constellation,
            'Nt': Nt,
            'Nr': Nr,
            'samples': samples,
            'SNR_dB_min': snr,
            'SNR_dB_max': snr,
            }
        
        # Generate Data
        GD=genData(params)
        H_real_test, tX_test, rX_test, init_feats, edge_weight_test, noise_info, xhat_MMSE,var_MMSE = GD.dataTrain()

        # Convert numpy to torch 
        x_test=torch.from_numpy(init_feats).to(dtype)
        Tx_test = torch.Tensor(tX_test).to(torch.int64)
        xhat_MMSE_test = torch.Tensor(xhat_MMSE).to(dtype)
        var_MMSE_test = torch.from_numpy(var_MMSE).to(dtype)
        
        
        test_dataset =CF_Dataset(x_test,edge_weight_test,Tx_test,noise_info,xhat_MMSE_test,var_MMSE_test)
        self.test_data =DataLoader(test_dataset, batch_size=batch_size)

    
    def getTestData(self):
        return self.test_data