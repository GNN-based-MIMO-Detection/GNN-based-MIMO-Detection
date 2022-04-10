import torch
import numpy as np
import torch.nn as nn

soft_max = nn.Softmax(dim=2)

def QAM_const(constellation):
    mod_n = len(constellation)**2
    sqrt_mod_n = np.int(np.sqrt(mod_n))
    real_qam_consts = np.empty((mod_n), dtype=np.int64)
    imag_qam_consts = np.empty((mod_n), dtype=np.int64)
    for i in range(sqrt_mod_n):
        for j in range(sqrt_mod_n):
                index = sqrt_mod_n*i + j
                real_qam_consts[index] = i
                imag_qam_consts[index] = j
                
    return(constellation[real_qam_consts], constellation[imag_qam_consts])

def joint_indices(indices,constellation):
    real_part, complex_part = np.split(indices, 2, axis=1)
    joint_indices = (len(constellation)*real_part + complex_part).astype(int)
    return joint_indices


def calc_perf(s,x_soft,constellation):
        
    real_QAM_const,imag_QAM_const = QAM_const(constellation)
    x_real, x_imag = np.split(x_soft, 2, -1)
    x_real = np.expand_dims(x_real,-1).repeat(real_QAM_const.size,-1)
    x_imag = np.expand_dims(x_imag,-1).repeat(imag_QAM_const.size,-1)

    x_real = np.power(x_real - real_QAM_const, 2)
    x_imag = np.power(x_imag - imag_QAM_const, 2)
    x_dist = x_real + x_imag
    estim_indices = np.argmin(x_dist, axis=-1)
    
    x_indices = joint_indices(s,constellation)
    ser = np.sum(x_indices!=estim_indices)/x_indices.size
    return ser


def train(model, GEPNet, device, train_dataloader, optimizer, epoch, criterion, user_num, dtype, constellation):
    
    model.train()
    sigma2_list=[]
    x_hat_list=[]
    label_list=[]
    
    for u_feats, edge, label, sigma2, H, y in train_dataloader:
          
        #Initialization
        u_feats=u_feats.to(dtype).to(device)
        edge=edge.to(dtype).to(device)
        label=label.to(device)
        sigma2 = sigma2.to(dtype).to(device)
        H = H.to(dtype).to(device)
        y = y.to(dtype).to(device)
        x_hats = []
        loss=0
        
        
        #Prediction
        y_pred = GEPNet.forward(model, H, y, sigma2, u_feats, edge)
        y_pred_soft = soft_max(y_pred)
        
        for idx_user in range (user_num): 
            loss_each =criterion(y_pred[:,idx_user,:],label[:,idx_user]) 
            loss=loss+loss_each
            constellation_expanded = np.expand_dims(constellation, axis=1)
            x_hat = np.matmul(y_pred_soft[:,idx_user,:].to('cpu').detach().numpy(), constellation_expanded)
            x_hats.append(x_hat) 
            
            
        x_hats = np.concatenate(x_hats,1)
        x_hat_list.append(x_hats)
        label_list.append(label.to('cpu').detach().numpy())
        sigma2_list.append(sigma2)
        
        #Backpropagation
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()
        
    x_hat_list = np.concatenate(x_hat_list,0)
    label_list = np.concatenate(label_list,0)
    
    #SER and loss calculations
    train_SER = calc_perf(label_list,x_hat_list,constellation)
    train_acc = 1-train_SER
    avg_SNR =  10*(torch.log(u_feats.shape[1] / (y.shape[1]*torch.mean(torch.cat(sigma2_list)*2)))/torch.log(torch.tensor(10)))
    loss_=loss.to('cpu').detach().numpy()
    
    return (loss_/user_num),train_acc,train_SER,avg_SNR

def evaluate(model, GEPNet, device, test_dataloader,criterion, user_num, dtype, constellation):

    model.eval()
    
    with torch.no_grad():
        x_hat_list=[]
        label_list=[]
        loss=0
        for u_feats, edge, label, sigma2, H, y in test_dataloader:
      
            u_feats=u_feats.to(dtype).to(device)
            edge=edge.to(dtype).to(device)
            label=label.to(device)
            sigma2 = sigma2.to(dtype).to(device)
            H = H.to(dtype).to(device)
            y = y.to(dtype).to(device)
          
            x_hats = []
            y_pred = GEPNet.forward(model, H, y, sigma2, u_feats, edge)
            y_pred_soft = soft_max(y_pred)
            
            for idx_user in range (user_num):            
                loss_each =criterion(y_pred[:,idx_user,:],label[:,idx_user]) 
                loss=loss+loss_each
                constellation_expanded = np.expand_dims(constellation, axis=1)
                x_hat = np.matmul(y_pred_soft[:,idx_user,:].to('cpu').detach().numpy(), constellation_expanded)
                x_hats.append(x_hat) 
                
                
            x_hats = np.concatenate(x_hats,1)
            x_hat_list.append(x_hats)
            label_list.append(label.to('cpu').detach().numpy())
        
        x_hat_list = np.concatenate(x_hat_list,0)
        label_list = np.concatenate(label_list,0)
    
        test_SER = calc_perf(label_list,x_hat_list,constellation)
        test_acc = 1-test_SER
        loss_=loss.to('cpu').detach().numpy()
        
    return (loss_/user_num),test_acc,test_SER
