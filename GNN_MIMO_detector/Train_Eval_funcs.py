import torch
import numpy as np
import torch.nn as nn
soft_max = nn.Softmax(dim=2)


def train(model, device, train_dataloader, optimizer, epoch, criterion, user_num, dtype,num_neuron,Nr,constellation):
    
    Nr_2 = 2*Nr
    model.train()
    correct_symbols=0
    total_symbols=0
    sigma2_list=[]

    for u_feats, edge, label, noise_info, x_hat_MMSE, var_MMSE in train_dataloader:
        
        # Initializations
        u_feats=u_feats.to(dtype).to(device)
        edge=edge.to(dtype).to(device)
        label=label.to(device)
        noise_info = noise_info.to(dtype).to(device)
        hx_init = torch.zeros(u_feats.shape[0], u_feats.shape[1], num_neuron).requires_grad_().to(device)
        cons = torch.from_numpy(constellation).to(dtype).to(device)
        X_hat_MMSE = x_hat_MMSE.to(dtype).to(device)
        var_MMSE = var_MMSE.to(dtype).to(device)
        predicted = []
        loss=0
        
        # Prediction
        y_pred = model(u_feats,edge,noise_info,hx_init,cons,X_hat_MMSE,var_MMSE)
        # Softmax
        y_pred_soft = soft_max(y_pred)
        
        # Calculate loss and soft symbols
        for idx_user in range (user_num):
            loss_each =criterion(y_pred[:,idx_user,:],label[:,idx_user]) 
            
            loss=loss+loss_each
            
            constellation_expanded = np.expand_dims(constellation, axis=1)
            
            prediction_soft = np.matmul(y_pred_soft[:,idx_user,:].to('cpu').detach().numpy(), constellation_expanded)
            prediction_min_constellation = np.abs(prediction_soft - np.transpose(constellation_expanded))
            prediction_index = np.argmin(prediction_min_constellation, axis=1)
            predicted.append(prediction_index) 
                
        # Backpropagation
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # Calculate SER
        total_symbols+=np.ones(label.to('cpu').detach().numpy().shape).sum()
        correct_symbols+= (np.stack(predicted,0) == label.to('cpu').detach().numpy().transpose()).sum()
        sigma2_list.append(noise_info)
   
    def calc_perf(correct):
        train_acc=float(correct) / (total_symbols)
        train_SER=1-train_acc
        return train_acc,train_SER
    
    train_acc,train_SER=calc_perf(correct_symbols)

    
    loss_=loss.to('cpu').detach().numpy()
    avg_SNR =  10*(torch.log(u_feats.shape[1] / (Nr_2*torch.mean(torch.cat(sigma2_list)*2)))/torch.log(torch.tensor(10)))
    
    return (loss_/user_num),train_acc,train_SER,avg_SNR

def evaluate(model,device,test_dataloader, user_num,dtype,num_neuron,constellation):
    
    model.eval()
    
    with torch.no_grad():
        correct = 0 
        total=0
        for u_feats,edge,label,noise_info,x_hat_MMSE, var_MMSE in test_dataloader:
            
            # Initializations
            u_feats=u_feats.to(dtype).to(device)
            edge=edge.to(dtype).to(device)
            label=label.to(dtype).to(device)
            noise_info = noise_info.to(dtype).to(device)
            hx_init = torch.zeros(u_feats.shape[0], u_feats.shape[1], num_neuron).requires_grad_().to(device)
            cons = torch.from_numpy(constellation).to(dtype).to(device)
            
            X_hat_MMSE = x_hat_MMSE.to(dtype).to(device)
            var_MMSE = var_MMSE.to(dtype).to(device)
            
            predicted = []
            
            y_pred = model(u_feats,edge,noise_info,hx_init,cons,X_hat_MMSE,var_MMSE)
            
            #Softmax
            y_pred_soft = soft_max(y_pred)
            
            # Calculate loss and soft symbols
            for idx_user in range (user_num):            
                constellation_expanded = np.expand_dims(constellation, axis=1)
                prediction_soft = np.matmul(y_pred_soft[:,idx_user,:].to('cpu').detach().numpy(), constellation_expanded)
                prediction_min_constellation = np.abs(prediction_soft - np.transpose(constellation_expanded))
                prediction_index = np.argmin(prediction_min_constellation, axis=1)
                predicted.append(prediction_index) 
    
    # Calculate SER
            total+=np.ones(label.to('cpu').detach().numpy().shape).sum()
            correct += (predicted == label.to('cpu').detach().numpy().transpose()).sum()
        
    test_acc=float(correct) / (total)
    test_SER=1-test_acc
    
    return test_acc,test_SER
