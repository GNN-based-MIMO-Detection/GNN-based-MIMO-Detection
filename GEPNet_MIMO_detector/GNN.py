import torch
import torch.nn as nn
from time_distributed import TimeDistributed,TimeDistributed_GRU
from get_idx_dicts import get_idx_dicts

class GNN(nn.Module):
    def __init__(self, num_iter, num_neuron, num_feature_su, num_classes,dropout_rate):
        super(GNN,self).__init__()
        
        Su =num_feature_su
        self.num_iter_GNN = num_iter
        self.num_neuron = num_neuron
        
        # MLP for nodes initialization
        self.fc1a=TimeDistributed(nn.Linear,True,3,int(Su))
        
        # MLP for factor nodes
        self.fc2a=TimeDistributed(nn.Linear,True,Su*2+2,num_neuron)
        self.fc2b=TimeDistributed(nn.Linear,True,num_neuron,int(num_neuron/2))
        self.fc2c=TimeDistributed(nn.Linear,True,int(num_neuron/2),Su)
                
        # MLP for GRU
        self.gru=TimeDistributed_GRU(torch.nn.GRUCell, Su+2,num_neuron)
        #self.fc4a=TimeDistributed(nn.Linear,True,num_neuron,Su)
        
        # MLP for readout
        self.fc3a=TimeDistributed(nn.Linear,True,Su,num_neuron)
        self.fc3b=TimeDistributed(nn.Linear,True,num_neuron,int(num_neuron/2))
        self.fc3c=TimeDistributed(nn.Linear,True,int(num_neuron/2),num_classes)
        
        self.fc4=TimeDistributed(nn.Linear,True,num_neuron,Su)
        
        # Activation functions
        self.a1=nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self,read_gru,gru_init, init_features, edge_weight, noise_info, x_hat_EP, variance_EP, GNN_iter):
        '''===========================custom layers========================'''
  
        def NN_iterations(init_feats, read_gru,gru_hidden, idx_iter,GNN_iter,temp_a,temp_b):
            
            if idx_iter == 0 and GNN_iter==0:
                x_init = init_feats
                init_nodes = self.fc1a(x_init)# sample, nodes, features
            else:
                init_nodes = read_gru
                
            
            # Interferance probability feature calculations
            slicing1 = init_nodes[:,temp_a,:]
            slicing2 = init_nodes[:,temp_b,:]
           
            edge_slicing = edge_weight[:,:,None]
            noise_slicing = noise_info.repeat(1,len(temp_a))[:,:,None]
            combined_slicing = torch.cat((slicing1,slicing2,edge_slicing,noise_slicing),2)
            
            messages_from_i_to_j = self.a1(self.fc2a(combined_slicing))
            messages_from_i_to_j = self.a1(self.fc2b(messages_from_i_to_j))
            messages_from_i_to_j = self.a1(self.fc2c(messages_from_i_to_j))
           
            # Interferance probabilitie features summation
            step = init_feats.shape[1] - 1
            sum_messages=[]
            for i in range(0,messages_from_i_to_j.shape[1],step):
                  sum_messages.append(torch.unsqueeze(torch.sum((messages_from_i_to_j[:,i:i+step,:]),1),1))
            sum_messages  =torch.cat(sum_messages,1)
            
            
            # GRU and outputs
            gru_feats_concat = torch.cat((sum_messages,torch.unsqueeze(x_hat_EP,2),torch.unsqueeze(variance_EP,2)),2)
            gru_out = self.gru(gru_feats_concat,gru_hidden)
            read_gru = self.fc4(gru_out)
            
            return read_gru,gru_out
                    
        # Initializations
        init_feats = init_features
        gru_hidden = gru_init
        temp_a,temp_b = get_idx_dicts(init_features.shape[1])
        
        for idx_iter in range(self.num_iter_GNN):
            read_gru,gru_hidden = NN_iterations(init_feats, read_gru,gru_hidden, idx_iter,GNN_iter,temp_a,temp_b)
            
        R_out1 = self.fc3a(read_gru)
        R_out2 = self.fc3b(R_out1) 
        p_y_x=self.fc3c(R_out2)
        
        return p_y_x, read_gru,gru_hidden
    