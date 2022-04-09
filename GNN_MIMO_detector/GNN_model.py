import torch
import torch.nn as nn
from time_distributed import TimeDistributed,TimeDistributed_GRU

class GNN(nn.Module):
    def __init__(self, num_iter, num_neuron, num_feature_su, num_classes, user_num, with_mmse,dropout_rate):
        # print("gnn model: Scotti || MMSE:", with_mmse);
        super(GNN,self).__init__()
         # '''================define custom variables========================'''
         
        self.with_mmse = with_mmse
        self.num_iter = num_iter
        self.user_num = user_num
    
        Su =num_feature_su
        # MLP for node initializations
        if with_mmse == True:
            self.fc1a=TimeDistributed(nn.Linear,True,5,int(Su))
        else:
            self.fc1a=TimeDistributed(nn.Linear,True,3,int(Su))
            
        
        # MLP for m_i to j
        self.fc2a=TimeDistributed(nn.Linear,True,Su*2+2,num_neuron)
        self.dropout1= nn.Dropout(p=0.1)
        self.fc2b=TimeDistributed(nn.Linear,True,num_neuron,int(num_neuron/2))
        self.fc2c=TimeDistributed(nn.Linear,True,int(num_neuron/2),Su)
        # self.fc2d=TimeDistributed(nn.Linear,True,int(num_neuron/2),Su)
        
        # MLP for after GRU
        self.fc3a=TimeDistributed(nn.Linear,True,Su,num_neuron)
        self.dropout2= nn.Dropout(p=0.2)
        self.fc3b=TimeDistributed(nn.Linear,True,num_neuron,int(num_neuron/2))
        self.fc3c=TimeDistributed(nn.Linear,True,int(num_neuron/2),num_classes)
        # self.fc3d=TimeDistributed(nn.Linear,True,int(num_neuron/2),num_classes)
        
        # self.gru_small=TimeDistributed_GRU(torch.nn.GRUCell, Su,Su)
        
        self.gru=TimeDistributed_GRU(torch.nn.GRUCell, Su,num_neuron)
        self.f4=TimeDistributed(nn.Linear,True,num_neuron,Su)
        
        self.a1=nn.ReLU()
        self.a2=nn.ReLU()
        self.a3=nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        # self.dropout= nn.Dropout(p=dropout_rate)
        
        

    def forward(self, init_features, edge_weight, noise_info, hx_init, cons, x_hat_MMSE, var_MMSE):
        '''===========================custom layers========================'''
 
        def NN_iterations(initfeats,idx_iter,gru_read,hx):
            
            if idx_iter == 0:
                if self.with_mmse:
                    #print("MMSE called here")
                    x_init = torch.cat([initfeats,torch.unsqueeze(x_hat_MMSE,2),torch.unsqueeze(var_MMSE,2)],2)
                else:
                    x_init = initfeats
                    
                init_nodes = self.fc1a(x_init)    # sample, nodes, features
            else:
                init_nodes = gru_read
           
            
            # Interferance probability feature calculation
            slicing1 = init_nodes[:,temp_a,:]
            slicing2 = init_nodes[:,temp_b,:]
           
            edge_slicing = edge_weight[:,:,None]
            noise_slicing = noise_info.repeat(1,len(temp_a))[:,:,None]
            combined_slicing = torch.cat((slicing1,slicing2,edge_slicing,noise_slicing),2)
                       
            messages_from_i_to_j = self.a2(self.fc2a(combined_slicing))        
            # messages_from_i_to_j= self.dropout1(messages_from_i_to_j)
            messages_from_i_to_j = self.a2(self.fc2b(messages_from_i_to_j))    
            # messages_from_i_to_j= self.dropout1(messages_from_i_to_j)
            messages_from_i_to_j = self.a2(self.fc2c(messages_from_i_to_j))    
           
            # Interferance probability features summation
            step = self.user_num - 1
                   
            sum_messages=[]
            for i in range(0,messages_from_i_to_j.shape[1],step):
                  sum_messages.append(torch.unsqueeze(torch.sum((messages_from_i_to_j[:,i:i+step,:]),1),1))
            sum_messages  =torch.cat(sum_messages,1)
           
            # GRU and outputs
            gru_out = self.gru(sum_messages,hx)
            gru_read = self.f4(gru_out) 
           
            # gru_out = self.gru_small(sum_messages,init_nodes)
            # gru_read = gru_out
            
            return gru_read,gru_out
        
        #slicing parameters
        # p_x_y_saved=[]
        temp_a=[]
        temp_b=[]
        for i in range(self.user_num):
            for j in range(self.user_num):
                if  i!=j:
                    temp_a.append(i)
                    temp_b.append(j)
                    
        # Placeholders and Initializations
        hx = hx_init
        initfeats = init_features
        xhat_MMSE = x_hat_MMSE
        var_MMSE = var_MMSE
        gru_read=None
            
        for idx_iter in range(self.num_iter):
            gru_read,hx = NN_iterations(initfeats,idx_iter,gru_read,hx)
                        
            
        # gru_out = self.f4(gru_out)
        R_out1 = self.fc3a(gru_read)
        # R_out1= self.dropout2(R_out1)
        R_out2 = self.fc3b(R_out1) 
        # R_out2= self.dropout2(R_out2)
        z=self.fc3c(R_out2)
        
        # z_reshaped = z.contiguous().view(-1,z.size(-1))  # (samples * nodes, features)
        # z_normalized = self.softmax(z_reshaped)
        # z_normalized = z_reshaped
   
        # p_x_y = z_normalized.contiguous().view(-1, z.size(1), z.size(-1))
                
        #     p_x_y_saved.append(p_x_y)
        # p_x_y_final =  torch.cat(p_x_y_saved,0)
    
   

        return z
    