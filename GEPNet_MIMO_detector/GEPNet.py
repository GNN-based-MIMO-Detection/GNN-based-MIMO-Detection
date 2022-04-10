import torch
from EP import EP


class GEPNet (object):
    def __init__(self, iter_GEP, beta ,Nr , su, num_neuron,cons, device, dtype):  
        
        self.iter_GEP = iter_GEP
        self.beta = beta
        self.hid_neuron_size= num_neuron
        self.cons = torch.from_numpy(cons).to(dtype).to(device)
        self.device = device
        self.dtype = dtype
        
    def forward(self, model, H, y, sigma2, u_feats, edge):
        
        # Initialization
        user_num = H.shape[2]
        lamda= (torch.ones((u_feats.shape[0],user_num,1))*2).to(self.dtype).to(self.device)
        gamma= torch.zeros((u_feats.shape[0],user_num,1)).to(self.dtype).to(self.device)
        
        mean_ab = None
        var_ab = None
        p_GNN = None
        read_gru=None
        gru = torch.zeros((u_feats.shape[0],user_num,self.hid_neuron_size)).to(self.dtype).to(self.device)
        EP_model = EP(H, y, sigma2, user_num, self.cons,u_feats.shape[0])
        
        # GEPNet iteration
        for iteration in range(self.iter_GEP):
            diag_lamda = torch.zeros((u_feats.shape[0],user_num,user_num)).to(self.dtype).to(self.device)
            #Perform EP
            mean_ab, var_ab, lamda, gamma = EP_model.performEP(self.beta,diag_lamda,p_GNN, mean_ab, var_ab, lamda, gamma, iteration)
            
            #Perform GNN
            p_GNN, read_gru,gru = model(read_gru,gru,u_feats,edge,sigma2, mean_ab, var_ab, iteration)

        return p_GNN
            
            