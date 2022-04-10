import torch
import torch.nn as nn


class EP (object):
    def __init__(self, H, y, sigma2, user_num, constellation,batch_size):   
        self.H = H
        self.y = y
        self.sigma2 = sigma2
        self.user_num = user_num
        self.batch_size = batch_size
        self.constellation = constellation
        self.soft_max = nn.Softmax(dim=2)        
        self.constellation_expanded = constellation.tile(self.batch_size,1).unsqueeze(2)
        self.constellation_expanded_transpose = constellation.tile(self.batch_size,user_num,1)

    def calculate_mean_var(self,pyx):
        mean = torch.matmul(pyx, self.constellation_expanded)
        var = torch.square(torch.abs(self.constellation_expanded_transpose - mean))
        var = torch.mul(pyx, var) 
        var = torch.sum(var, axis=2)
        return torch.squeeze(mean), var
            

    def LMMSE(self,diag_lamda, H, y, sigma2, lamda, gamma):
        HtH = torch.matmul(H.permute(0,2,1), H)
        Hty = torch.squeeze(torch.matmul(H.permute(0,2,1), torch.unsqueeze(y,2)))
        torch.einsum('ijj->ij',diag_lamda)[...] = lamda*sigma2
        var = (torch.linalg.inv(HtH + diag_lamda )) 
        mean = (Hty) + gamma* sigma2
        mean = torch.matmul(var,torch.unsqueeze(mean,2))
        var = var* torch.unsqueeze(sigma2,2)
        del diag_lamda
        del HtH
        del Hty
        return mean, var

    def performEP(self,beta_NEEP,diag_lamda, p_y_x_GNN, mean_ab_prev, var_ab_prev, lamda_prev, gamma_prev, iter_num):
    
        if (iter_num == 0):
            lamda = lamda_prev.squeeze()
            gamma = gamma_prev.squeeze()
        else:
            # Calculating mean and variance of \hat{P}_x_y
            p_y_x_GNN = self.soft_max(p_y_x_GNN)
            mean_b, var_b = self.calculate_mean_var(p_y_x_GNN)
            var_b = torch.clamp(var_b, 1e-13, None)
            
            # Calculating new lamda and gamma
            lamda = 1/var_b - 1/var_ab_prev
            gamma = mean_b /var_b - mean_ab_prev/ var_ab_prev        
            
            
            # Avoiding negative lamda and gamma
            if torch.any(lamda < 0):
                indices = torch.where(lamda<0)
                lamda[indices]=lamda_prev[indices]
                gamma[indices]=gamma_prev[indices]
                
                
            # Updating lamda and gamma
            lamda = beta_NEEP*lamda_prev + (1-beta_NEEP)*lamda
            gamma = beta_NEEP*gamma_prev + (1-beta_NEEP)*gamma
        
        mean_mmse, var_mmse = self.LMMSE(diag_lamda,self.H, self.y, self.sigma2, lamda, gamma)

        var_ab = 1 / (1/torch.diagonal(var_mmse, dim1=1, dim2=2) - lamda )
        var_ab = torch.clamp(var_ab, 1e-13, None)
            
        mean_ab = torch.squeeze(mean_mmse)/torch.diagonal(var_mmse, dim1=1, dim2=2) - gamma
        mean_ab = var_ab * mean_ab
    

        return  mean_ab, var_ab, lamda, gamma