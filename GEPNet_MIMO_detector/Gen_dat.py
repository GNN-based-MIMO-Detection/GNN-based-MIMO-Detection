import numpy as np


class genData(object):
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.constellation = params['constellation']
        self.Nr = params['Nr']
        self.Nt = params['Nt']
        self.SNR_dB_min = params['SNR_dB_min']
        self.SNR_dB_max = params['SNR_dB_max']
        self.iter_EP_gD = params['iter_EP_gD']
        self.compare = params['compare']
    
    
    def dataTrain(self):
        
        # Generate random indices
        s = np.random.randint(low=0, high=np.shape(self.constellation)[0], size=[self.batch_size,2 * self.Nt])
        
        # Map indices to constellations
        self.s=s
        x_real = self.constellation[s]
        
        # Generate channel matrix
        Hr = np.random.randn(self.batch_size,self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
        Hi = np.random.randn(self.batch_size,self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
        H_real = np.concatenate([np.concatenate([Hr, -Hi], axis=2), np.concatenate([Hi, Hr], axis=2)], axis=1)
        
        # Generate non-corrupted received signal
        y_real = np.squeeze(np.matmul(H_real, np.expand_dims(x_real,2)))
        
        # Generate a range of training SNRs
        snr_db = np.random.uniform(self.SNR_dB_min, self.SNR_dB_max,[self.batch_size,1])
        
        # Calculatet noise variance
        sigma2 =  (2 * self.Nt) / (np.power(10, snr_db / 10) * (2*self.Nr))
        noise = np.sqrt(sigma2 / 2) * np.random.randn(self.batch_size, 2 * self.Nr)
        
        # Generate corrupted received signal
        y_noise_real = y_real + noise
        
        # Calculate real-valued noise variance
        sigma2 = sigma2/2
        
        SER_mmse            =None
        SER_EP              =None
        if self.compare:
            # Compute MMSE
            SER_mmse, xhat_MMSE, var_MMSE = self.MMSE(x_real,y_noise_real,H_real,sigma2)
            # Compute EP
            SER_EP =  self.EP(x_real,y_noise_real,H_real,sigma2,self.iter_EP_gD)
            
            
        # Calculate initial and edge features of the GNN
        init_feats,edge_i_j_feats= self.Feature_gens(y_noise_real,H_real,sigma2,x_real)
        edge_i_j_feats = edge_i_j_feats.reshape(self.batch_size,-1)
        
        
        # Get the symbol indices
        for idx,i in enumerate(self.constellation):
            indices=np.where(x_real==i)
            x_real[indices[0].tolist(),indices[1].tolist()]=idx
            
        return H_real,x_real,y_noise_real,init_feats,edge_i_j_feats,sigma2,SER_mmse,SER_EP


    def joint_indices(self,indices):
        real_part, complex_part = np.split(indices, 2, axis=1)
        joint_indices = (len(self.constellation)*real_part + complex_part).astype(int)
        return joint_indices
            
    def QAM_const(self):
        mod_n = len(self.constellation)**2
        sqrt_mod_n = np.int(np.sqrt(mod_n))
        real_qam_consts = np.empty((mod_n), dtype=np.int64)
        imag_qam_consts = np.empty((mod_n), dtype=np.int64)
        for i in range(sqrt_mod_n):
            for j in range(sqrt_mod_n):
                    index = sqrt_mod_n*i + j
                    real_qam_consts[index] = i
                    imag_qam_consts[index] = j
                    
        return(self.constellation[real_qam_consts], self.constellation[imag_qam_consts])
    
    
    def calc_perf(self,x_soft):
        
        real_QAM_const,imag_QAM_const = self.QAM_const()
        x_real, x_imag = np.split(x_soft, 2, -1)
        x_real = np.expand_dims(x_real,-1).repeat(real_QAM_const.size,-1)
        x_imag = np.expand_dims(x_imag,-1).repeat(imag_QAM_const.size,-1)
    
        x_real = np.power(x_real - real_QAM_const, 2)
        x_imag = np.power(x_imag - imag_QAM_const, 2)
        x_dist = x_real + x_imag
        estim_indices = np.argmin(x_dist, axis=-1)
        
        
        x_indices = self.joint_indices(self.s)
        ser = np.sum(x_indices!=estim_indices)/x_indices.size
        return ser


    def MMSE(self,x_real,y_noise_real,H_real,sigma2):
        # Projected channel output
        Hty = np.squeeze(np.matmul(np.transpose(H_real,[0,2,1]), np.expand_dims(y_noise_real,2)))

        # Gramian of transposed channel matrix
        HtH = np.matmul(np.transpose(H_real,[0,2,1]), H_real)
        # Inverse Gramian
        HtHinv = np.linalg.inv(
            HtH + np.reshape(sigma2, [-1, 1,1]) * np.expand_dims(np.eye(H_real.shape[2]),0).repeat(H_real.shape[0], axis=0))

        # MMSE Detector
        xhat = np.squeeze(np.matmul(HtHinv, np.expand_dims(Hty,2)))
        
        SER = self.calc_perf(xhat)
        
        var_MMSE = HtHinv*sigma2[0,0]
        
        return SER,xhat,var_MMSE
    
        

    def Feature_gens(self,y,H,noiseLevel,x_hats):
        # init_feats = np.ones((self.batch_size,self.Nt*2,3))
        edge_i_j_feats = np.ones((self.batch_size,self.Nt*2,(self.Nt*2-1)))
        
        
        yTh = np.matmul(np.expand_dims(y,2).transpose(0,2,1),H ) 
        hTh = np.matmul(H.transpose(0,2,1),H)
        diag_hTh = np.expand_dims(hTh.diagonal(0,1,2),2).transpose(0,2,1)
        noise_arr = np.tile(np.expand_dims(noiseLevel,2),[1,1,self.Nt*2])
        init_feats = np.concatenate((yTh,-1*diag_hTh,noise_arr),1) ## I PUT MINUS HERE!!!!!!!!!!!!!
        init_feats = init_feats.transpose(0,2,1)
        
        for u_idx  in range(self.Nt*2):
            t=0;
            for j_idx  in range(self.Nt*2):
                if np.not_equal(j_idx, u_idx):
                    edge_i_j_feat = -1*np.matmul(np.expand_dims(H[:,:,j_idx],2).transpose(0,2,1) ,np.expand_dims(H[:,:,u_idx],2))
                    edge_i_j_feats[:,u_idx,t] = np.squeeze(edge_i_j_feat);
                    t=t+1;
        
        return init_feats,edge_i_j_feats
        
        
        
    
    def EP(self,x_real,y_noise_real,H_real,sigma2,num_iter):

        user_num = self.Nt *2
        lamda_init = np.ones((H_real.shape[0],self.Nt*2))*2
        gamma_init = np.zeros((H_real.shape[0],self.Nt*2))
        sigma2 = np.mean(sigma2)
        H = H_real
        y = y_noise_real
        constellation_expanded = np.expand_dims(self.constellation, axis=1)
        constellation_expanded= np.repeat(constellation_expanded[None,...],H.shape[0],axis=0)
        
    
        def calculate_mean_var(pyx, constellation_expanded):
            constellation_expanded_transpose = np.repeat(constellation_expanded.transpose(0,2,1), user_num, axis=1)
            mean = np.matmul(pyx, constellation_expanded)
            var = np.square(np.abs(constellation_expanded_transpose - mean))
            var = np.multiply(pyx, var) 
            var = np.sum(var, axis=2)
            
            return np.squeeze(mean), var
        
        def calculate_pyx( mean, var, constellation_expanded):
            constellation_expanded_transpose = np.repeat(constellation_expanded.transpose(0,2,1), user_num, axis=1)
            arg_1 = np.square(np.abs(constellation_expanded_transpose - np.expand_dims(mean,2)))
            log_pyx = (-1 * arg_1)/(2*np.expand_dims(var,2))
            log_pyx = log_pyx - np.expand_dims(np.max(log_pyx,2),2)
            p_y_x = np.exp(log_pyx)
            p_y_x = p_y_x/(np.expand_dims(np.sum(p_y_x, axis=2),2) + np.finfo(float).eps)
            
            return p_y_x

        def LMMSE( H, y, sigma2, lamda, gamma):
            HtH = np.matmul(np.transpose(H,[0,2,1]), H)
            Hty = np.squeeze(np.matmul(np.transpose(H,[0,2,1]), np.expand_dims(y,2)))
            diag_lamda = np.zeros((HtH.shape[0],user_num,user_num))
            np.einsum('ijj->ij',diag_lamda)[...] = lamda
            var = np.linalg.inv(HtH + diag_lamda * sigma2)
            mean = (Hty) + gamma* sigma2
            mean = np.matmul(var,np.expand_dims(mean,2))
            var = var* sigma2
            return mean, var

        lamda = lamda_init
        gamma = gamma_init
    
        for iteration in range(num_iter):
                                  
            mean_mmse, var_mmse = LMMSE(H, y, sigma2, lamda, gamma)

            # Calculating mean and variance of P_y_x
            diag_mmse=np.diagonal(var_mmse, axis1=1, axis2=2)
            var_ab = (diag_mmse/ (1 - diag_mmse*lamda )) +np.finfo(float).eps
            
            # var_ab = 1 / (1/np.diagonal(var_mmse, axis1=1, axis2=2) - lamda )
            mean_ab = (np.squeeze(mean_mmse)/np.diagonal(var_mmse, axis1=1, axis2=2) - gamma) 
            mean_ab = var_ab * mean_ab

            # Calculating P_y_x
            p_y_x_ab = calculate_pyx (mean_ab, var_ab, constellation_expanded)

            # Calculating mean and variance of \hat{P}_x_y
            mean_b, var_b = calculate_mean_var(p_y_x_ab, constellation_expanded)
            var_b = np.clip(var_b, 1e-13, None)

            # Calculating new lamda and gamma
            lamda_new = ((var_ab-var_b) / var_b )/ var_ab
            
            # lamda_new = 1/ var_b - 1/var_ab
            gamma_new = mean_b /var_b - mean_ab/ var_ab

            # Avoiding negative lamda and gamma
            if np.any(lamda_new < 0):
                indices = np.where(lamda_new<0)
                lamda_new[indices]=lamda[indices]
                gamma_new[indices]=gamma[indices]

            # Appliying updating weight
            lamda = lamda*0.9 + lamda_new*0.1
            gamma = gamma*0.9 + gamma_new*0.1
            
        
        SER = self.calc_perf(mean_b)
        
        
        return SER
