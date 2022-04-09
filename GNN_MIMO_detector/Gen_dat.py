import numpy as np

class genData(object):
    def __init__(self, params):
        self.batch_size = params['samples']
        self.constellation = params['constellation']
        self.Nr = params['Nr']
        self.Nt = params['Nt']
        self.SNR_dB_min = params['SNR_dB_min']
        self.SNR_dB_max = params['SNR_dB_max']
    
        
    def dataTrain(self):
        
        # Generate random indices
        s = np.random.randint(low=0, high=np.shape(self.constellation)[0], size=[self.batch_size,2 * self.Nt])
        
        # Map indices to constellations
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
        noise = np.sqrt(sigma2 / 2).repeat(2 * self.Nr,1) * np.random.randn(self.batch_size, 2 * self.Nr)
        
        # Generate corrupted received signal
        y_noise_real = y_real + noise
        
        # Calculate real-valued noise variance
        sigma2 = sigma2/2
        
        # Compute MMSE
        xhat_MMSE, var_MMSE = self.MMSE(x_real,y_noise_real,H_real,sigma2)
        
        # Calculate initial and edge features of the GNN
        init_feats,edge_i_j_feats= self.Feature_gens(y_noise_real,H_real,sigma2,x_real)        
        edge_i_j_feats = edge_i_j_feats.reshape(self.batch_size,-1)
        
        # Get the symbol indices
        for idx,i in enumerate(self.constellation):
            indices=np.where(x_real==i)
            x_real[indices[0].tolist(),indices[1].tolist()]=idx

        return H_real,x_real,y_noise_real,init_feats,edge_i_j_feats,sigma2,xhat_MMSE, np.diagonal(var_MMSE,0,1,2)



    def MMSE(self,x_real,y_noise_real,H_real,sigma2):
        noise_list = []
        var_MMSE=[]
        # Projected channel output
        Hty = np.squeeze(np.matmul(np.transpose(H_real,[0,2,1]), np.expand_dims(y_noise_real,2)))

        # Gramian of transposed channel matrix
        HtH = np.matmul(np.transpose(H_real,[0,2,1]), H_real)
        # Inverse Gramian
        noise_list = [ sigma2[idx]*np.identity(x_real.shape[1]) for idx in range (sigma2.shape[0]) ]
        noiseDiag=np.concatenate(noise_list).reshape(-1,x_real.shape[1],x_real.shape[1])
        HtHinv = np.linalg.inv(HtH + noiseDiag)

        # MMSE Detector
        xhat = np.squeeze(np.matmul(HtHinv, np.expand_dims(Hty,2)))

        var_MMSE = [HtHinv[idx,:,:]*sigma2[idx] for idx in range(sigma2.shape[0])  ]
        var_MMSE=np.concatenate(var_MMSE).reshape(-1,x_real.shape[1],x_real.shape[1])
        
        return xhat,var_MMSE
            


    def Feature_gens(self,y,H,noiseLevel,x_hats):
        edge_i_j_feats = np.ones((self.batch_size,self.Nt*2,(self.Nt*2-1)))
        
        
        yTh = np.matmul(np.expand_dims(y,2).transpose(0,2,1),H ) 
        hTh = np.matmul(H.transpose(0,2,1),H)
        diag_hTh = np.expand_dims(hTh.diagonal(0,1,2),2).transpose(0,2,1)
        noise_arr = np.tile(np.expand_dims(noiseLevel,2),[1,1,self.Nt*2])
        init_feats = np.concatenate((yTh,-1*diag_hTh,noise_arr),1) 
        init_feats = init_feats.transpose(0,2,1)
        
        for u_idx  in range(self.Nt*2):
            t=0;
            for j_idx  in range(self.Nt*2):
                if np.not_equal(j_idx, u_idx):
                    edge_i_j_feat = -1*np.matmul(np.expand_dims(H[:,:,j_idx],2).transpose(0,2,1) ,np.expand_dims(H[:,:,u_idx],2))
                    edge_i_j_feats[:,u_idx,t] = np.squeeze(edge_i_j_feat);
                    t=t+1;
        
        return init_feats,edge_i_j_feats
        