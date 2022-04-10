# import pandas as pd
# import torch
import torch.nn as nn

class TimeDistributed(nn.Module): # input must be num_samples, nodes, features
    def __init__(self, module,batch_first,*args):
        super(TimeDistributed, self).__init__()
        self.module = module(*args)
        self.batch_first = batch_first

    def __call__(self, x):
               

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1,x.size(-1))  # (samples * nodes, features)

        y = self.module(x_reshape)

        # We have to reshape Y
        # if self.batch_first:
        y = y.contiguous().view(-1, x.size(1), y.size(-1))  # (samples, nodes, features)
        # else:
        #     y = y.view(-1, x.size(1), y.size(-1))  # (nodes, samples, output_size)

        return y

class TimeDistributed_GRU(nn.Module): # input must be num_samples, nodes, features
    def __init__(self, module,*args):
        super(TimeDistributed_GRU, self).__init__()
        self.module = module(*args)
        self.reset_parameters()

    def __call__(self, x,hx):

        # if len(x.size()) <= 2:
        #     return self.module(x,hx)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * nodes, input_size)
        hx_reshape = hx.contiguous().view(-1, hx.size(-1))  # (samples * nodes, input_size)

        y = self.module(x_reshape,hx_reshape) #400x128

        y = y.contiguous().view(-1, x.size(1), y.size(-1))  # (samples, nodes, output_size)
        return y
            

    def reset_parameters(self):
        # uniform(self.out_channels, self.weight)
        self.module.reset_parameters()
    
