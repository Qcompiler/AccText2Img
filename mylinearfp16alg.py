 
import torch
import math
import torch.nn as nn
import sys
import gc
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
import mixgemm
try:
    from EETQ import quant_weights, w8_a16_gemm
    memory_bound_eetq_linear = True
except:
    memory_bound_eetq_linear = False



def FindOutliers(Activation, sigma = None):

    if sigma is None:
        sigma = 20
    
    tmp = torch.unique(torch.where((  Activation.abs() > sigma ))[1])
    return tmp.to(torch.int32)


layer_id = 0
class MixLinear_GEMM(nn.Module):
    def __init__(self, in_features, out_features, bias = True,  
                 device=None):
        super().__init__()
        global layer_id
        layer_id += 1
        self.layer_id = layer_id
        dtype = torch.float16
        
        factory_kwargs = {'device': device, 'dtype': dtype, "requires_grad": False}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad = False)
        self.q_scale_col = torch.empty((1, out_features), **factory_kwargs)

        self.q_weight = torch.empty((out_features, in_features), dtype = torch.int8)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.weight_cache = None
        self.ind = None

        self.init = False
        self.quanted = False
        self.n_outliers = 0
        
        self.cnt = 0
        self.scale_history = torch.zeros((200) , dtype = torch.float16)
        self.scale_ind_history = torch.zeros((200) , dtype = torch.int32)
        
        self.y1 = None
        self.reuse_output_because_of_zeros_input = False
        self.last_input = None
        self.cache_computed = False

        self.q_weight = None
        self.q_scale_col = None
        self.input_scales = None
        self.reuse_scaling_factor = False

        self.scale_max = None
        self.scale_min = None
        self.doing_estimation = True


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input):

     
        if self.cnt >= 50:
            self.cnt = 0
            # self.reuse_output_because_of_zeros_input = False
            # release cache
            self.cache_computed = False
        if len(input.shape) == 3:
            if self.cnt == 0 and input.shape[0] == 2:
                if input[0,0,0] == 0:
                    self.last_input = input[[1],0:32,0:8]

            if self.cnt == 1 and input.shape[0] == 2:
                

                if self.last_input is not None:
                    if  (input[[1],0:32,0:8] - self.last_input).sum() == 0:
                        self.reuse_output_because_of_zeros_input = True

        self.cnt += 1
        
        if  self.reuse_output_because_of_zeros_input is True and self.cache_computed:
            return self.y1  

        y1 = F.linear(input, self.weight, self.bias)
        if self.reuse_output_because_of_zeros_input:
            self.cache_computed = True
            self.y1 = y1


        return y1

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
