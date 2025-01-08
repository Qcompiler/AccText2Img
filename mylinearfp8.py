 
import torch
import math
import torch.nn as nn
import sys
import gc
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
import mixgemm


layer_id = 0
class MixLinear_FP8GEMM(nn.Module):
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

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        

        self.init = False

        self.cnt = 0
        self.scale_weight  = None


        

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
         
        weight_dtype = torch.float8_e4m3fn
        if not input.is_contiguous():
            return F.linear(input, self.weight, self.bias)
            input = input.contiguous()
        if   self.init is False:
            self.scale_weight = torch.ones((1), device=input.device, dtype=torch.float32)
            self.w = self.weight.t().to(weight_dtype)
            self.init = True

        out_shape =  input.shape[:-1] + (self.out_features, )
        
        if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            # if len(input.shape) == 3:
            if weight_dtype == torch.float8_e4m3fn:
                inn = input.reshape(-1, input.shape[-1]).to(torch.float8_e5m2)
            else:
                inn = input.reshape(-1, input.shape[-1]).to(torch.float8_e4m3fn)
            

            
            scale_input = self.scale_weight


            out_dtype = torch.float16

            if self.bias is not None:
                o = torch._scaled_mm(inn, self.w, out_dtype=out_dtype, bias=self.bias, scale_a=scale_input, scale_b=self.scale_weight)
            else:
                o = torch._scaled_mm(inn, self.w, out_dtype=out_dtype, scale_a=scale_input, scale_b=self.scale_weight)

        debug = False
        if debug:

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            tmp = input.to(torch.float8_e5m2)
            end_event.record()
            torch.cuda.synchronize()
            ms_cast = start_event.elapsed_time(end_event)

            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            o  = F.linear(input, self.weight, self.bias)
            end_event.record()
            torch.cuda.synchronize()
            ms_fp = start_event.elapsed_time(end_event)


            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            o = torch._scaled_mm(inn, self.w, out_dtype=out_dtype, bias=self.bias, scale_a=scale_input, scale_b=self.scale_weight)
            end_event.record()
            torch.cuda.synchronize()
            ms_fp8 = start_event.elapsed_time(end_event)
            
            print("ms fp8 = %.4f  ms cast = %.4f ms_fp16 = %.4f, acc = %.4f"%(ms_fp8, ms_cast, ms_fp, ms_fp/(ms_fp8 + ms_cast)))
            
        return o[0].reshape(out_shape)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
