 
import torch
import math
import torch.nn as nn
import sys
import gc
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
import mixgemm
import mixlib
from vllm import _custom_ops as ops
import vllm._C
import mixgemm_v2
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

        self.q_weight = torch.empty((1, 1), dtype = torch.float8_e4m3fn)

        self.init = False

        self.cnt = 0
        
        self.scale_weight  = None


        self.q_scale_col = None
 
        self.scale_input = None
        

        self.y1 = None
        self.reuse_output_because_of_zeros_input = False
        self.last_input = None
        self.cache_computed = False

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
         
        if input.dtype == torch.float32:
            return F.linear(input, self.weight, self.bias)
        weight_dtype = torch.float8_e4m3fn
        if not input.is_contiguous():
            input = input.contiguous()
            # return F.linear(input, self.weight, self.bias)
            
        

        if   self.init is False:
            x = input.reshape(-1, input.shape[-1]) 

            self.scale_weight = torch.ones((1), device=input.device, dtype=torch.float32)
            self.scale_input = torch.ones((1), device=input.device, dtype=torch.float32)
            
            # self.w = self.weight.t().to(weight_dtype)
            self.init = True

            self.output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            torch.ops._C.dynamic_scaled_fp8_quant(self.output, x, self.scale_input)

            self.q_weight = torch.empty_like(self.weight, dtype=torch.float8_e4m3fn)
            torch.ops._C.static_scaled_fp8_quant(self.q_weight, self.weight, self.scale_weight)

            self.weight.data = self.weight.cpu()
            # del self.weight


            self.find_zeros = torch.zeros((1,), dtype = torch.int32, pin_memory = True)
            self.reuse_output = torch.zeros((1,), dtype = torch.int32, pin_memory = True)
            self.last_input = torch.zeros((1, 32, 8 ), dtype = torch.float16, device = input.device)

            
        out_shape =  input.shape[:-1] + (self.out_features, )
        
        if len(input.shape) == 3:

            if self.cnt >= 50:
                
                self.cnt = 0
                self.find_zeros[0] = 0

                # self.reuse_output_because_of_zeros_input = False
                # release cache
                self.cache_computed = False

        
            if self.cnt == 0 and input.shape[0] == 2:
                mixgemm.find_zeros(self.find_zeros, input, input.shape[0], input.shape[1], input.shape[2], self.last_input)
                

            if self.cnt == 1 and input.shape[0] == 2:
                
                if self.find_zeros[0] == 1:
                    mixgemm.reuse_output(self.reuse_output, input, input.shape[0], input.shape[1],  input.shape[2], self.last_input)

                    if self.reuse_output[0] == 1:
                        self.reuse_output_because_of_zeros_input = True
                    
            self.cnt += 1
            if  self.reuse_output_because_of_zeros_input is True and self.cache_computed:
                return self.y1  
        
        # if self.cnt >  1:
        #     assert  input.dtype == torch.float8_e4m3fn
        if not  input.dtype == torch.float8_e4m3fn:
            torch.ops._C.static_scaled_fp8_quant(self.output, input, self.scale_input)
            input_tensor = self.output
        else:
            input_tensor = input
        bs = 1 if len(input.shape) == 2 else input.shape[0]
        seq = input.shape[0] if len(input.shape) == 2 else input.shape[1]
        y1 =   mixgemm_v2.cutlass_scaled_mm_fp8(bs, seq, 
                                            self.out_features, self.in_features,
                                            input_tensor,
                                            self.q_weight.T,
                                            self.scale_input,
                                            self.scale_weight,
                                            self.bias, len(input.shape))
        if len(input.shape) == 3:
            if self.reuse_output_because_of_zeros_input:
                self.cache_computed = True
                self.y1 = y1
        return y1
    
        return ops.cutlass_scaled_mm(self.output, self.w.T,
                                                out_dtype=torch.float16,
                                                scale_a=self.scale_input,
                                                scale_b=self.scale_weight, bias = self.bias).reshape(out_shape)
        
        if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            # if len(input.shape) == 3:
            if weight_dtype == torch.float8_e4m3fn:
                inn = input.reshape(-1, input.shape[-1]).to(torch.float8_e4m3fn)
            else:
                inn = input.reshape(-1, input.shape[-1]).to(torch.float8_e4m3fn)
            

            x = input.reshape(-1, input.shape[-1]) 
            scale_input = self.scale_weight


            out_dtype = torch.float16
            o = ops.cutlass_scaled_mm(inn, self.w,
                                            out_dtype=torch.float16,
                                            scale_a=scale_input,
                                            scale_b=self.scale_weight, bias = self.bias)
            # if self.bias is not None:
            #     o = torch._scaled_mm(inn, self.w, out_dtype=out_dtype, bias=self.bias, scale_a=scale_input, scale_b=self.scale_weight)
            # else:
            #     o = torch._scaled_mm(inn, self.w, out_dtype=out_dtype, scale_a=scale_input, scale_b=self.scale_weight)

        debug = True
        if debug:

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            tmp = input.reshape(-1, input.shape[-1]).to(torch.float8_e4m3fn)
            end_event.record()
            torch.cuda.synchronize()
            ms_cast = start_event.elapsed_time(end_event)

            


            output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
 
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            for i in range(10):
                
                torch.ops._C.static_scaled_fp8_quant(output, x, scale)
                ops.cutlass_scaled_mm(output, self.w,
                                                out_dtype=torch.float16,
                                                scale_a=scale_input,
                                                scale_b=self.scale_weight)
            end_event.record()
            torch.cuda.synchronize()
            ms_fp8_applied_ai = start_event.elapsed_time(end_event) 


            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            for i in range(10):
                torch._scaled_mm(inn, self.w, out_dtype=out_dtype,   scale_a=scale_input, scale_b=self.scale_weight)
                # print(inn.dtype)
                # exit()
                
                # o = ops.cutlass_scaled_mm(x.to(torch.float8_e4m3fn), self.w,
                #                                 out_dtype=torch.float16,
                #                                 scale_a=scale_input,
                #                                 scale_b=self.scale_weight)
            end_event.record()
            torch.cuda.synchronize()
            ms_fp8 = start_event.elapsed_time(end_event)
            

            M = x.shape[0]
            K = x.shape[1]

            scaleRow = torch.zeros((M, 1) , dtype= torch.float32, device= input.device)
            q_xcache = mixlib.FindRowScaleF32(x, scaleRow, M, K, 8) 
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            for i in range(10):
                # torch._scaled_mm(inn, self.w, out_dtype=out_dtype,   scale_a=scale_input, scale_b=self.scale_weight)
                            
                
                # print(q_xcache.shape)
                # print(self.q_weight.shape)
                # print(scaleRow.shape)
                # print(self.q_scale_col.shape)
                # exit()

                y1 =   ops.cutlass_scaled_mm(
                                                q_xcache,
                                                self.q_weight.T,
                                                out_dtype=torch.float16,
                                                scale_a=scaleRow,
                                                scale_b=self.q_scale_col
                                    )
            end_event.record()
            torch.cuda.synchronize()
            ms_int8 = start_event.elapsed_time(end_event)

           
            print("fp8 = %.4f, applied ai = %.4f, acc1 = %.4f, acc2 = %.4f, acci8 = %.4f"%(ms_fp8, ms_fp8_applied_ai,
                                                                                        self.ms_fp / ms_fp8 , 
                                                                                        self.ms_fp/( ms_fp8_applied_ai),
                                                                                         self.ms_fp/( ms_int8)))
            
        return o.reshape(out_shape)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
