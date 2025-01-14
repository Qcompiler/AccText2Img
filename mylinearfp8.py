 
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


        self.q_scale_col = None
        self.q_weight = None
        self.scale_input = None
        

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
        x = input.reshape(-1, input.shape[-1]) 

        if   self.init is False:
            self.scale_weight = torch.ones((1), device=input.device, dtype=torch.float32)
            self.scale_input = torch.ones((1), device=input.device, dtype=torch.float32)
            
            # self.w = self.weight.t().to(weight_dtype)
            self.init = True


            # tmp = self.weight.cpu()
            # self.q_scale_col =   (torch.max(torch.abs(tmp), dim=1)[0].unsqueeze(1) / (127)).to(torch.float16).reshape((1,self.out_features))
            # tmp  /= self.q_scale_col.T
            # tmp = torch.clamp(tmp, -128, 127)

            # self.q_weight = tmp.round().to(torch.int8).cuda()
            # self.q_scale_col = self.q_scale_col.cuda().to(torch.float32).reshape((self.out_features))

            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # torch.cuda.synchronize()
            # start_event.record()
            # for i in range(10):
            #     o  = F.linear(input, self.weight, self.bias)
            # end_event.record()
            # torch.cuda.synchronize()
            # self.ms_fp = start_event.elapsed_time(end_event)
            # self.weight.data = self.weight.cpu()
            # del self.weight

            self.output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            torch.ops._C.dynamic_scaled_fp8_quant(self.output, x, self.scale_input)

            self.w = torch.empty_like(self.weight, dtype=torch.float8_e4m3fn)
            torch.ops._C.static_scaled_fp8_quant(self.w, self.weight, self.scale_weight)
            
        out_shape =  input.shape[:-1] + (self.out_features, )
        
        # grand =  F.linear(input, self.weight, self.bias)

        torch.ops._C.static_scaled_fp8_quant(self.output, x, self.scale_input)
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
