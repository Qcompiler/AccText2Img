 
import torch
import math
import torch.nn as nn
import sys
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
import mixgemm
from vllm import _custom_ops as ops


def FindOutliers(Activation, sigma = None):

    if sigma is None:
        sigma = 50
    
    tmp = torch.unique(torch.where((  Activation.abs() > sigma ))[1])
    return tmp.to(torch.int32)

class MixLinear_GEMM(nn.Module):
    def __init__(self, in_features, out_features, bias = True,  
                 device=None):
        super().__init__()
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

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        if  self.init is not True:
            # print("I am init the wegith do not disturb me and count me during time estimation")
            self.init = True
            if len(input.shape) == 3 :
                if input.shape[0] * input.shape[1] > 128:
                    # print("I should quant this layer")
                    
                    tmp = input.reshape(-1, input.shape[-1])
                    # print(tmp.shape)
                    # print("------------")
                    local_ind = FindOutliers(tmp)
                    # print(local_ind)
            
                    self.n_outliers = len(local_ind)
                    if self.n_outliers == 0: 
                        #print("without outliers hhh !")
                        self.weight.data = self.weight.data.cpu()
                        tmp = self.weight.data
                        self.q_scale_col =   (torch.max(torch.abs(tmp), dim=1)[0].unsqueeze(1) / (127)).to(torch.float16).reshape((1,self.out_features))
                        tmp  /= self.q_scale_col.T
                        tmp = torch.clamp(tmp, -128, 127)

                        self.q_weight = tmp.round().to(torch.int8).cuda()
                        self.q_scale_col = self.q_scale_col.cuda().reshape((self.out_features))


                        # 把 bias 打包到 scale 里面
                        if self.bias is not None:
                            tmp = self.bias
                        else:
                            tmp = torch.zeros((self.out_features), dtype= torch.float16, device= input.device)
                        # print(self.q_scale_col)
                        # print(self.bias)
                        tmp = torch.cat([self.q_scale_col, tmp]).reshape((2, self.out_features))
                        self.q_scale_col = tmp.t().contiguous().cuda()
                        # print(tmp)
                        # exit()
                        # self.q_weight = tmp.round().to(torch.int8).cuda().T
                        # self.q_scale_col = self.q_scale_col.cuda().to(torch.float32)
                        self.quanted = True
                        
                        # print(self.weight.shape)
                        # print(input.shape)
                        # print(input)
                        # grand =  F.linear(input, self.weight, self.bias)
                        # M = input.shape[0] * input.shape[1]
                        # K = self.in_features
                        # N = self.out_features
                        
                        
                        # y1 = mixlib.mixgemmforward_direct(M,N,K,
                        #             input,
                        #             self.q_weight, 
                        #             self.q_scale_col)
                        # if self.bias is not None:
                        #     y1 += self.bias
                        # grand = grand.reshape(y1.shape)

                        
                        # print(grand[0:3,0:3])
                        # print(y1[0:3,0:3])
                        # exit()
                    else:
                        # pass

                        self.weight_cache = self.weight.data[:, local_ind]
                        self.ind = local_ind
                        tmp = self.weight.cpu()
                        tmp[:,local_ind] = 0
                        self.q_scale_col =   (torch.max(torch.abs(tmp), dim=1)[0].unsqueeze(1) / (127)).to(torch.float16).reshape((1,self.out_features))
                        tmp  /= self.q_scale_col.T
                        tmp = torch.clamp(tmp, -128, 127)


                        # self.q_weight = tmp.round().to(torch.int8).cuda().T
                        self.q_scale_col = self.q_scale_col.cuda()
                        self.q_weight = tmp.round().to(torch.int8).cuda()
                        
                        self.quanted = True

                        

                        
        if   self.quanted is False or self.n_outliers:
            return F.linear(input, self.weight, self.bias)
        else:
            assert ( len(input.shape) == 3) 
            input_shape0 = input.shape[0]
            input_shape1 = input.shape[1]
            M = input.shape[0] * input.shape[1]
            K = self.in_features
            N = self.out_features
            # to optimize the in continues memory!
            # tmp = torch.zeros(input.shape)
            if not input.is_contiguous():
            # if not input.is_contiguous():
            #     print(input.stride(0))
            #     print(input.stride(1))
            #     print(input.stride(2))
            #     print(tmp.stride(0))
            #     print(tmp.stride(1))
            #     print(tmp.stride(2))
            #     print("input shape is ")
            #     print(input.shape)
                
            #     exit()
                input = input.reshape(M, K)
             

            if self.n_outliers == 0:

                
                 
                
                y1 = mixgemm.mixgemmforward_direct(M, N, K,
                                    input,
                                    self.q_weight, 
                                    self.q_scale_col, 
                                    input_shape0, input_shape1)
                
                debug = 0
                use_ops = 0
                
                if debug:
                    import mixlib

                    if use_ops:
                        scaleRow = torch.zeros((M, 1) , dtype= torch.float32, device= input.device)
                        q_xcache = mixlib.FindRowScaleF32(input, scaleRow, M, K, 8)
                        self.q_scale_col = self.q_scale_col.cuda().to(torch.float32)

                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_event.record()
                 

                    for i in range(10):
                        y1 = mixlib.mixgemmforward_direct(M, N, K,
                                                input,
                                                self.q_weight, 
                                                self.q_scale_col)
                        if use_ops:
                            y1 =   ops.cutlass_scaled_mm(
                                        q_xcache,
                                        self.q_weight.T,
                                        out_dtype=torch.float16,
                                        scale_a=scaleRow,
                                        scale_b=self.q_scale_col,
                                        bias = self.bias
                            )
                 
                    end_event.record()
                    torch.cuda.synchronize()
                    ms4 = start_event.elapsed_time(end_event)

                if debug:
           
                    start_event.record()
                    for i in range(10):
                        y1 = F.linear(input, self.weight, self.bias)
                 
                    end_event.record()
                    torch.cuda.synchronize()
                    ms5 = start_event.elapsed_time(end_event)  
                    print("int8 time = %.8f  fp16 time = %.8f, %d %d %d"%(ms4, ms5, M, N, K))               
            else:
                assert (False)
 
                q_xcache, scaleRow, outliers = mixlib.FindRowScaleFusedExtracOutliersF32(input, 
                            self.ind, len(self.ind), M , K )
          
                y1 =   ops.cutlass_scaled_mm(
                            q_xcache,
                            self.q_weight.T,
                            out_dtype=torch.float16,
                            scale_a=scaleRow,
                            scale_b=self.q_scale_col,
                            bias = self.bias
                ) + torch.mm(outliers, self.weight_cache.T)

            # 把 bias 打包到 scale 里面
            # if self.bias is not None:
            #     y1 += self.bias
            # optimize 1: opt the output shape
            return y1

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
