 
import torch
import math
import torch.nn as nn
import sys
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
import mixgemm



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
                    # self.n_outliers = 0
                    if self.n_outliers == 0: 
                        #print("without outliers hhh !")
                        self.weight.data = self.weight.data.cpu()
                        tmp = self.weight.data
                        

                        # # 把 bias 打包到 scale 里面
                        # if self.bias is not None:
                        #     tmp = self.bias
                        # else:
                        #     tmp = torch.zeros((self.out_features), dtype= torch.float16, device= input.device)
                        # # print(self.q_scale_col)
                        # # print(self.bias)
                        # tmp = torch.cat([self.q_scale_col, tmp]).reshape((2, self.out_features))
                        # self.q_scale_col = tmp.t().contiguous().cuda()
                        # # print(tmp)
                        # # exit()
                        # # self.q_weight = tmp.round().to(torch.int8).cuda().T
                        # # self.q_scale_col = self.q_scale_col.cuda().to(torch.float32)
                        # self.quanted = True
                        
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

                    # self.weight.data = self.weight.data.cpu()
                    # del self.weight
                    self.q_scale_col =   (torch.max(torch.abs(tmp), dim=1)[0].unsqueeze(1) / (127)).to(torch.float16).reshape((1,self.out_features))
                    tmp  /= self.q_scale_col.T
                    tmp = torch.clamp(tmp, -128, 127)

                    self.q_weight = tmp.round().to(torch.int8).cuda()
                    self.q_scale_col = self.q_scale_col.cuda().reshape((self.out_features))



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
                    self.quanted = True

                        
            


        # print("my layer id is %d\t"%(self.layer_id))  
        #   
        # if self.cnt > 4:
        #     exit()
        if   self.quanted is False :
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
                input = input.contiguous()
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
            #     input = input.reshape(M, K)
             
            # input = input.reshape(M, K)
            # self.scale_history = self.scale_history.to(input.device)
            # # print(input.shape)
            """
            further analysis of max value
            """
            if self.cnt >= 50:
                self.cnt = 0
                self.reuse_output_because_of_zeros_input = False
                # release cache
                self.cache_computed = False


            # if self.reuse_output_because_of_zeros_input:
                
            #     return self.y1
            # if self.reuse_output_because_of_zeros_input and self.y1 is not None:
            #     return self.y1       
            if self.cnt == 0 and input.shape[0] == 2:
                self.last_input = input[[0, 1],0:32,0:8]

            if self.cnt == 1 and input.shape[0] == 2:
                # to do  写一个kernel 加速

                
                if  (input[[0, 1],0:32,0:8] - self.last_input).sum() == 0:
                    self.reuse_output_because_of_zeros_input = True
                    


                                                  
                    

                # if input.shape[0] == 1:
                #     if self.last_input is not None:
                #         if  (input[[0],0:32,0:8] - self.last_input).sum() == 0:
                #             self.reuse_output_because_of_zeros_input = True
                #             print("!!locallity!!!!!")

                #     else:
                #         self.last_input = input[[0],0:32,0:8]
                # if input[0:20].sum() == 0:
                #     if self.last_input is not None:
                #         if  (input - self.last_input).abs().sum() == 0:
                #             self.reuse_output_because_of_zeros_input = True
                #     else:
                #         self.last_input = input
                #print("layer id is %d"%(self.layer_id))
                #print(input)
                #exit()
            # if self.layer_id == 30 or self.layer_id == 300:
            #     print(self.cnt)

            # if self.cnt == 1:
            #     exit()
            self.cnt += 1
            # if self.cnt == 50:
            #     print(self.scale_history[0:self.cnt])
            """"
            #     #exit()
            during this analysis we found that the location of the max index keeps vary stable
            so we should not compute the scaling factors at each steps?
            maybe we could try it
            """
            if  self.reuse_output_because_of_zeros_input is True and self.cache_computed:
                return self.y1  
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
                            from vllm import _custom_ops as ops
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

                y1 = mixgemm.mixgemmforward_dynamic(M, N, K,
                                    input,
                                    self.q_weight, 
                                    self.q_scale_col, 
                                    input_shape0, input_shape1,
                                    self.weight_cache, 
                                    self.ind, self.n_outliers)
                
                # assert (False)
 
                # q_xcache, scaleRow, outliers = mixlib.FindRowScaleFusedExtracOutliersF32(input, 
                #             self.ind, len(self.ind), M , K )
          
                # y1 =   ops.cutlass_scaled_mm(
                #             q_xcache,
                #             self.q_weight.T,
                #             out_dtype=torch.float16,
                #             scale_a=scaleRow,
                #             scale_b=self.q_scale_col,
                #             bias = self.bias
                # ) + torch.mm(outliers, self.weight_cache.T)

            # optimize 把 bias 打包到 scale 里面
            # if self.bias is not None:
            #     y1 += self.bias
            # optimize 1: opt the output shape
            # if self.layer_id == 800:
            #     print(tmp[0])
            #     print(y1[0, 0:2, 0:10])
            # if self.reuse_output_because_of_zeros_input and self.y1 is not None:

          
            if self.reuse_output_because_of_zeros_input:
                self.cache_computed = True
                self.y1 = y1
            return y1

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
