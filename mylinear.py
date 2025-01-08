 
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

        """
        To do
        (1) I should write a kernel to accelerated the code : if  (input[[1],0:32,0:8] - self.last_input).sum() == 0:
            2025.1.7: DONE
            not required now ---------- I use my locality algorithm
        (2) I should do some further analyse of max value pattern 
            2025.1.8 : DONE
            no need ! failed! No locality.....
        (3) INT8 GEMM kernel:
            todo 1: autotune by triton :   github.com/AlibabaPAI/FLASHNN/blob/main/flashnn/triton_kernels/gemm_a8w8.py
            todo 2: tune by cutlass param : add gemm shape in param see in git log
        (4) FP8 GEMM kernel:
            todo :  test in H100
            2025.1.8 : DONE
            I implemented in mylinearfp8.py
            and change debug = True in the file to get the profile result
        (5) optimize:   
            input = input.contiguous()
            this only affect the computing scaling factor and quantization steps
            so we shoud add support for these two kernels with in_contiguous input

        """
        if  self.init is not True:
            if self.layer_id == 1:
                print("I am init the weight do not disturb me and count me during time estimation")

            if len(input.shape) == 3:
                M = input.shape[0] * input.shape[1]
                self.input_scales = torch.zeros((M, 1), dtype = torch.float32, device = input.device)
            self.init = True
            computed_bound = False
            if len(input.shape) == 3 and input.shape[0] * input.shape[1] > 64:
                 computed_bound = True
            if len(input.shape) == 2  and  input.shape[0]  > 64:
                computed_bound = True
            if not computed_bound:
                if memory_bound_eetq_linear:
                    int8_weight_cpu = torch.t(self.weight.data).contiguous().cpu()
                    int8_weight, scales = quant_weights(int8_weight_cpu, torch.int8, False)
                    self.eetq_weight = (int8_weight).cuda()
                    self.eetq_scale_col = (scales.half()).cuda()

                    self.weight.data = self.weight.data.cpu()
                    del self.weight
            if computed_bound :
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
                self.weight.data = self.weight.data.cpu()

                # 删除原来的权重
                del self.weight
                del tmp
                
                                    
            


        # print("my layer id is %d\t"%(self.layer_id))  
        #   
        # if self.cnt > 4:
        #     exit()

        if   self.quanted is False :
            self.cnt += 1
            # if self.cnt == 0:
            #     print(input.shape)
            #     self.cnt += 1
            # if not input.is_contiguous():
            #     input = input.contiguous()

            # return y
            # inputs = input.reshape(-1, input.shape[-1])
            # shape = input.shape[:-1] + (self.out_features, )
            if memory_bound_eetq_linear:
                y =  w8_a16_gemm(input, self.eetq_weight, self.eetq_scale_col)

                if self.bias is not None:
                    y += self.bias
                return y
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
            
            if self.cnt >= 50:
                self.cnt = 0
                # self.reuse_output_because_of_zeros_input = False
                # release cache
                self.cache_computed = False


            # if self.reuse_output_because_of_zeros_input:
                
            #     return self.y1
            # if self.reuse_output_because_of_zeros_input and self.y1 is not None:
            #     return self.y1       
            if self.cnt == 0 and input.shape[0] == 2:
                if input[0,0,0] == 0:
                    self.last_input = input[[1],0:32,0:8]

            if self.cnt == 1 and input.shape[0] == 2:
                

                if self.last_input is not None:
                    if  (input[[1],0:32,0:8] - self.last_input).sum() == 0:
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
            during this analysis we found that the location of the max index keeps vary stable
            so we should not compute the scaling factors at each steps?
            maybe we could try it
            2025.1.7
            sorry
            not work
            """
            if  self.reuse_output_because_of_zeros_input is True and self.cache_computed:
                return self.y1  
            if self.n_outliers == 0:

                if self.reuse_scaling_factor:

                    if self.cnt == 0:
                        y1 = mixgemm.mixgemmforward_direct(M, N, K,
                                        input,
                                        self.input_scales,
                                        self.q_weight, 
                                        self.q_scale_col, 
                                        input_shape0, input_shape1)
                    else:
                        y1 = mixgemm.mixgemmforward_direct_with_scaling(M, N, K,
                                            input,
                                            self.input_scales,
                                            self.q_weight, 
                                            self.q_scale_col, 
                                            input_shape0, 
                                            input_shape1 )
                 
                else:
                    # 记录最大和最小scale
                    # if self.doing_estimation:
                    #     current_scale = torch.amax(input)

                    #     if self.scale_max is None:
                    #         self.scale_max = current_scale
                    #         self.scale_min = current_scale
                    #     self.scale_max =  max (self.scale_max, current_scale)
                    #     self.scale_min =  min (self.scale_min, current_scale)
                    #     if self.cnt == 49:
                    #         self.doing_estimation = False
                    #         if self.scale_max / self.scale_min < 1.5:
                    #             self.reuse_scaling_factor = True
                                




                    y1 = mixgemm.mixgemmforward_direct(M, N, K,
                                        input,
                                        self.input_scales,
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
