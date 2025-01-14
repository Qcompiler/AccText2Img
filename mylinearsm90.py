 
import torch
import math
import torch.nn as nn
import sys
import gc
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
import mixgemm
from vllm import _custom_ops as ops
import mixlib
import mixgemm_v2
from gemm_a8w8 import triton_gemm_a8w8_forward
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
            this only affects the computing scaling factor and quantization steps
            so we should add support for these two kernels with in_contiguous input
        (6) fuse bias and scaling kernel together:
            2025.1.4 done!
            see in file /home/chenyidong/seperated_kernel/kernel/symmetric/epilogue/thread/linear_combination_dequant.h
        (7) load sparse outliers from global to RF instread of global to shared 
            developing 
            in file /home/chenyidong/seperated_kernel/kernel/mixgemm.cu __global___ void mma_sparse_A_dense_B_kernel
        """
        if  self.init is not True:
            if self.layer_id == 1:
                print("I am init the weight do not disturb me and count me during time estimation")

            if len(input.shape) == 3:
                M = input.shape[0] * input.shape[1]
                self.input_scales = torch.zeros((M, 1), dtype = torch.float32, device = input.device)
            self.init = True
            computed_bound = False

            self.find_zeros = torch.zeros((1,), dtype = torch.int32, pin_memory = True)
            self.reuse_output = torch.zeros((1,), dtype = torch.int32, pin_memory = True)
            self.last_input = torch.zeros((1, 32, 8 ), dtype = torch.float16, device = input.device)

            if len(input.shape) == 3 and input.shape[0] * input.shape[1] > 64:
                 computed_bound = True
            if len(input.shape) == 2  :
                computed_bound = False


            if computed_bound :
                # print("I should quant this layer")
                tmp = input.reshape(-1, input.shape[-1])
                local_ind = FindOutliers(tmp)
                # print(local_ind)
        
                self.n_outliers = len(local_ind)
                
                if self.n_outliers == 0: 
                    #print("without outliers hhh !")
     
                    tmp = self.weight.data.cpu()
                    
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


                self.q_scale_col = self.q_scale_col.to(torch.float32)
                self.quanted = True
                # self.weight.data = self.weight.data.cpu()

                # 删除原来的权重
                # del self.weight
                # del tmp
                
                                    
            


        # print("my layer id is %d\t"%(self.layer_id))  
        #   
        # if self.cnt > 4:
        #     exit()

        if   self.quanted is False :
            self.cnt += 1


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

        
            # input = input.reshape(M, K)
            # scaleRow = torch.zeros((M, 1) , dtype= torch.float32, device= input.device)   
            y1 =   mixgemm_v2.cutlass_scaled_mm( input_shape0, input_shape1, N, K,
                                         input,
                                         self.q_weight.T,
                                         self.input_scales,
                                         self.q_scale_col,
                                         self.bias)
            # scaleRow = torch.zeros((M, 1) , dtype= torch.float32, device= input.device)
            # q_xcache = mixlib.FindRowScaleF32(input, scaleRow, M, K, 8)    
            # y1 =   ops.cutlass_scaled_mm(
            #                     q_xcache,
            #                     self.q_weight.T,
            #                     out_dtype=torch.float16,
            #                     scale_a=scaleRow,
            #                     scale_b=self.q_scale_col
            #         )

            debug = True
            if debug:
                      


    
                scaleRow = torch.zeros((M, 1) , dtype= torch.float32, device= input.device)
                q_xcache = mixlib.FindRowScaleF32(input, scaleRow, M, K, 8) 
                y1 =   ops.cutlass_scaled_mm(
                                                q_xcache,
                                                self.q_weight.T,
                                                out_dtype=torch.float16,
                                                scale_a=scaleRow,
                                                scale_b=self.q_scale_col
                                    ).reshape(input_shape0, input_shape1, N)
                out = torch.empty([M, N], dtype=torch.float16, device=input.device)
                input_scales = self.input_scales.to(torch.float16) 
                q_scale_col = self.q_scale_col.to(torch.float16) 
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
                # scaleRow = torch.zeros((M, 1) , dtype= torch.float32, device= input.device)
                # q_xcache = mixlib.FindRowScaleF32(input, scaleRow, M, K, 8)    
                for i in range(10):
                    # y1 =   mixgemm_v2.cutlass_scaled_mm( input_shape0, input_shape1, 
                    #                                 N, K,
                    #                         input,
                    #                         self.q_weight.T,
                    #                         self.input_scales,
                    #                         self.q_scale_col,
                    #                         self.bias)
                    
                    triton_gemm_a8w8_forward(out, q_xcache, self.q_weight, input_scales, q_scale_col)
          
                end_event.record()
                torch.cuda.synchronize()
                ms_int8 = start_event.elapsed_time(end_event)
                
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
                for i in range(10):
                    F.linear(input, self.weight, self.bias)
                end_event.record()
                torch.cuda.synchronize()
                ms_fp16 = start_event.elapsed_time(end_event)
                print("ms int8 = %.4f   fp16 = %.4f"%(ms_int8, ms_fp16))
          

            return y1

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
