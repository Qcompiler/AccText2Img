Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)    Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ----------  -----------  ----------------------------------------------------------------------------------------------------
     16.2    2,002,729,481      6,000    333,788.2    333,791.0    329,951     337,055      1,032.2  void cutlass::Kernel<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)
     13.9    1,724,212,105     10,502    164,179.4    181,599.0     51,616     345,855     77,136.6  ampere_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_stages_32x1_tn                                     
     13.0    1,608,224,691     24,600     65,375.0     65,376.0     64,544      66,208        211.8  ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_32x3_tn                                         
      9.3    1,146,829,471      2,200    521,286.1    431,199.0    376,575   1,162,813    203,057.3  sm80_xmma_fprop_implicit_gemm_indexed_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize128x128x64_stage3_war…
      8.5    1,050,898,786      7,000    150,128.4     88,096.0     82,400     529,119    153,511.5  void pytorch_flash::flash_fwd_kernel<pytorch_flash::Flash_fwd_kernel_traits<(int)64, (int)128, (int…
      6.7      832,760,614     12,600     66,092.1     66,304.0     64,575      68,128        585.8  void cutlass::Kernel<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)


     25.4    2,372,981,113     59,156     40,114.0     21,472.0     13,472     192,896     36,065.7  void cutlass::Kernel<cutlass::gemm::kernel::symmetric::GemmDequant<cutlass::gemm::threadblock::MmaM…
     12.2    1,145,140,448      2,200    520,518.4    428,831.0    377,502   1,254,045    203,471.6  sm80_xmma_fprop_implicit_gemm_indexed_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize128x128x64_stage3_war…
     10.9    1,020,474,433      7,000    145,782.1     82,688.0     81,887     569,406    153,728.8  void pytorch_flash::flash_fwd_kernel<pytorch_flash::Flash_fwd_kernel_traits<(int)64, (int)128, (int…
      7.7      720,195,439      1,100    654,723.1    417,503.0    415,679   1,491,036    346,694.2  sm86_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize128x128x32_stage3_warpsize2x2…
      6.2      582,899,289     59,156      9,853.6      6,400.0      1,663      56,288      7,715.0  void FindRowScaleKernel_<(int)256, float>(signed char *, const __half *, T2 *, int, int)            
      4.3      405,250,277     62,080      6,527.9      3,232.0        768      69,920     10,134.2  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<c10::Half>, at::deta…
      3.9      365,604,472      7,000     52,229.2     37,311.0     34,432     144,832     37,160.1  void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::n…
      2.9      270,246,922     59,156      4,568.4      3,232.0        832      21,952      3,074.3  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<signed char>, at::de…