
# from sageattention import sageattn
import torch 
import os
from diffusers import DiffusionPipeline
torch.manual_seed(0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mix_linear", action = 'store_true' )
parser.add_argument("--fp8_linear", action = 'store_true' )
parser.add_argument("--q_attn", action = 'store_true')
parser.add_argument("--reproduce", action = 'store_true')
parser.add_argument("--fp16_algo", action = 'store_true')
parser.add_argument("--model", type=str, default="Llama-2-7b")
args = parser.parse_args()

if args.mix_linear:
    from mylinear import MixLinear_GEMM
    torch.nn.Linear = MixLinear_GEMM

if args.fp8_linear:
    from mylinearfp8 import MixLinear_FP8GEMM
    torch.nn.Linear = MixLinear_FP8GEMM
if args.q_attn:

    from sageattention import sageattn
    import torch.nn.functional as F

    F.scaled_dot_product_attention = sageattn


if args.fp16_algo:
    from mylinearfp16alg import MixLinear_GEMM
    torch.nn.Linear = MixLinear_GEMM



os.system("rm -r ./inference_text*")
pipe = DiffusionPipeline.from_pretrained("/home/dataset/stabilityai/stable-diffusion-xl-base-1.0/stable-diffusion-xl-base-1.0", 
                                         torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")



prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt).images[0]
images.save("inference_text2img.png")
print("start estimating!!")


prompt = "An astronaut riding a red dog"
images = pipe(prompt=prompt).images[0]
images.save("inference_text3img.png")


if args.reproduce:
    
    
   
    for num_inference_steps in [5, 8 , 10, 50]:
        torch.manual_seed(0)
        prompt = "An astronaut riding a green horse"

        images = pipe(prompt=prompt, num_inference_steps = num_inference_steps).images[0]
        prompt = "An astronaut riding a red dog"
        images = pipe(prompt=prompt, num_inference_steps = num_inference_steps).images[0]
        images.save(str(num_inference_steps) + "_inference_text2img.png")
 
# import time
# start = time.time()
# for i in range(5):
#     prompt = "A girl playing with a red dog"
#     images = pipe(prompt=prompt).images[0]

# end = time.time()

# print(end - start)
    
