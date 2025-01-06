
# from sageattention import sageattn
import torch 
import os
from diffusers import DiffusionPipeline
torch.manual_seed(0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mix_linear", action = 'store_true' )
parser.add_argument("--q_attn", action = 'store_true')

parser.add_argument("--model", type=str, default="Llama-2-7b")
args = parser.parse_args()

if args.mix_linear:
    from mylinear import MixLinear_GEMM
    torch.nn.Linear = MixLinear_GEMM

if args.q_attn:

    from sageattention import sageattn
    import torch.nn.functional as F

    F.scaled_dot_product_attention = sageattn





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


# import time
# start = time.time()
# for i in range(5):
#     prompt = "A girl playing with a red dog"
#     images = pipe(prompt=prompt).images[0]

# end = time.time()

# print(end - start)
    
