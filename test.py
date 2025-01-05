
# from sageattention import sageattn
import torch 
from mylinear import MixLinear_GEMM

torch.manual_seed(0)

torch.nn.Linear = MixLinear_GEMM
import os
os.system("rm -r ./inference_text*")

from diffusers import DiffusionPipeline,StableDiffusionXLImg2ImgPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("/home/dataset/stabilityai/stable-diffusion-xl-base-1.0/stable-diffusion-xl-base-1.0", 
                                         torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")


# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt).images[0]
images.save("inference_text2img.png")
print("start estimating!!")

prompt = "An astronaut riding a red dog"
images = pipe(prompt=prompt).images[0]
images.save("inference_text3img.png")
