
# from sageattention import sageattn
import torch 
import os
from diffusers import DiffusionPipeline
from diffusers import StableDiffusion3Pipeline
from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler

torch.manual_seed(0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mix_linear", action = 'store_true' )
parser.add_argument("--fp8_linear", action = 'store_true' )
parser.add_argument("--q_attn", action = 'store_true')
parser.add_argument("--reproduce", action = 'store_true')
parser.add_argument("--fp16_algo", action = 'store_true')
parser.add_argument("--sm90", action = 'store_true')
parser.add_argument("--bench", action = 'store_true')
parser.add_argument("--bnb", action = 'store_true')
parser.add_argument("--model", type=str, default="sdxl")

args = parser.parse_args()

if args.mix_linear:
    from mylinear import MixLinear_GEMM
    torch.nn.Linear = MixLinear_GEMM

if args.sm90:
    import transformer_engine.pytorch as te
    torch.nn.Linear = te.Linear

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


from diffusers  import BitsAndBytesConfig  

from diffusers import Transformer2DModel, SD3Transformer2DModel

# os.system("rm -r ./inference_text*")

if args.model == "sdxl":
    model = "/home/dataset/stabilityai/stable-diffusion-xl-base-1.0/stable-diffusion-xl-base-1.0"

    pipe = DiffusionPipeline.from_pretrained(model, 
                                            torch_dtype=torch.float16, 
                                            use_safetensors=True, 
                                            variant="fp16")
    
if args.model == "sd3.5":
    model = "/home/dataset/stable-diffusion-3.5-medium"
    if args.bnb:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        transformer_8bit = SD3Transformer2DModel.from_pretrained(
            model,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        )

        pipe = StableDiffusion3Pipeline.from_pretrained(model, 
                                                torch_dtype=torch.float16,
                                                transformer = transformer_8bit, 
                                                use_safetensors=True, variant="fp16")
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(model, 
                                                torch_dtype=torch.float16, 
                                                use_safetensors=True, variant="fp16")

if args.model == "sd3":
    model = "/home/dataset/stable-diffusion-3-medium-diffusers"
    if args.bnb:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        transformer_8bit = SD3Transformer2DModel.from_pretrained(
            model,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        )

        pipe = StableDiffusion3Pipeline.from_pretrained(model,transformer = transformer_8bit,  
                                             torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(model, 
                                             torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    


pip = pipe.to("cuda")

pip = pipe.to(torch.float16)




# with torch.autocast("cuda", torch.float16):
prompt = "An astronaut riding a green horse"
num_inference_steps = 50
# guidance_scale = 7.0


torch.manual_seed(0)

images = pipe(prompt=prompt, num_inference_steps = num_inference_steps).images[0]
images.save("inference_text2img.png")
print("start estimating!!")


prompt = "An astronaut riding a red dog"
images = pipe(prompt=prompt, num_inference_steps =  num_inference_steps).images[0]
images.save("inference_text3img.png")


if args.reproduce:
    
    

    for num_inference_steps in [5, 8 , 10, 50]:
        torch.manual_seed(0)
        prompt = "An astronaut riding a green horse"

        images = pipe(prompt=prompt, num_inference_steps = num_inference_steps).images[0]
        prompt = "An astronaut riding a red dog"
        images = pipe(prompt=prompt, num_inference_steps = num_inference_steps).images[0]
        images.save(str(num_inference_steps) + "_inference_text2img.png")


if args.bench:

    for i in range(4):
        torch.manual_seed(0)
        prompts = ["girl,blue|red dress,grass,playing", 
                   "lake,many aquatic_plants and stones , fish", 
                   "A girl,in the square,playing drums,smiling",
                   "a girl,wear Hanfu" ]
        images = pipe(prompt=prompts[i], num_inference_steps = num_inference_steps).images[0]
        out = str(i) + ".png"
        if args.mix_linear:
            out = "q_" + out
        if args.fp8_linear:
            out = "fp8_" + out
        images.save(out)
        
