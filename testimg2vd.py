from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import torch

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

image = load_image(
"https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
image = image.resize((1024, 576))

frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
export_to_video(frames, "generated.mp4", fps=7)