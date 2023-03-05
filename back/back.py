from auth.auth import auth
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import autocast
from diffusers import StableDiffusionPanoramaPipeline
from io import BytesIO
import base64 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

dev="cpu"
model="CompVis/stable-diffusion-v1-4"
pipe= StableDiffusionPanoramaPipeline.from_pretrained(model,revision=None,torch_dtype=torch.float64, use_auth_token=auth)
pipe.to(dev)

@app.get("/")
def gen(search : str):
    with autocast(dev):
        image = pipe(search , guidance_scale=8.5).images[0]
        
    image.save('output.png')
    extra = BytesIO()
    image.save(extra , format='PNG')
    img=base64.b64encode(extra.getvalue())
    return Response(content=img , media_type="image/png")    