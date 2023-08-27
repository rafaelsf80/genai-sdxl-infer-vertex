""" 
    FastAPI app with the Uvicorn server that servers SDXL 1.0
"""
 
from fastapi import FastAPI, Request
from fastapi.logger import logger

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, DiffusionPipeline, KDPM2AncestralDiscreteScheduler, StableDiffusionXLPipeline, AutoencoderKL
import gc

import logging
import base64
import os

app = FastAPI()

gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers

if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.INFO)

logger.info(f"Is CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

model_base = "stabilityai/stable-diffusion-xl-base-1.0"
v_autoencoder = "madebyollin/sdxl-vae-fp16-fix" # fix vae for run in fp16 precision without generating NaNs

logger.info(f"Loading model {model_base} and autoencoder {v_autoencoder}. This takes some time ...")

vae = AutoencoderKL.from_pretrained(v_autoencoder, torch_dtype=torch.float16)

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_base,
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    add_watermarker=False, # no watermarker
    )

pipe.to("cuda")

model_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"

pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    model_refiner,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    add_watermarker=False, # no watermarker
    )

#pipe_refiner.to("cuda")
pipe_refiner.enable_model_cpu_offload()

#(Optional) Change the scheduler
pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(
pipe.scheduler.config, use_karras_sigmas=True
)
#generator = torch.Generator().manual_seed(42)

logger.info(f"Loading model DONE.")


@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {"status": "healthy"}

@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
  
    body = await request.json()  # {'instances': [['who are you ?']]}
    logger.info(f"Body: {body}")

    instances = body["instances"]  # [['who are you ?']]
    logger.info(f"Instances: {instances}")

    prompt = instances[0][0]  # max 77 tokens in prompt
    prompt2 = instances[1][0]
    negative_prompt = instances[2][0]
    negative_prompt2 = instances[3][0]
    inference_steps = int(instances[4][0])
    guidance_scale = int(instances[5][0])
    filter = instances[6][0]
    if filter:
        negative_prompt = "((out of frame)), ((extra fingers)), mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), (((tiling))), ((naked)), ((tile)), ((fleshpile)), ((ugly)), (((abstract))), blurry, ((bad anatomy)), ((bad proportions)), ((extra limbs)), cloned face, (((skinny))), glitchy, ((extra breasts)), ((double torso)), ((extra arms)), ((extra hands)), ((mangled fingers)), ((missing breasts)), (missing lips), ((ugly face)), ((fat)), ((extra legs)), anime"

    logger.info(f"Creating BASE image guidance_scale:{guidance_scale} and inference_steps:{inference_steps}.")
    image_base = pipe(
        prompt=prompt,
        prompt_2=prompt2,
        width=1024, # default 1024
        height=1024, # default 1024
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt2,
        guidance_scale=guidance_scale, # ex:9.0,
        num_inference_steps=inference_steps # ex:50,
        ).images[0]

    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"Creating REFINER image...")
    image_refiner = pipe_refiner(
        prompt=prompt,
        prompt_2=prompt2,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt2,
        image=image_base,
        num_inference_steps=inference_steps,#50,
        strength=0.3,
        ).images[0]

    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"Encoding image to base64....")
    encoded_image = base64.b64encode(image_refiner.tobytes()) # bytes
  
    return {"predictions": [encoded_image.decode('utf-8')]}