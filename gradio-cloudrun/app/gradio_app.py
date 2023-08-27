""" 
    Gradio app
    Text to Image generation with SDXL 1.0 deployed in Vertex AI Prediction
"""

from google.cloud import aiplatform
from PIL import Image

import base64
import gradio as gr

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/989788194604/locations/europe-west4/endpoints/6803241391003533312"  # <---- CHANGE THIS !!!
)

def get_image(prompt, prompt2, negative_prompt, negative_prompt2, inference_steps, guidance_scale, safety_filter):

    instance = [[prompt], [prompt2], [negative_prompt], [negative_prompt2], [inference_steps], [guidance_scale], [safety_filter]]
    response = endpoint.predict(instance)
    recovered_image_bytes = base64.b64decode(response.predictions[0])
    return Image.frombytes("RGB", (1024, 1024), recovered_image_bytes)

# def get_image_sdxl_api(key, prompt, inference_steps, filter):
#   payload = {
#     "key": key,
#     "prompt": prompt,
#     "negative_prompt": "((out of frame)), ((extra fingers)), mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), (((tiling))), ((naked)), ((tile)), ((fleshpile)), ((ugly)), (((abstract))), blurry, ((bad anatomy)), ((bad proportions)), ((extra limbs)), cloned face, (((skinny))), glitchy, ((extra breasts)), ((double torso)), ((extra arms)), ((extra hands)), ((mangled fingers)), ((missing breasts)), (missing lips), ((ugly face)), ((fat)), ((extra legs)), anime",
#     "width": "512",
#     "height": "512",
#     "samples": "1",
#     "num_inference_steps": inference_steps,"safety_checker": filter,"enhance_prompt": "yes","guidance_scale": 7.5}
#   headers = {}
#   response = requests.request("POST", url, headers=headers, data=payload)
#   url1 = str(json.loads(response.text)['output'][0])
#   r = requests.get(url1)
#   i = Image.open(BytesIO(r.content))
#   return i


title = """Text to Image Generation with Stable Diffusion in Vertex AI"""
description = """#### [Vertex AI Prediction](https://cloud.google.com/vertex-ai/docs/predictions/overview) and [Stable Diffusion XL 1.0](https://stability.ai/stablediffusion)"""
examples = [
    ["highly detailed portrait of an underwater city, with towering spires and domes rising up from the ocean floor. The city is bathed in a soft, golden light, and there is a sense of ancient mystery and wonder. The city is surrounded by a vibrant coral reef, with colorful fish swimming amongst the ruins. In the distance, there is a sunken shipwreck, aesthetic!!"],
    ["underwater city, with its mast and sails visible above the waterby atey ghailan, by greg rutkowski, by greg tocchini, by james gilleard, by joe fenton, by kaethe butcher, gradient yellow, grunge aesthetic!!!"],
    ["full body,Cyber goth Geisha in the rain in a  tokyo future  city city wide, Pretty Face, Beautiful eyes, Anime, Portrait, Dark Aesthetic, Neon sunset blade runner background, Concept Art, Digital Art, Anime Art, unreal engine, greg rutkowski, loish, rhads, beeple, makoto shinkai, haruyo morita and lois"],
    ["Cyber goth Geisha in the rain, stylized cyberpunk black tokyo market, indoor in the style of blade runner, stands illuminated by greens neon lights, crowded with cyborgs photorealistic background, 3 5 mm, grainy ruined film, dark color scheme, ray tracing, unreal engine, 4 k long shot"]
]

demo = gr.Interface(fn=get_image,
                    inputs = [gr.Textbox(label="Enter Prompt"),  gr.Textbox(label="Enter Prompt2"), gr.Textbox(label="Enter the Negative Prompt"), gr.Textbox(label="Enter the Negative Prompt2"), gr.Number(label="Enter number of inference steps", value=50), gr.Number(label="Guidance scale", value=9.0), gr.Checkbox(label="Safety filter")],
                    outputs = gr.Image(type='pil'), title = title, description = description, examples=examples)

demo.launch(server_name="0.0.0.0", server_port=7860)




