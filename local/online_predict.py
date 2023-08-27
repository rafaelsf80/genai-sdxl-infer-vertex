""" 
    Simple test app to make inference on SDXL 1.0 stored in Vertex AI Prediction
"""

from google.cloud import aiplatform
import base64
from PIL import Image


endpoint = aiplatform.Endpoint(
    endpoint_name="projects/989788194604/locations/europe-west4/endpoints/6803241391003533312"  # <---- CHANGE THIS !!!
)

prompt = 'highly detailed portrait of an underwater city, with towering spires and domes rising up from the ocean floor. The city is bathed in a soft, golden light, and there is a sense of ancient mystery and wonder. The city is surrounded by a vibrant coral reef, with colorful fish swimming amongst the ruins. In the distance, there is a sunken shipwreck, aesthetic!!'
prompt2 = "underwater city, with its mast and sails visible above the waterby atey ghailan, by greg rutkowski, by greg tocchini, by james gilleard, by joe fenton, by kaethe butcher, gradient yellow, grunge aesthetic!!!"
negative_prompt = ''
negative_prompt2 = ''

test_instance=[[prompt], [prompt2], [negative_prompt], [negative_prompt2], [50], [9], [True]]

response = endpoint.predict(test_instance)

content = base64.b64decode(response.predictions[0])
image = Image.frombytes("RGB", (1024, 1024), content)
image.save("image.png","PNG")

print("DONE. Image saved as image.png")


