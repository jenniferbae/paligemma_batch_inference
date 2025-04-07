import requests
import io
import fsspec
import os
import numpy as np
import ray
from PIL import Image
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForVision2Seq # pyright: ignore[reportPrivateImportUsage]
import torch

# Get the token from environment variable
HF_TOKEN = os.environ["HF_TOKEN"]  # This will be set by job.yaml

# Log in to Hugging Face hub
login(token=HF_TOKEN)

# Use fsspec to list all JPEG files in the bucket
fs = fsspec.filesystem("gcs")
matches = fs.glob("coco-batch-inference/val2017/*.jpg")

# Convert GCS paths to HTTP URLs
base_url = "https://storage.googleapis.com/coco-batch-inference/val2017/"
urls = [base_url + path.split("/")[-1] for path in matches]

# Create a Ray dataset with 100 image URLs
ds = ray.data.from_items([{"url": url} for url in urls[:100]])

# Define the fetch and decode function
def fetch_and_decode(row):
    response = requests.get(row["url"])
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    return {
        "image": np.array(img),  # Arrow-compatible
        "prompt": "<image> caption en:"
    }

# Map the dataset with fetch_and_decode
decoded_ds = ds.map(fetch_and_decode)

# Define the model class
@ray.remote  # Removed runtime_env to avoid conflicts with image_uri
class PaliGemmaModel:
    def __init__(self, model_id="google/paligemma-3b-mix-224"):
        # Login in the worker using the environment variable
        login(token=os.environ["HF_TOKEN"])
        
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def preprocess(self, img_array):
        img_pil = Image.fromarray(img_array)
        text = "<image> caption en:"
        inputs = self.processor(images=img_pil, text=text, return_tensors="pt", padding=True, truncation=True)
        return inputs

    def generate_caption(self, img_array):
        inputs = self.preprocess(img_array)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)
            generated_caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return generated_caption

# Initialize the model as a Ray actor
pali_gemma_model = PaliGemmaModel.remote()

# Perform batch inference
results = decoded_ds.map_batches(
    lambda batch: ray.get([
        pali_gemma_model.generate_caption.remote(img["image"])
        for img in batch
    ]), 
    batch_size=10  # Adjust batch size as needed
)

# Collect the results
prediction_batch = results.take_batch(5)