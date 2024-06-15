#using tf servings
from fastapi import FastAPI, File, UploadFile
import uvicorn
import requests
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle


app = FastAPI()

endpoint = "http://localhost:8501/v1/models/tomatoes_model:predict"

# MODEL = tf.keras.layers.TFSMLayer("../models/1")
CLASS_NAMES = ["Yellow curl virus", "Mosaic virus", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    json_data = {
        "instances": img_batch.tolist()
    }
    
    response = requests.post(endpoint, json=json_data)
    pass

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)