from fastapi import FastAPI, File, UploadFile
import uvicorn
from ultralytics import YOLO
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import gradio as gr
# Create the Gradio interface


# Launch the Gradio interface

app = FastAPI()

# Load the YOLO model
model = YOLO("/best.pt")  # Replace "best.pt" with the correct path to your trained model
def detect_objects(image):
    # Run YOLO model on input image
    results = model(image)
    result_img = results[0].plot()  # Render the detection results
    return Image.fromarray(result_img)

interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="FName",
    description="Upload a image ."
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load and preprocess the image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Perform inference using YOLO
    results = model(image)

    # Process the results
    predictions = []
    for result in results:
        for box in result.boxes:
            predictions.append({
                "box": box.xyxy.tolist(),  # Bounding box [x_min, y_min, x_max, y_max]
                "score": float(box.conf),  # Confidence score
            })

    return {"predictions": predictions}


if __name__ == "__main__":
    interface.launch()
    uvicorn.run(app, host="0.0.0.0", port=8000)
