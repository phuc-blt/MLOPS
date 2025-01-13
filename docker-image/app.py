import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Load the YOLO model
model = YOLO("best.pt")  # Replace "best.pt" with the correct path to your trained model

def detect_objects(image):
    # Run YOLO model on input image
    results = model(image)
    result_img = results[0].plot()  # Render the detection results
    return Image.fromarray(result_img)

# Create Gradio interface
interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Object Detection",
    description="Upload an image to detect objects."
)

# Launch the Gradio interface
interface.launch(server_name="0.0.0.0", server_port=8000)
