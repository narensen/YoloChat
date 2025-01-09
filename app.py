import torch
import torch.nn as nn
from transformers import BlipProcessor, BlipForConditionalGeneration, BertTokenizer, BertForQuestionAnswering
from flask import Flask, request, jsonify, render_template
import os
from PIL import Image

# Load YOLO model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize Flask app
app = Flask(__name__)

# Updated Flask routes
@app.route('/')
def index():
    return render_template('index.html')  # Place index.html in root folder

# Add static file handling
app.static_folder = '.'
app.static_url_path = ''

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    # Save the uploaded image
    image = request.files['image']
    image_path = os.path.join('.', image.filename)
    image.save(image_path)

    # Perform object detection
    results = yolo_model(image_path)
    detected_objects = results.pandas().xyxy[0]['name'].tolist()

    # Generate image caption
    img = Image.open(image_path)
    inputs = processor(img, return_tensors="pt")
    outputs = blip_model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    # Remove the uploaded image
    os.remove(image_path)

    return jsonify({'detected_objects': detected_objects, 'caption': caption})

if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)