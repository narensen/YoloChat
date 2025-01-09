from flask import Flask, request, jsonify, send_from_directory
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
from PIL import Image
import tempfile

app = Flask(__name__)

# Initialize models in global scope
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/')
def home():
    return send_from_directory('../public', 'index.html')

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image = request.files['image']
    
    # Use temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        image.save(tmp.name)
        
        # Object detection
        results = yolo_model(tmp.name)
        detected_objects = results.pandas().xyxy[0]['name'].tolist()
        
        # Caption generation
        img = Image.open(tmp.name)
        inputs = processor(img, return_tensors="pt")
        outputs = blip_model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        os.unlink(tmp.name)
        
    return jsonify({
        'detected_objects': detected_objects,
        'caption': caption
    })

app = app.wsgi_app