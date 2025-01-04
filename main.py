import torch
import torch.nn as nn
from transformers import BlipProcessor, BlipForConditionalGeneration, BertTokenizer, BertForQuestionAnswering
from flask import Flask, request, jsonify, render_template
import os
from PIL import Image

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Perform object detection on an image
img_path = "horse.jpg"
results = model(img_path)
detected_objects = results.pandas().xyxy[0]['name'].tolist()
print("Detected Objects:", detected_objects)

# Load BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Generate image caption
img = Image.open(img_path)
inputs = processor(img, return_tensors="pt")
outputs = model.generate(**inputs)

# Decode the output
caption = processor.decode(outputs[0], skip_special_tokens=True)
print("Generated Caption:", caption)

# Load BERT model for question answering
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Example query
query = "Is the horse middle of running?"
inputs = tokenizer.encode_plus(query, caption, return_tensors="pt")

start_scores, end_scores = model(**inputs, return_dict=False)
start_idx = torch.argmax(start_scores)
end_idx = torch.argmax(end_scores) + 1

answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx]))
print("Answer:", answer)

# Initialize Flask app
app = Flask(__name__)

# Load models for Flask endpoints
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    # Save the uploaded image
    image = request.files['image']
    image_path = os.path.join('uploads', image.filename)
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
