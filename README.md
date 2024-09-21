# Image Captioning and Query Response System

This repository contains an end-to-end image captioning and query response system using **YOLOv5** for object detection, **BLIP** for image captioning, and **BERT** for query understanding and response generation. The system can detect objects in an image, generate descriptive captions, and respond to natural language queries about the image.

## Introduction

This project is designed to provide an interactive image captioning and query response system:
- **YOLOv5** detects objects in the image.
- **BLIP** generates a natural language caption describing the scene.
- **BERT** handles user queries related to the caption and generates responses based on the caption.

### Example Workflow:
1. **Input**: An image.
2. **Object Detection**: YOLOv5 detects objects like `["person", "car", "dog"]`.
3. **Caption Generation**: BLIP generates a caption such as: *"A person is standing next to a car while a dog is sitting in the back seat."*
4. **Query Handling**: BERT processes user queries like "What is the dog doing?" and responds with: *"The dog is sitting in the back seat."*

## Features
- Object Detection using YOLOv5
- Image Captioning using BLIP
- Query Interpretation and Response using BERT
- Fully open-source with no external API dependencies

## Installation

### Prerequisites
1. Python 3.x
2. Install required libraries:
   - PyTorch
   - Hugging Face Transformers
   - Pillow for image processing

### Clone the repository:
```bash
git clone https://github.com/narensen/yolochat
cd yolochat
