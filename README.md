# Image Classifier Project

## Overview
**Image Classifier** is a simple web application that allows users to upload an image and have a pretrained AI model classify its contents.  
The application uses a convolutional neural network to analyze images and return predicted labels for objects or scenes within the image.

This project demonstrates how deep learning models can be integrated into an interactive web interface using Python and Streamlit.

**Author:** Kayjay

---

## Features
- Upload images directly through a web interface
- AI-powered image classification
- Pretrained **MobileNetV2** convolutional neural network
- Fast predictions using TensorFlow
- Simple and interactive UI built with Streamlit
- Image preprocessing using OpenCV and PIL

---

## Technology Stack
- **Python** – Core programming language  
- **Streamlit** – Web interface for interacting with the model  
- **TensorFlow / Keras** – Machine learning framework used for inference  
- **NumPy** – Numerical operations and array processing  
- **OpenCV** – Image preprocessing and manipulation  
- **PIL (Python Imaging Library)** – Image loading and handling  
- **UV** – Python dependency and environment management

---

## Machine Learning Model
This project uses **MobileNetV2**, a lightweight convolutional neural network (CNN) designed for efficient image classification.

MobileNetV2 was trained on the **ImageNet dataset**, which contains millions of labeled images across thousands of object categories. The model can recognize a wide variety of objects, animals, and scenes.

The model is accessed through the **TensorFlow Keras API** and is used here for inference on uploaded images.

---

## Known Limitations
- The model performs best on objects and landscapes commonly found in the ImageNet dataset.
- Accuracy may be lower for images that are heavily focused on people or unusual subjects.
- Predictions depend on the quality and clarity of the uploaded image.

---

## Running the Application

### Try it Online
You can test the application using the hosted Streamlit app:

https://test.streamlit.app/

### Run Locally

1. Clone the repository

```bash
git clone https://github.com/yourusername/image-classifier-project.git
cd image-classifier-project