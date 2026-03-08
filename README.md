# Image Classifier Project

## Overview

**Image Classifier** is a simple web application that allows users to upload an image and have a pretrained AI model classify its contents.  
The application uses a convolutional neural network to analyze images and return predicted labels for objects within the image.

This project demonstrates how deep learning models can be integrated into an interactive web interface using Python and Streamlit.

**Author:** Kirujan Jegatheeswaran

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

## Image Processing Pipeline

Uploaded images must go through several preprocessing steps before they can be analyzed by the model.

Because **MobileNetV2** requires a fixed input size, each uploaded image is resized to **224×224 pixels**. The image is then converted into a NumPy array, expanded to include a batch dimension, and normalized using TensorFlow’s `preprocess_input()` function. The processed image is then passed to the MobileNetV2 model, which outputs classification probabilities for ImageNet categories.

Updated version: Test-Time Augmentation was included to improve prediction accuracy by applying eight different image transformations before the `preprocess_input` step in the pipeline. This approach enhances robustness while avoiding any modifications to the machine learning architecture.

---

### Simplified Flow

Image Upload
      ↓
Resize to 224×224
      ↓
Convert to NumPy Array
      ↓
Add Batch Dimension (np.expand_dims)
      ↓
Normalize Pixels (preprocess_input)
      ↓
MobileNetV2 Prediction
      ↓
Decode Predictions
      ↓
Human-readable labels

---

## Known Limitations

- The model performs best on objects and landscapes commonly found in the ImageNet dataset.
- Accuracy may be lower for images that are heavily focused on people or unusual subjects.
- MobileNetV2 requires input images to be resized to **224×224 pixels** before inference.  
- As a result, uploaded images must be downscaled and converted into numerical arrays before being processed by the model. This resizing step can reduce image detail and may negatively impact classification accuracy for complex or high-resolution images. 

---

## Running the Application

### Try it Online

You can test the application using the hosted Streamlit app:

https://image-classifier-test-kirujan.streamlit.app/

### Run Locally

```bash

git clone https://github.com/KirujanJe/image_classifier
cd image-classifier-project
pip install -r requirements.txt
streamlit run app.py

```

Open locally via http://localhost:8501 (verify port is correct to your local machine)