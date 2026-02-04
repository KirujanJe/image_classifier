# Author: Kirujan Je / December 2025
# Tech: Python, UV, Streamlit, Tensorflow, Numpy, OpenCV, & PIL. 
# ML Algorithm: MobileNetV2, a light weight CNN is used to classify the images. From Tensorflow Keras API.

import cv2 #opencv
import numpy as np #installed within tensorflow as default
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image #image library in python, in cv2 as default

# mobilenet_v2 is a popular pretrained model in the tensorflow library,
# It is a lightweight model, suitbale to use on a laptop
# this avoids having to train our own model, and a significant amount of time
# MobileNetV2 is a convolutional nueral network (CNN)

# simple function to load model
def load_model():
    model = MobileNetV2(weights='imagenet') # weights (unique numbers) act as the learned values for the model to work
    return model

# process image for something MobileNetV2 can understand
# we have to format any image we upload into the appropriate format for MobileNetV2
def preprocess_image(image):
    img = np.array(image) # comvert the image into arrays, ie. lists of numbers, that represent the pixels in rows and columns.  
    img = cv2.resize(img, (224,224)) # resizing the image to 224 by 224, this is what MobileNetV2 accepts. We will loose some details.
    img = preprocess_input(img)
    img = np.expand_dims(img, axis = 0) 
    
    # ^ (exapnd.dims) wrapping the image into a list of images ie. adding another dimension to the image, 
    # to satisfy the input for the model. As it typically analyzes multiple images at a time. 
    return img

# now we load the processed image into the model
def classify_image(model, image):
    try: 
        # it will take the numeric array values output of the model and convert into string labels
        # the output index of the array, indicate confidences of different classification values
        # the decoded_predictions will decode what the classifcations and confidences actually are
        processed_image = preprocess_image(image) #preprocess image
        predictions = model.predict(processed_image) #pass to the model
        decoded_predictions = decode_predictions(predictions, top=3)[0] #take the top 3 predictions of the only response [0], since one image only
        return decoded_predictions # to print in the streamlit ui
    
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None

# create UI with streamlit in main function
def main():
    st.set_page_config(page_title="Image Classifier using AI", page_icon ='🔬', layout="centered")
    st.title('Image Classification via AI')

    st.markdown("""---""")

    st.write('Tech: Python, UV, Streamlit, Tensorflow, Numpy, OpenCV, & PIL. ')
    st.write('ML Algorithm: MobileNetV2, a light weight CNN is used to classify the images. From Tensorflow Keras API.')

    st.markdown("""---""")

    st.write('Please uplaod an image below')

    @st.cache_resource
    def load_cached_model(): # this will prevent running the model everytime
        return load_model()  
    # caching this resource, will just return the loaded model if it already has loaded in a previous run of the streamlit application.
    
    model = load_cached_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg','png'])

    if uploaded_file is not None:
        image = st.image(
            uploaded_file, caption="Uploaded Image", width = "content"
        )
        # display the image

        btn = st.button('Classify Image', icon = '🔍')

        if btn:
            with st.spinner("Analyzing Image..."):
                image = Image.open(uploaded_file)
                # Image is coming from PIL, this utlity allows us to use images in python
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Predictions")
                    # we are going to loop through the predictions and label them with percentages
                    for _, label, score in predictions: # _ is known as the anonymous variable in python
                        st.write(f"**{label}**: {score:2%}") 
 
if __name__ =="__main__":
    main() 
    #call and run the main function if we are running python directly 

