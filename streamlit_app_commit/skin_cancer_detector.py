import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64

loaded_model = tf.keras.models.load_model("model.keras")

def assess_image(photo):

    img = load_img(photo).resize((299,299))
    st.image(img, caption='cool, huh?', output_format='JPEG')
    img = tf.image.resize(img, (299,299), method='bilinear')
    classify_this = [tf.image.resize([img],(299,299), method='bilinear')]

    prediction = loaded_model.predict(classify_this)

    type = np.argmax(prediction)
    percentage = round(prediction[0][type]*100,2)
        
    if type == 0:
                
        st.info(f"The model predicts that this is noncancerous skin with {percentage}% probability.")
       
    elif type == 1:
            
        st.info(f"The model predicts that this is cancerous skin with {percentage}% probability.")

    else:

        st.info(f"Something went wrong.  Try again.")
    
   

def main():
    # Navigation select box in the sidebar

    st.set_page_config(
    page_title="Skinscan",
    page_icon="👋",
    )
 
    st.sidebar.success("Paginate on the sidebar, just click!")

    
    st.title("Steve's Skin Cancer Detector")
    st.title("By: Steve Schluchter")
    st.header("This streamlit app takes input in the form of a user-entered graphic of suspected skin cancer and returns a probability that\
           the image is of a benign or malignant skin leision.")
    st.subheader("This app is NOT a replacement for actual medical diagnostics or medical diagnoses, so don't treat it as either or both.")
    st.markdown('Upload your image here.')
    st.info("The model will start running right after you upload an image or click a demo button.")

    if st.button('Demo cancerous skin image.'):
        #image_path = Path(__file__).with_name("malignant.JPEG")
        #print(image_path)
        f = open('./images/malignant.JPEG','rb')
        image = BytesIO(f.read())
        assess_image(image)
    
    if st.button('Demo noncancerous skin image.'):
        f = open('./images/benign.JPEG', 'rb')
        image = BytesIO(f.read())
        assess_image(image)

    photograph = st.file_uploader('Upload a photo', type=['jpeg'])


    if photograph:
        assess_image(photograph)
        
    

if __name__ == "__main__":
    main()
        