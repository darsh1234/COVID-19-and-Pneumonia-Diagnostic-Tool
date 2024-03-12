import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.applications.densenet import preprocess_input

#set title

st.title('COVID-19 & Pneumonia Classification')

#set header

st.header('Please upload a chest X-ray image')

#upload file

file = st.file_uploader('',type=['jpeg','jpg','png'])


#load classifier

model = load_model('model/modelCovid19___1.keras')
class_names = ['COVID19', 'NORMAL', 'PNEUMONIA']

#display image

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image,use_column_width=True) 


    #classify image

    image = image.resize((300, 300))  # Resize the image
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]

    class_name , conf_score = class_names[predicted_class], confidence

    #write classification

    st.write("# {}".format(class_name))
    st.write("## Confidence Score: {}%".format(conf_score*100))