import streamlit as st
import os
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
import time

fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Fruit Classifier')

st.markdown("Welcome to our web application that classifies fruits based on type and ripeness.")

## Boilerplate code to get an imge from a folder. 
def read_image(image):
    #newsize = (224, 224)
    #image.resize(newsize)
    img_arr = img_to_array(image) # Turn the image into an array. 
    ## IMPORTANT
    ## Since the model is trained on batches, the model expects an input of (Batch_size,224,224,3), but the image is different.
    ## It has a a size of (224,224,3)
    ## In order to fix this, we just expand the dimensions of the image array to make it (1,224,224,3).
    ## The 1 is there because we are only predicting one image. 
    img_arr = np.expand_dims(img_arr, axis = 0)
    img_arr /= 255 # Important to rescale the immage, otherwise the prediction will be wrong. 
    return img_arr

def load_image(image_file):
    img = Image.open(image_file)
    return img

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:
        file_details = {"FileName":file_uploaded.name,"FileType":file_uploaded.type}
        st.write(file_details)
        img = load_image(file_uploaded)
        #st.image(img,height=250,width=250)
        with open(os.path.join("tempDir",file_uploaded.name),"wb") as f:
            f.write(file_uploaded.getbuffer())
        #image_data = file_uploaded.read()
        #bytes_data = image_data.getvalue()
        #scr = BytesIO(file_uploaded.getvalue()).read()
        st.success("Saved File")
        #st.write(file_uploaded.name)
        #data = Image.open(img)
        img = load_img('./tempDir/' + file_uploaded.name, target_size = (224, 224))
        img = img_to_array(img)
        image = Image.open(file_uploaded)

        img = np.expand_dims(img, axis = 0)
        img /= 255
        st.write("hi")
        st.image(img, caption='Uploaded Image', use_column_width=True)
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                #plt.imshow(img)
                #plt.axis("off")
                predictions = predict(img)
                time.sleep(1)
                outputer(predictions)
                
def outputer(predictions):
    if predictions == 'S_Strawberry':
        st.success('Classified')
        st.write('Spoiled Strawberry')
        st.pyplot(fig)
    else:
        st.success('Classified')
        st.write(predictions)
        st.pyplot(fig)

        

def predict(image):
    CLASSES = ['F_Banana', 'F_Lemon', 'F_Lulo', 'F_Mango', 'F_Orange', 'F_Strawberry', 'F_Tamarillo', 'F_Tomato', 'S_Banana', 'S_Lemon', 'S_Lulo', 'S_Mango', 'S_Orange', 'S_Strawberry', 'S_Tamarillo', 'S_Tomato']

    model = keras.models.load_model('exportedModels') # 'exportedModels' is a folder not a file. Keras takes care of everything. 
    prediction = model.predict(image) # Making the actual prediction. 
    #print(prediction) # The model simply returns a list of propabilities for what the object could be. 
    #print("\nIndex of the highest probability:", np.argmax(prediction))
    #print("\nPrediction: ",(CLASSES[np.argmax(prediction)])) # We want the higest probability. Use that to index int
  
    result = CLASSES[np.argmax(prediction)]
    #f"{CLASSES[np.argmax(prediction)]} with a { (100 * np.argmax(prediction)).round(2) } % confidence." 
    return result


if __name__ == "__main__":
    main()
