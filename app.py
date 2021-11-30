import streamlit as st
import os
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
import time

# Sources:
# https://blog.jcharistech.com/2021/01/21/how-to-save-uploaded-files-to-directory-in-streamlit-apps/
# https://towardsdatascience.com/image-classification-of-uploaded-files-using-streamlits-killer-new-feature-7dd6aa35fe0
# https://medium.com/analytics-vidhya/deploying-image-classification-model-on-streamlit-e9579ccda157

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
        img = load_image(file_uploaded)
        with open(os.path.join("tempDir",file_uploaded.name),"wb") as f:
            f.write(file_uploaded.getbuffer())

        img = load_img('./tempDir/' + file_uploaded.name, target_size = (112, 112))
        img = img_to_array(img)

        red = img[:,:,2].copy()
        blue = img[:,:,0].copy()

        img[:,:,0] = red
        img[:,:,2] = blue

        img /= 255

        image = Image.open(file_uploaded)

        img = np.expand_dims(img, axis = 0)
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

    if predictions == 'F_Lemon':
        st.success('Classified')
        st.write('Fresh Lemon')

    if predictions == 'F_Strawberry':
        st.success('Classified')
        st.write('Fresh Strawberry')

    if predictions == 'F_Banana':
        st.success('Classified')
        st.write('Fresh Banana')

    if predictions == 'S_Banana':
        st.success('Classified')
        st.write('Spoiled Banana')

    if predictions == 'S_Lemon':
        st.success('Classified')
        st.write('Spoiled Lemon')

    if predictions == 'S_Orange':
        st.success('Classified')
        st.write('Spoiled Orange')

    if predictions == 'F_Orange':
        st.success('Classified')
        st.write('Fresh Orange')

    if predictions == 'S_Mango':
        st.success('Classified')
        st.write('Spoiled Mango')

    if predictions == 'F_Mango':
        st.success('Classified')
        st.write('Fresh Mango')

    if predictions == 'S_Lulo':
        st.success('Classified')
        st.write('Spoiled Lulo')

    if predictions == 'F_Lulo':
        st.success('Classified')
        st.write('Fresh Lulo')

    if predictions == 'F_Tamarillo':
        st.success('Classified')
        st.write('Fresh Tamarillo')

    if predictions == 'S_Tamarillo':
        st.success('Classified')
        st.write('Spoiled Tamarillo')

    if predictions == 'F_Tomato':
        st.success('Classified')
        st.write('Fresh Tomato')

    if predictions == 'S_Tomato':
        st.success('Classified')
        st.write('Spoiled Tomato')


        
def predict(image):
    CLASSES = ['F_Banana', 'F_Lemon', 'F_Lulo', 'F_Mango', 'F_Orange', 'F_Strawberry', 'F_Tamarillo', 'F_Tomato', 'S_Banana', 'S_Lemon', 'S_Lulo', 'S_Mango', 'S_Orange', 'S_Strawberry', 'S_Tamarillo', 'S_Tomato']

    model = keras.models.load_model('exportedModel5') # 'exportedModels' is a folder not a file. Keras takes care of everything. 
    prediction = model.predict(image) # Making the actual prediction. 
    #print(prediction) # The model simply returns a list of propabilities for what the object could be. 
    #print("\nIndex of the highest probability:", np.argmax(prediction))
    #print("\nPrediction: ",(CLASSES[np.argmax(prediction)])) # We want the higest probability. Use that to index int
  
    result = CLASSES[np.argmax(prediction)]
    #f"{CLASSES[np.argmax(prediction)]} with a { (100 * np.argmax(prediction)).round(2) } % confidence." 
    return result


if __name__ == "__main__":
    main()
