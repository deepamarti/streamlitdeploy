import streamlit as st

from PIL import Image

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
    img_arr = img_to_array(image) # Turn the image into an array. 
    ## IMPORTANT
    ## Since the model is trained on batches, the model expects an input of (Batch_size,224,224,3), but the image is different.
    ## It has a a size of (224,224,3)
    ## In order to fix this, we just expand the dimensions of the image array to make it (1,224,224,3).
    ## The 1 is there because we are only predicting one image. 
    img_arr = np.expand_dims(img_arr, axis = 0)
    img_arr /= 255 # Important to rescale the immage, otherwise the prediction will be wrong. 
    return img_arr

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
      
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)
    

def predict(image):
    testImage = read_image(image)
    CLASSES = ['F_Banana', 'F_Lemon', 'F_Lulo', 'F_Mango', 'F_Orange', 'F_Strawberry', 'F_Tamarillo', 'F_Tomato', 'S_Banana', 'S_Lemon', 'S_Lulo', 'S_Mango', 'S_Orange', 'S_Strawberry', 'S_Tamarillo', 'S_Tomato']

    model = keras.models.load_model('exportedModels') # 'exportedModels' is a folder not a file. Keras takes care of everything. 
    prediction = model.predict(testImage) # Making the actual prediction. 
    print(prediction) # The model simply returns a list of propabilities for what the object could be. 
    print("\nIndex of the highest probability:", np.argmax(prediction))
    print("\nPrediction: ",(CLASSES[np.argmax(prediction)])) # We want the higest probability. Use that to index int

    """
    classifier_model = "base_dir.h5"
    IMAGE_SHAPE = (224, 224,3)
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    test_image = image.resize((224,224))
    one image 224 x224, 3 color channels
    (1, 224, 224, 3)
    reshape to one (32, 224, 224, 3)

    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [
          'Backpack',
          'Briefcase',
          'Duffle', 
          'Handbag', 
          'Purse']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'Backpack': 0,
          'Briefcase': 0,
          'Duffle': 0, 
          'Handbag': 0, 
          'Purse': 0
}

"""    
    result = f"{CLASSES[np.argmax(prediction)]} with a { (100 * np.argmax(prediction)).round(2) } % confidence." 
    return result


if __name__ == "__main__":
    main()
