# READ ME:

To run and test:

1. Download and check all dependencies listed below including streamlit (https://docs.streamlit.io/library/get-started/installation).
2. Run the command "streamlit run app.py"
3. Drag and drop an image into the indicated area and click classify. For your convinience, we have added an image called "myTomato.jpeg" for testing.
4. Scroll below the image to see the results of the model.

Dependecies Needed: 

[[source]]

url = "https://pypi.org/simple"

verify_ssl = true

name = "pypi"


[packages]
tensorflow = "==2.2.0"

streamlit = "==0.78.0"

tensorflow-hub = "==0.11.0"

numpy = "==1.19.5"

matplotlib = "==3.2.2"

Pillow = "==8.1.2"


[dev-packages]


[requires]

python_version = "3.7"


