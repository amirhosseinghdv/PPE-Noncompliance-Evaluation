import streamlit as st
st.text('This is TEXT')

import os
import shutil
import gdown
import tensorflow as tf

url = 'https://drive.google.com/uc?export=download&id=1upo5sgFRlAZiPYXm7-nf-yv6ajqkINCz'
output = 'YOLOX.pth'
gdown.download(url, output, quiet=False)


url = 'https://drive.google.com/uc?export=download&id=1-6-SIWsOhPZGnPfL8ZNVZGp1Buz5tiDD'
output = 'keras_metadata.pb'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?export=download&id=1-eLE7SqUEoB-NxRcJ09znYs9_sNasvte'
output = 'saved_model.pb'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?export=download&id=1-ee_xJUDNm5HNs8JFhlDapWJNClkjxuE'
output = 'variables.data-00000-of-00001'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?export=download&id=1-pmC0tEtJsUMylmDAgHtRDxtgubn0FHv'
output = 'variables.index'
gdown.download(url, output, quiet=False)



os.mkdir('EfficientNet')
os.mkdir('EfficientNet/assets')
os.mkdir('EfficientNet/variables')

source = 'keras_metadata.pb'
destination = 'EfficientNet/keras_metadata.pb'
dest = shutil.move(source, destination)

source = 'saved_model.pb'
destination = 'EfficientNet/saved_model.pb'
dest = shutil.move(source, destination)

source = 'variables.index'
destination = 'EfficientNet/variables/variables.index'
dest = shutil.move(source, destination)

source = 'variables.data-00000-of-00001'
destination = 'EfficientNet/variables/variables.data-00000-of-00001'
dest = shutil.move(source, destination)


model = tf.keras.models.load_model('EfficientNet')

print(model.input)
