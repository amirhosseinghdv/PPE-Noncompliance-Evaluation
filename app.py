import streamlit as st
st.text('This is TEXT')

import os
import shutil
import gdown
import tensorflow as tf

st.text('All imported.')

url = 'https://drive.google.com/uc?export=download&id=1upo5sgFRlAZiPYXm7-nf-yv6ajqkINCz'
output = 'YOLOX.pth'
gdown.download(url, output, quiet=False)


model = tf.keras.models.load_model('EfficientNet')

print(model.input)

st.text('Done.')
