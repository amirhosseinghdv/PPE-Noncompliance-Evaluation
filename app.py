import streamlit as st
st.text('This is TEXT')

import gdown

url = 'https://drive.google.com/uc?export=download&id=1upo5sgFRlAZiPYXm7-nf-yv6ajqkINCz'
output = 'YOLOX.pth'
gdown.download(url, output, quiet=False)


url = 'https://drive.google.com/uc?export=download&id=1AsLvheuc4ljW6bGCMJepLJ8YMW0mnCIb'
output = 'EfficientNet.zip'
gdown.download(url, output, quiet=False)

!unzip /content/EfficientNet.zip
model = tf.keras.models.load_model('./B0_2ndPhaseAll3_ckpt')

print(model.input)
