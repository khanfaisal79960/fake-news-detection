import streamlit as st
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import neattext.functions as nfx
import joblib
import numpy as np

labels = ['Fake', 'Real']

model = load_model('../Models/fake_news_detector.h5')
token = joblib.load('../Models/token.joblib')


def make_prediction(title):
  title_without_stopwords = nfx.remove_stopwords(title)
  sequence = token.texts_to_sequences([title_without_stopwords])
  x_test = pad_sequences(sequence, maxlen=100)

  prediction = model.predict(x_test)
  label = labels[np.argmax(prediction)]
  return label


title = st.text_area(label='Enter the Title of the News')

if st.button(label='Submit'):
  st.write(f'The news is likely to be {make_prediction(title)}')

footer = """<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by Khan Faisal</p><br>
<a href='https://khanfaisal.netlify.app'>  Portfolio</a>
<a href="https://github.com/khanfaisal79960">  Github</a>
<a href="https://medium.com@khanfaisal79960">  Medium</a>
<a href="https://www.linkedin.com/in/khanfaisal79960">  Linkedin</a>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
