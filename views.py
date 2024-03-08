from flask import Flask, url_for, render_template, request, redirect
import neattext.functions as nfx
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import joblib
import numpy as np

labels = ['Fake', 'Real']

model = load_model('./Models/fake_news_detector.h5')

token = joblib.load('./Models/token.joblib')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        title = request.form['title']
        title_cleaned = nfx.remove_stopwords(title)

        sequence = token.texts_to_sequences([title_cleaned])

        x_test = pad_sequences(sequence, maxlen=100)
        
        prediction = model.predict(x_test)

        label = labels[np.argmax(prediction)]

        return render_template('index.html', label=label)

    return render_template('index.html')