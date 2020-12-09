from flask import *
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB

##Loading the model

clf = joblib.load('message_model.pkl')
tfidf = joblib.load('transform.pkl')
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction_text = my_prediction)


if __name__ == '__main__':
    app.run(debug= True)

