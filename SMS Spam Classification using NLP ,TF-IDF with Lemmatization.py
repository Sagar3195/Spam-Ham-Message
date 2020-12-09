### Importing required libraries
import pandas as pd
import numpy as np

#Loading sms spam classifier dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

print(data.head())

print(data.columns)

data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis = 1, inplace =True)

print(data.shape)

#Checking missing values in dataset
print(data.isnull().sum())

#### We can see that there is no missing values in dataset

#Now Data cleaning and preprocessing

import nltk
import re

from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer

#now create variable for Lemmatization
lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review) #joining word into sentence
    corpus.append(review)

#print(corpus)

print(len(corpus))

#Now create model for TF - IDF
from sklearn.feature_extraction.text import TfidfVectorizer

#Here we set max feature bcoz some of the words are not more frequently present.
#It may be present one or two times.
vectorizer  = TfidfVectorizer(max_features= 5000)

#Now creating independent variable
X = vectorizer.fit_transform(corpus)

X = X.toarray()
## no of observations and no of features
print(X.shape)

y = pd.get_dummies(data['class'])

print(y.head())


#Here we have dependent variable
y = y.iloc[:, 1]

#Now splitting dataset into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

print(x_train.shape, x_test.shape)

#Now create classification model using Naive Bayes
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

#Now fit the model and the predict the model
model.fit(x_train, y_train)

prediction = model.predict(x_test)

print(prediction)


#Now check the accuracy and performance of the model
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_test, prediction)
print("Accuracy of the model: ", accuracy)

cm = confusion_matrix(y_test, prediction)
print("Confusion matrix:\n ", cm)
#Here we can see that using Tf-Idf method, lematization and naive bayer classifier we get 97% accuracy.

#Now we predict the model on other dataset.

df = pd.DataFrame([['The message is not a spam'],['The computer is hacked']], columns = ['message'])

print(df.head())

#Now data cleaning and preprocessing
corpus_sent = []
for i in range(len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus_sent.append(review)

print(corpus_sent)


print(len(corpus_sent))

#Now Tf-IDF vectorizer
y = vectorizer.transform(corpus_sent)

y = y.toarray()
print(y)

#Now we will predict the label
pred_y = model.predict(y)

print(pred_y.shape)

print(pred_y)


label = pred_y[0]

if label == 1:
    print("Message is a ham")
else:
    print('Message is a spam')

#Here above we predict model on new dataset.
##Now we save the model
import joblib

#joblib.dump(model, 'message_model.pkl')
joblib.dump(vectorizer, 'transform.pkl')



