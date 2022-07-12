from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
import sklearn as sk
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

model = pickle.load(open('model_mnb.pkl','rb'))

@app.route('/', methods=['GET','POST'])
def root():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    dataset = pd.read_csv('deceptive-opinion.csv')
    required_dataset = dataset[['verdict', 'review']]
    required_dataset.loc[required_dataset['verdict'] == 'deceptive', 'verdict'] = 0
    required_dataset.loc[required_dataset['verdict'] == 'truthful', 'verdict'] = 1
    X = required_dataset['review']
    Y = np.asarray(required_dataset['verdict'], dtype = int)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42) # 75% training and 25% test
    cv = CountVectorizer()
    x = cv.fit_transform(X_train)
    y = cv.transform(X_test)

    message = request.form.get('enteredinfo')
    data = [message]
    
    vect = cv.transform(data).toarray()
    prediction = model.predict(vect)

    return render_template('result.html', prediction_text = prediction)

@app.route('/tryagain')
def tryagain():
    return render_template('index.html')

    
if __name__ == '__main__':
    app.run(debug=True)