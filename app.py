from flask import Flask,render_template, request , redirect,url_for
import numpy as np
import pickle
import pandas as pd

model = pickle.load(open('heart.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def start():
    return render_template('index.html')

@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/heartform.html')
def form():
    return render_template('heartform.html')

@app.route('/submit', methods=['POST'])
def predict():
    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = float(request.form['oldpeak'])
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']
    
    # create a list of input features
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    #return(input_data)
     # changing the input_data to numpy array
    arr = np.array(input_data)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = arr.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)
    
    
    # return the prediction result
    return render_template('next.html',data=prediction)
    



  

if __name__ == '__main__':
    app.run(debug=True)
