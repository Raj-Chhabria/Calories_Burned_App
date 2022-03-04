from crypt import methods
import pandas as pd
import numpy as np
from flask import Flask,render_template,request
import pickle

model = pickle.load(open('model.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/result',methods=['POST'])
def result():
    gender = request.form.get('gender')
    age = request.form.get('age')
    height = request.form.get('height')
    weight = request.form.get('weight')
    duration = request.form.get('duration')
    heartrate = request.form.get('heartrate')
    bodytemp = request.form.get('bodytemp')

    prediction = model.predict(pd.DataFrame(columns=['Gender','Age','Height','Weight','Duration','Heart_Rate','Body_temp'],
    data=np.array([gender,age,height,weight,duration,heartrate,bodytemp]).reshape(1,7)))

    
    return render_template('res.html',data = prediction)

if __name__=='__main__':
    app.run(debug=True)