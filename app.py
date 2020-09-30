from flask import Flask,render_template,request
import numpy as np
import pickle
import sklearn


app=Flask(__name__)

model=pickle.loads(open('heart_log_modal.pkl','rb'))

@app.route("/",method=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict",method=['POST'])
def predict():

    if request.method == 'POST':

        age=int(request.form['age'])

        sex=request.form['sex']
        if sex=='male':
            sex=1
        else:
            sex=0

        cp=request.form['cp']
        if cp=="Typical angina":
            cp=0
        elif cp=="Atypical angina":
            cp=1
        elif cp == "Non-anginal pain":
            cp = 2
        else:
            cp=3

        trestbps = int(request.form['trestbps'])

        chol = int(request.form['cholestoral'])

        fbs=request.form["fasting_blood_sugar"]
        if fbs=="<_120_mg/dl":
            fbs=0
        else:
            fbs=1

        restecg=request.form["resting_electr"]
        if restecg == 0:
            restecg=0
        elif restecg== 1:
            restecg=1
        else:
            restecg=2

        thalach= int(request.form['thalach'])

        exang = int(request.form['exang'])
        if exang==0:
            exang=0
        else:
            exang=1

        oldpeak = float(request.form['oldpeak'])

        slope=request.form['slope']
        if slope=="Upsloping":
            slope=0
        elif slope=="Flat":
            slope=1
        else:
            slope=2

        ca=int(request.form['ca'])

        thal=int(request.form['thal'])

        prediction=model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])

        return render_template('result.html', prediction=prediction)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)








