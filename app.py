import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application= Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['POST','GET'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data =CustomData(
            station = request.form.get('station'),
            no2_hour = int(request.form.get('no2_hour')),
            no2_quality = request.form.get('no2_quality'),
            no2_value = int(request.form.get('no2_value')),
            month = int(request.form.get('month')),
            day = int(request.form.get('day')),
            longitude=float(request.form.get('longitude')),
            latitude=float(request.form.get('latitude'))

        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predic_pipeline=PredictPipeline()

        results=predic_pipeline.predict(pred_df)
        print(results)

        if results[0]==1.0:
            display="Good"
        else:
            display="Bad"
        return render_template('Home.html', results=display)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)

    


