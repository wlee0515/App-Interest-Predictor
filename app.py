from flask import Flask, render_template, request
from joblib import dump, load
import pandas as pd
import numpy as np
import os
import json

from InterestVectorConvertor import InterestVectorConvertor

app = Flask(__name__, static_url_path='/static/')

@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict_interest', methods=['POST', 'GET'])
def predict_interest():
    # get the parameters
    user_age = float(request.form['user_age'])
    user_gender = int(request.form['user_gender'])
    user_longitude = float(request.form['user_longitude'])
    user_latitude = float(request.form['user_latitude'])
    day_of_week = int(request.form['day_of_week'])
    hour_of_day = int(request.form['hour_of_day'])

    # load the model and predict
    model = load('model/LinearRegression.joblib')
    converter = load('model/InterestVectorConverter.joblib')
    
    iAge = user_age
    iDay_of_Week = day_of_week
    iHour_of_Day = hour_of_day
    iLongitude = user_longitude
    iLatitude = user_latitude
    iSex = "Male"
    if user_gender > 0.5:
        iSex = "Female"

    prediction = model.predict([[iAge, iDay_of_Week, iHour_of_Day, iLongitude, iLatitude, (iSex != "Male"), (iSex == "Male")]])
    df_LabelCategory = converter.convertVector(prediction)
    
    list_ordered = df_LabelCategory.reset_index()["category-mod"].to_list()
    list_ordered_str = json.dumps(list_ordered)
    
    return render_template('results.html',
                           user_age = iAge,
                           user_gender = iSex,
                           user_longitude = iLongitude,
                           user_latitude = iLatitude,
                           day_of_week = iDay_of_Week,
                           hour_of_day = iHour_of_Day,
                           interest_list=list_ordered_str
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
