from flask import Flask, redirect, url_for, request, render_template
from init import app, db
from create_db import Prediction
from showtable import getHtmlTable

import random
import json
import re

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix

labels = json.load(open("labels.json"))




app = Flask(__name__)

@app.route('/')
def home_page():
    num = random.randint(1,100)
    return render_template('index.html', res = num)


@app.route('/recognize', methods=['GET'])
def recognize():
    """View that process a POST with new song input

    :return: redirect to index page
    """

    size = 196
    base = 28
    num_classes = 62
    factor = int(size/base)
    list = request.args.get('hiddenBox')
    large_image = np.array([float(elem) for elem in list.split(',')]).reshape(size,size)
    small_image = np.zeros(base*base).reshape(base, base)

    for i in range(base):
        for j in range(base):
            print(i,j)
            # mean of the positive values
            small_image[i,j] = large_image[factor*i:factor*(i+1), factor*j:factor*(j+1)][np.where(large_image[factor*i:factor*(i+1), factor*j:factor*(j+1)]>0)].mean()
            
            # 255 if any value is greater than 0
            # small_image[i,j] = 255 if large_image[factor*i:factor*(i+1), factor*j:factor*(j+1)].sum()>0 else 0



    small_image[np.isnan(small_image)] = 0.0
    small_image = small_image.reshape(base*base)


    new_model = load_model('../develop/models/deep_cnn_byclass1.h5') 
    input = np.array(small_image).reshape(1,28,28,1)/255
    prediction = new_model.predict(input)
    prediction = str(prediction.reshape(num_classes).argsort()[-1])
    prediction = labels[prediction]
    if prediction == 'null' : 
        output = "Failed to recognize, try again!"
    else:
        output = "Predicted " + prediction + "!"

    # output = np.array2string(np.round(prediction, 3)).strip('[]')
    # output = ", ".join([str(float(i)) for i in re.findall("\\d+.\\d*", output)])
    # output = "Probabilities = " + output
    # output = output + "\n Top 3 Predictions = " + np.array2string(prediction.reshape(num_classes).argsort()[-3:][::-1]).strip('[]').replace(" ", ", ")
    
        character = prediction
        # sunlight = float(request.form["sunlight"])
        # outcome = predict(date,area,genre,visitor,avg_temp,low_temp,precip,wind,sunlight,model_outcome)
        # visitor_pred = round(max(outcome,0))
        # print(visitor_pred)
        instance = Prediction(character=character)
        db.session.add(instance)
        db.session.commit()
    
    
    return render_template('recognize.html', output = output)


@app.route('/history',methods=["GET"])
def show_hist():
    db.session.commit()
    results = db.session.execute('SELECT * FROM PREDICTION;')
    table = getHtmlTable(results)
    return render_template("history.html",table=table)


if __name__ == '__main__':
    # new_model = load_model('deep_cnn_byclass1.h5')
    # app.run(host='0.0.0.0', debug=True)ed
    app.run(debug = True)