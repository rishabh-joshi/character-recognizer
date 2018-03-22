from flask import Flask, url_for, request, render_template
from __init__ import app, db
from create_db import Prediction
from showtable import get_html_table
import numpy as np
from helper import *


@app.route('/', methods=['GET'])
def home_page():
    # render index.html
    return render_template('index.html', output="")


@app.route('/recognize', methods=['POST'])
def recognize():
    # height and width of the square canvas image
    size = 196

    # the base height and width the image should be shrunk to
    base = 28

    # getting the pixel data from the html
    pixel_list = request.form["hiddenBox"]

    # shrink the image down to 28x28
    small_image = compress_image(pixel_list, size, base)

    # defining the metadata file path
    metadata_file_path = "develop/metadata.yaml"

    # reading the metadata file
    metadata = read_metadata(metadata_file_path)

    # specify the model path and get the predicted character
    model_path = "develop/models/" + metadata['model_name']
    prediction = predict_character(model_path, small_image)

    # if no prediction exists
    if prediction == 'null':

        # default output
        output = "Failed to recognize!"
    else:

        # output the predicted character
        output = prediction

        # saving the prediction into RDS 
        store_prediction(prediction)

    # render recognize.html with the appropriate output
    return render_template('recognize.html', output=prediction)


@app.route('/history', methods=["GET"])
def show_history():
    # execute the query to select the last 5 entries in the "Prediction" table
    db.session.commit()
    results = db.session.execute('SELECT * FROM PREDICTION ORDER BY id DESC LIMIT 5;')

    # get the html code to display this table
    table = get_html_table(results)

    # render history.html with the table
    return render_template("history.html", table=table)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)