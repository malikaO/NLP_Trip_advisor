from flask import Flask, render_template, request, Response
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
from prediction import Prediction

import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

#app initialization
app = Flask(__name__, template_folder='templates')

# global variables for data persistence across requests
model_output=""
cm = Prediction("text")
cm.preprocessing()

# main index page route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    cr = cm.getcr()
    print(cr)
    return render_template("dashboard.html", dataframe=cr.to_html())

@app.route('/tables')
def tables():
    df = cm.getsampledata(16)
    df2 = cm.getsampledata2(16)
    return render_template("tables.html", dataframe=df.to_html(), dataframe2=df2.to_html())


@app.route('/confusion_matrix.png')
def confusion_matrix():
    fig = cm.create_cm()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


# route for prediction of sentiment analysis model and classifier
@app.route('/predict', methods=['POST'])
def predict():
    # retrieve global variables to store output
    global model_output
    
    # get text from the incoming request (submitted on predict button click)
    text = request.form['input_text']

    option = request.form['options']

    # store model output
    model_output = cm.getprediction([text], option)
    print(model_output)

    return model_output[0]


if __name__ == "__main__":
    app.run(debug=True)