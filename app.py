from flask import Flask, request, render_template
from model import deploy
import numpy as np

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods= ['POST'])
def predict():
    features = [int(i) for i in request.form.values()]
    features = np.array([features])

    output = deploy(features)

    return render_template('index.html', prediction_text = f'The class is estimated as {str(output)}')

if __name__ =="__main__":
    app.run(host= "0.0.0.0", port=8000 ,debug=True)

