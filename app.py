import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

 
app = Flask(__name__)

# load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scalar.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    print('request', request)
    data = request.json['data']
    print('data', data)
    print('pppp', np.array(list(data.values())).reshape(1, -1))

    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)

    print('output', output)

    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug=True)
