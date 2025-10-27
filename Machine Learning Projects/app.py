from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load model and column info
model = pickle.load(open('home_prices_model.pickle', 'rb'))
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']
locations = data_columns[3:]  # first 3 are total_sqft, bath, bhk

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    total_sqft = float(request.form['total_sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])
    location = request.form['location']

    # Create feature array
    x = np.zeros(len(data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    prediction = model.predict([x])[0]
    return jsonify({'estimated_price': round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True)