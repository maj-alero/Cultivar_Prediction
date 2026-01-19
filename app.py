from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open('model.h5', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    scaler = saved_data['scaler']
    target_names = saved_data['target_names']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get inputs (we will take the first 4 features for simplicity)
    user_inputs = [float(x) for x in request.form.values()]
    
    # KNN expects 13 features. Fill the rest with average-like values (0 after scaling)
    full_features = np.zeros((1, 13))
    full_features[0, :len(user_inputs)] = user_inputs
    
    # Scale and Predict
    scaled_features = scaler.transform(full_features)
    prediction = model.predict(scaled_features)
    
    wine_type = target_names[prediction[0]]
    
    return render_template('index.html', prediction_text=f'Predicted Wine Class: {wine_type.capitalize()}')

if __name__ == "__main__":
    app.run(debug=True)