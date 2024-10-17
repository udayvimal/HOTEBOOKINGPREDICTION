from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and scaler from the pickle files
with open('logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve the form data from the POST request
        lead_time = float(request.form['lead_time'])
        adults = int(request.form['adults'])
        previous_cancellations = int(request.form['previous_cancellations'])
        total_of_special_requests = int(request.form['total_of_special_requests'])

        # Prepare input for prediction (make sure it matches the model input shape)
        input_features = np.array([[lead_time, adults, previous_cancellations, total_of_special_requests]])
        
        # Scale the input features using the same scaler used during model training
        input_scaled = scaler.transform(input_features)
        
        # Predict the outcome using the loaded model
        prediction = model.predict(input_scaled)
        
        # Render result back to the index.html page
        prediction_text = f"Booking confirmed! Your reservation is likely to succeed!" if prediction[0] == 1 else f"Booking likely to be canceled. Please reconsider."
        return render_template('index.html', prediction_text=prediction_text)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
