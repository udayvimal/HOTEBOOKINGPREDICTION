import streamlit as st
import pickle
import numpy as np

# Load model and scaler once (cache to speed up)
@st.cache_data(show_spinner=False)
def load_model_and_scaler():
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# Page config
st.set_page_config(
    page_title="🏨 Hotel Booking Prediction",
    page_icon="🏨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling (dark background, container styling, fonts)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');

    body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #002D62, #0B6DAB);
        color: #fff;
    }
    .main-container {
        background-color: rgba(255,255,255,0.1);
        padding: 40px 60px;
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        max-width: 700px;
        margin: auto;
        margin-top: 60px;
        margin-bottom: 60px;
    }
    h1 {
        text-align: center;
        font-weight: 700;
        color: #FFD700;
        margin-bottom: 30px;
        font-size: 3rem;
        text-shadow: 0 0 10px #FFD700;
    }
    label {
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 10px;
        color: #eee;
    }
    input, .stNumberInput>div>input {
        background-color: #f0f0f0;
        color: #000;
        border-radius: 8px;
        padding: 12px;
        font-size: 1rem;
        border: none;
        width: 100%;
    }
    button {
        background-color: #002D62;
        color: white;
        border: none;
        padding: 15px;
        font-size: 1.3rem;
        cursor: pointer;
        border-radius: 50px;
        font-weight: 700;
        width: 100%;
        margin-top: 25px;
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 0 15px #FFD700;
    }
    button:hover {
        background-color: #0050A1;
        box-shadow: 0 0 25px #FFD700;
    }
    .result-box {
        margin-top: 30px;
        padding: 20px;
        border-radius: 15px;
        font-size: 1.5rem;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 0 30px #FFD700;
    }
    .success {
        background-color: #28a745;
        color: white;
    }
    .error {
        background-color: #dc3545;
        color: white;
    }
    footer {
        text-align: center;
        color: #ddd;
        margin-bottom: 20px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("<h1>Hotel Booking Prediction</h1>", unsafe_allow_html=True)

# Input fields using Streamlit number inputs
lead_time = st.number_input(
    "Lead Time (days):", min_value=0, step=1, format="%d", help="Enter number of days before booking"
)
adults = st.number_input(
    "Number of Adults:", min_value=1, step=1, format="%d", help="Enter number of adults"
)
previous_cancellations = st.number_input(
    "Previous Cancellations:", min_value=0, step=1, format="%d", help="Number of previous cancellations"
)
total_special_requests = st.number_input(
    "Total Special Requests:", min_value=0, step=1, format="%d", help="Enter total special requests"
)

# Predict button
predict_clicked = st.button("Predict Booking Success")

if predict_clicked:
    try:
        # Prepare input array
        input_features = np.array(
            [[lead_time, adults, previous_cancellations, total_special_requests]]
        )
        # Scale input
        input_scaled = scaler.transform(input_features)
        # Predict
        prediction = model.predict(input_scaled)[0]

        # Show results with styled box
        if prediction == 1:
            st.markdown(
                '<div class="result-box success">Booking confirmed! Your reservation is likely to succeed!</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-box error">Booking likely to be canceled. Please reconsider your booking details.</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <footer>
        &copy; 2024 Hotel Predictor. Developed by <a href="https://yourcompany.com" style="color:#FFD700;" target="_blank">Your Company</a>.
    </footer>
    """,
    unsafe_allow_html=True,
)
