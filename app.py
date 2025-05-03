import streamlit as st
import numpy as np
import tensorflow as tf

# Load model
interpreter = tf.lite.Interpreter(model_path="diamond_price_prediction_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Manual encoding sesuai dengan training
cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

st.title("Diamond Price Prediction")

# Input form
carat = st.number_input("Carat", min_value=0.0, step=0.01)
cut = st.selectbox("Cut", list(cut_map.keys()))
color = st.selectbox("Color", list(color_map.keys()))
clarity = st.selectbox("Clarity", list(clarity_map.keys()))
depth = st.number_input("Depth", min_value=0.0, step=0.1)
table = st.number_input("Table", min_value=0.0, step=0.1)
x = st.number_input("X (Length in mm)", min_value=0.0, step=0.1)
y = st.number_input("Y (Width in mm)", min_value=0.0, step=0.1)
z = st.number_input("Z (Depth in mm)", min_value=0.0, step=0.1)

if st.button("Predict"):
    # Encode inputs
    cut_encoded = cut_map[cut]
    color_encoded = color_map[color]
    clarity_encoded = clarity_map[clarity]

    input_data = np.array([[carat, cut_encoded, color_encoded, clarity_encoded, depth, table, x, y, z]], dtype=np.float32)

    # Set input and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    st.success(f"Predicted Diamond Price: ${prediction[0][0]:,.2f}")
