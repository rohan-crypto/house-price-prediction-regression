import streamlit as st
import numpy as np
import joblib

#Load saved model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="House Price Predictor")

st.title("House Price Prediction App")
st.write("Enter house details below to predict the price.")
st.write("Use realistic numbers for accurate predictions.")


#Input Fields
#If we use value i.e. dafault value and also Reset the inputs with the reset button then streamlit gets confused which one to trust and gives
#warning, thus keep the statements with default values in reset_inputs func and remove the value arguments from the inputs
#square_footage = st.number_input("Square Footage (area in sq ft)", min_value=500, max_value=6000, value=2000)
square_footage = st.number_input("Square Footage (area in sq ft)", min_value=500, max_value=6000, value=2000, key="square_footage")
num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, key="num_bedrooms")
num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, key="num_bathrooms")
year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000, key="year_built")
lot_size = st.number_input("Lot Size", min_value=0.1, max_value=10.0, value=2.0, key="lot_size")
garage_size = st.number_input("Garage Size", min_value=0, max_value=5, value=1, key="garage_size")
neighborhood_quality = st.slider("Neighborhood Quality (1 = Low, 10 = High)", 1, 10, 5, key="neighborhood_quality")

#Predict Button
if st.button("Predict Price"):
    input_data = np.array([[square_footage,
                                     num_bedrooms,
                                     num_bathrooms,
                                     year_built,
                                     lot_size,
                                     garage_size,
                                     neighborhood_quality]])
    # Scale input
    input_scaled = scaler.transform(input_data)

    #Predict
    prediction = model.predict(input_scaled) #predict() returns a NumPy array, e.g., [618861.23]

    #[0] → take the first element of that array
    #, → adds commas as thousand separators
    #.2f → rounds to 2 decimal places (float formatting)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")

def reset_inputs():
    st.session_state.square_footage = 2000
    st.session_state.num_bedrooms = 3
    st.session_state.num_bathrooms = 2
    st.session_state.year_built = 2000
    st.session_state.lot_size = 2.0
    st.session_state.garage_size = 1
    st.session_state.neighborhood_quality = 5

#Refreshes the app and resets all inputs to default
st.button("Reset Inputs", on_click=reset_inputs)

#Commenting this because if both the statements inside if will be used together then we'll get error
#if st.button("Reset Inputs", on_click=reset_inputs):
    #reset_inputs()
    #st.rerun()

st.info("The average housing price is $618,861")