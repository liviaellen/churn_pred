import streamlit as st
import joblib
import requests

'''
# Churner Prediction

This website predict if a customer will churn or not.'''
# model_path="local"
model_path=st.secrets["model_path"]

@st.cache_resource  # 👈 Add the caching decorator
def load_prep():
    return joblib.load('scaler_prep.joblib')



@st.cache_resource  # 👈 Add the caching decorator
def load_model():
    return joblib.load('model.joblib')




nb_past_orders = st.number_input('number of past order', value=5)
avg_basket = st.number_input('Average basket in $', value=50)
total_purchase_cost = st.number_input('Total purchase cost', value=50)
avg_quantity = st.number_input('average quantity', min_value=1, max_value=100, step=1, value=8)
total_quantity = st.number_input('total quantity', min_value=1, max_value=500, step=1, value=40)
avg_nb_unique_products = st.number_input('average number of unique product', value=20)
total_nb_codes = st.number_input('total number of discount code used', value=2)
if st.button('Submit'):
    value_to_predict = [nb_past_orders,avg_basket,total_purchase_cost,avg_quantity,total_quantity,avg_nb_unique_products,total_nb_codes]
    if model_path=="local":
        # Data Preparation
        scaler= load_prep()
        scaler.transform([value_to_predict])
        # Load model
        model = load_model()
        # Make prediction
        prediction = model.predict([value_to_predict])

    else:

        url = 'https://taxifare.lewagon.ai/predict'

        response = requests.get(url, params=value_to_predict)

        prediction = response.json()['prediction']

    # Display
    if prediction[0]==0:
        st.write("Churner")
    else:
        st.write("Not Churner")
