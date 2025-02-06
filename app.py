import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle 

# Loading the model
model = tf.keras.models.load_model('model.h5')

# loading the pickle files
with open('onehot_encoder_geo.pkl','rb') as f:
    onehot_encoder_geo=pickle.load(f)

with open('label_encoder_gender.pkl','rb') as f:
    label_encoder_gender=pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)


#Streamlit 

st.title("Salary Estimator")

# User input data 
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age',20,100)
balance = st.number_input('Balance')
credit_Score = st.number_input('Credit Score')
exited = st.slider('Exited',0,1)
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,3)
has_cr_card = st.select_slider('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])


# Preparing the input data

input_data = pd.DataFrame({
    'CreditScore' : [credit_Score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' :[age],
    'Tenure' :[tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]

})

# encoding geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


# concatenating the data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# scaling the data
scaled_input_data = scaler.transform(input_data)


# prediction
predicted_salary = model.predict(scaled_input_data)[0][0]

# Display the predicted salary
st.write(f'Estimated Salary: ${predicted_salary:,.2f}')