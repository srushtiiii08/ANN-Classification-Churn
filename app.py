#streamlit end to end webapp where I will be able to enter inputs and do th prediction from there itself

import streamlit as st 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#load the trained model
model= tf.keras.models.load_model('model.h5')

#load the encoders and scalers
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo_pickle=pickle.load(file)
    
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender_pickle=pickle.load(file)
    
with open('scaler.pkl', 'rb') as file:
    scaler_pickle=pickle.load(file)
    

#streamlit app - using streamlit cz dont want to use html css
st.title('Customer Churn Prediction')

#user input

#Create a dropdown called ‘Geography’ where the options are exactly the categories that the encoder learned for the Geography feature.
#For a fitted OneHotEncoder, categories_ is an attribute that stores the list of categories it learned for each input feature during fit / fit_transform
#so for 1 geography, the stored categories are 
geography = st.selectbox('Geography', onehot_encoder_geo_pickle.categories_[0])   #[0] so it by default selects the 1st index value
gender = st.selectbox('Gender', label_encoder_gender_pickle.classes_)   #.classes is for label encoder as .categories is for onehot encoder
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender_pickle.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo_pickle.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo_pickle.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
#reset_index: turns the current index into a simple 0, 1, 2, ... index.
#drop=True: means “don’t keep the old index as a column; just throw it away”.
#This is important because when concatenating columns, pandas aligns rows by index. 
# If the indices don’t match, you can get NaNs or extra rows - so the rows not matching are dropped.
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler_pickle.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')