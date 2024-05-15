# Author: Manh Ha Nguyen - Finance and Data Science Undergraduate at the University of Adelaide
# Date: 2021-09-26
# Description: main file for deploying the model on streamlit

import streamlit as st
import pandas as pd
import pickle
import sklearn 



# Load the model
model = pickle.load(open('knn_diabeties_model.pkl', 'rb'))

# Initialize variables
pregnancies = 0
glucose = 0
blood_pressure = 0
skin_thickness = 0
insulin = 0
bmi = 0
diabetes_pedigree_function = 0
age = 0

# Create a title for the app
st.title('Diabetes Prediction App')

# Create a form for users to enter the data
pregnancies = st.slider('Pregnancies', 0, 17, 0)
glucose = st.slider('Glucose', 0, 199, 0)
blood_pressure = st.slider('Blood Pressure', 0, 122, 0)
skin_thickness = st.slider('Skin Thickness', 0, 99, 0)
insulin = st.slider('Insulin', 0, 846, 0)
bmi = st.slider('BMI', 0.0, 67.1, 0.0)
diabetes_pedigree_function = st.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.0)
age = st.slider('Age', 21, 81, 21)

# Create a button to make the prediction
if st.button('Predict'):
    # Create a dictionary to store the data
    data = {
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    }

    # Create a dataframe
    df = pd.DataFrame(data)

    # Make the prediction
    prediction = model.predict(df)

    # Display the prediction
    if prediction[0] == 0:
        st.write('The patient does not have diabetes')
    else:
        st.write('The patient has diabetes')
        
    # Display the probability
    probability = model.predict_proba(df)
    st.write('The probability of having diabetes is: ', probability[0][1])
    

