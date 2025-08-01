# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 22:55:25 2025

@author: Jitanjali Patel
"""

import numpy as np
import pickle
import streamlit as st
import os

model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pkl')
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)



#loaded_model = pickle.load(open('trained_model.sav', 'rb'))
#with open('trained_model.pkl', 'rb') as f:
   # loaded_model = pickle.load(f)


# creating a function for prediction
def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    #print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'


def main():

#giving a title
  st.title("Diabetes Prediction Web App")

#getting the input data from user
  
  Pregnancies  = st.text_input('Number of Pregnancies')
  Glucose = st.text_input('Glucose Level')
  BloodPressure = st.text_input('Blood Pressure Value')
  SkinThickness = st.text_input('Skin Thickness Value')
  Insulin = st.text_input('Insulin Level')
  BMI = st.text_input('BMI Value')
  DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
  Age = st.text_input('Age of the Person')


  # code for prediction
  diagnosis = ''

# creating a button for prediction part
  if st.button('Diabetes Test Result'):
     diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
     
     
     
     
     
  st.success(diagnosis)



if __name__ == '__main__':
   main()







