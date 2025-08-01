# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import os

model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pkl')
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)


#we give path where train_model.sav file exist by forward slash it shoul contain .sav file
# loading the saved model
#loaded_model = pickle.load(open('trained_model.pkl', 'rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
