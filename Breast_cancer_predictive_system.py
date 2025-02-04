# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 20:15:38 2025

@author: Somendra Mishra
"""

import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/Somendra Mishra/OneDrive/Desktop/Breast_Cancer_Detection/breast_cancer_model.sav','rb'))

input_data = (842302,17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
)


input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast cancer is Banign')