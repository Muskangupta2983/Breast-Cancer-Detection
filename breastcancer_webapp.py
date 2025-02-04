# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 19:17:56 2025

@author: Muskan Gupta
"""
"""@author: Muskan Gupta"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/Sarvesh/OneDrive/Desktop/breastcancer/breast_cancer_model.sav','rb'))


def breast_cancer_prediction(input_data):
    
    
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return('The Breast cancer is Malignant')

    else:
        return('The Breast cancer is Banign')
        
        
def main():
    
    st.title('Breast_Cancer_Detection_App')
    
    id = st.text_input('id')
    radius_mean = st.text_input('Radius Mean')
    texture_mean = st.text_input('Texture_Mean')
    perimeter_mean = st.text_input('Perimeter_Mean')
    area_mean = st.text_input('Area_Mean')
    smoothness_mean = st.text_input('Smoothness_Mean')
    compactness_mean = st.text_input('Compactness_Mean')
    concavity_mean = st.text_input('Concavity_Mean')
    concave_points_mean = st.text_input('Concave points_Mean')
    symmetry_mean = st.text_input('Symmetry_Mean')
    fractal_dimension_mean = st.text_input('Fractal_dimension_mean')
    radius_se = st.text_input('Radius_se')
    texture_se = st.text_input('Texture_se')
    perimeter_se = st.text_input('Perimeter_se')
    area_se = st.text_input('Area_se')
    smoothness_se = st.text_input('smoothness_se')
    compactness_se = st.text_input('Compactness_se')
    concavity_se = st.text_input('Concavity_se')
    concave_points_se = st.text_input('Concave_points_se')
    symmetry_se = st.text_input('Symmetry_se')
    fractal_dimension_se = st.text_input('Fractal_Dimension')
    radius_worst = st.text_input('Radius_Worst')
    texture_worst = st.text_input('Texture_worst')
    perimeter_worst = st.text_input('Perimeter_worst')
    area_worst = st.text_input('Area_worst')
    smoothness_worst = st.text_input('Smoothness_worst')
    compactness_worst = st.text_input('Compactness_worst')
    concavity_worst = st.text_input('Concavity_worst')
    concave_points_worst = st.text_input('Concave_point_worst')
    symmetry_worst = st.text_input('Symmetry_worst')
    fractal_dimension_worst = st.text_input('Fractal_Dimension_worst')
    
    
    diagnosis = ''
    
    if st.button('Breast_Cancer_Detection_Test_Result'):
        try:
            
            
            input_data = [float(id),
                                             float(radius_mean),
                                             float(texture_mean),
                                             float(perimeter_mean),
                                             float(area_mean),
                                             float(smoothness_mean),
                                             float(compactness_mean),
                                             float(concavity_mean),
                                             float(concave_points_mean),
                                             float(symmetry_mean),
                                             float(fractal_dimension_mean),
                                             float(radius_se),
                                             float(texture_se),
                                             float(perimeter_se),
                                             float(area_se),
                                             float(smoothness_se),
                                             float(compactness_se),
                                             float(concavity_se),
                                             float(concave_points_se),
                                             float(symmetry_se),
                                             float(fractal_dimension_se),
                                             float(radius_worst),
                                             float(texture_worst),
                                             float(perimeter_worst),
                                             float(area_worst),
                                             float(smoothness_worst),
                                             float(compactness_worst),
                                             float(concavity_worst),
                                             float(concave_points_worst),
                                             float(symmetry_worst),
                                             float(fractal_dimension_worst)]
            diagnosis = breast_cancer_prediction(input_data)
        except ValueError:
            diagnosis = "Please enter valid numeric for all field"
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()

    
    
