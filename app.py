# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:56:34 2020

@author: anind
"""

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
SizeFitModelNames = ['SeamlessTopFit.pkl','SeamlessBottomsFit.pkl','LoungeTopFit.pkl','BraTop.pkl']

SizeFitModel = {}
for i in range(len(SizeFitModelNames)):
    SizeFitModel[i] = pickle.load(open(SizeFitModelNames[i], 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    cols = ['user_age', 'user_weight', 'user_height', 'tummy_type_Curvier',
       'tummy_type_Flatter', 'hip_type_Straighter', 'hip_type_Wider',
       'size_preference_type_Loose', 'size_preference_type_Looser',
       'size_preference_type_Slightly Loose',
       'size_preference_type_Slightly Tight', 'size_preference_type_Tight',
       'size_preference_type_Very Tight']
    
    #pred_map = {0: 'X-Small', 1: 'Small', 2: 'Medium', 3: 'Large', 4: 'X-Large'}
    collection_map = {'Seamless Top Fit':0, 'Seamless Bottoms Fit':1, 'Lounge Top Fit':2, 'Bra Top':3}
    collection_name = request.form.get('collection_name')
    user_age = float(request.form.get('user_age'))
    user_weight = float(request.form.get('user_weight'))
    user_height = float(request.form.get('user_height'))
    tummy_shape = request.form.get('tummy_shape')
    hip_shape = request.form.get('hip_shape')
    size_preference = request.form.get('size_preference')
    
    #tummy_shape='Curvier'
    #hip_shape='Wider'
    #size_preference='Very Tight'
    
    input_params = np.zeros(13)
    
    input_params[0]=user_age #age
    input_params[1]=user_weight #weight
    input_params[2]=user_height #height
    
    if tummy_shape=='Curvier':
        input_params[3]=1
    elif tummy_shape=='Flatter':
        input_params[4]=1
    
    if hip_shape=='Straighter':
        input_params[5]=1
    elif hip_shape=='Wider':
        input_params[6]=1
    
    size_choices = ['Loose','Looser','Slightly Loose', 'Slightly Tight', 'Tight', 'Very Tight']
    for c,value in enumerate(size_choices,7):
        if value==size_preference:
            input_params[c]=1
        
    input_features = pd.DataFrame([input_params],columns=cols)
        
    size_proba = SizeFitModel[collection_map[collection_name]].predict_proba(input_features)
    size_proba_sorted = sorted(zip(size_proba[0], ['X-Small','Small','Medium','Large','X-Large']), reverse=True)
    
    x=[]
    size=[]
    for i in range(len(size_proba_sorted)):
        x.append(size_proba_sorted[i][0])
        size.append(size_proba_sorted[i][1])
        
    return render_template('index.html', prediction_text1='Recommended Size is: {} {}%'.format(size[0],np.round(x[0]*100,2)),
                           prediction_text2='Next Best Recommended Size is: {} {}%'.format(size[1],np.round(x[1]*100,2)))

if __name__ == "__main__":
    app.run(debug=True)