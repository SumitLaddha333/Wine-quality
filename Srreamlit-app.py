#Importing Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st

## RED WINE
# Importing red wine quality file
RED=pd.read_csv('winequality-red.csv',sep=";")

# Splitting Target Value(Quality) and Features X--> Input features to Model Y--> Wine Quality Score
X_red=RED.drop(["quality"],axis=1)
Y_red=RED["quality"]

# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
x_train_red = scaler.fit_transform(X_red)

# Train the model with the best hyperparameters on the entire training set
red_model = RandomForestRegressor(n_estimators=500, max_depth=60, random_state=42)
red_model.fit(x_train_red, Y_red)

## WHITE WINE
# Importing white wine quality file
WHITE=pd.read_csv('winequality-white.csv',sep=";")

# Splitting Target Value(Quality) and Features X--> Input features to Model Y--> Wine Quality Score
X_white=WHITE.drop(["quality"],axis=1)
Y_white=WHITE["quality"]

# Scale the features to have zero mean and unit variance
x_train_white = scaler.fit_transform(X_white)

# Train the model with the best hyperparameters on the entire training set
white_model = RandomForestRegressor(n_estimators=200, max_depth=22, random_state=42)
white_model.fit(x_train_white, Y_white)

def predict_quality(model, features):
    prediction = model.predict(features)
    return prediction

# Creating Streamlit UI
st.title('Wine Quality Predictor')

# Option to choose red or white wine
wine_type = st.radio("Select wine type:", ("red", "white"))

if wine_type == "red":
    df = RED
    model = red_model
else:
    df = WHITE
    model = white_model

df = pd.read_csv('winequality-{}.csv'.format(wine_type), sep=';')
X = df[df.keys()[:-1]]  # To exclude the target variable 'quality'

# Add sliders for feature input
features = {}
for i,column in enumerate(X.columns):
    input= st.slider(f"Select value of {column}",
                     float(X[column].min()), 
                     float(X[column].max()), 
                     float(X[column].mean()),
                     0.001, 
                     key=f'slider_{i}')  
    features[column] = (input - np.mean(X[column])) / np.std(X[column]) # Normalizing raw input with same values as used for training data

if st.button('Get Quality'):
    features_df = pd.DataFrame(features,index=['0'])  # Converting features to a DataFrame
    prediction = predict_quality(model, features_df) # Making prediction
    st.write(f'Predicted Wine Quality: {prediction[0]}') # Display prediction result