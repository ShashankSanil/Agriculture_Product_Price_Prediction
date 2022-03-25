import streamlit as st
import pickle
import numpy as np
import pandas as pd

# import the model
pipe = pickle.load(open('LinearRegressionModel1.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("==============================")
st.title("Welcome to Prediction World!!!")
st.title("==============================")

st.title("Agriculture Product Price Predictor")


# State
st.header('State:')
state = st.selectbox('Select state',df['state'].unique())

# District
st.header('District:')
district = st.selectbox('Select District',df['district'].loc[df['state']==state].unique())

# Commodity
st.header('Commodity:')
commodity = st.selectbox('Select Commodity',df['commodity'].unique())

# Variety
st.header('Variety:')
variety = st.selectbox('Select Commodity',df['variety'].loc[df['commodity']==commodity].unique())

if st.button('Predict_Price'):
    if state == '"--Select State--"' or district == '"--Select District--"' or commodity == '"--Select Commodity--"' or variety == '"--Select Variety--"':
        st.error('Please Enter Valid Input')
    else:
        # Querry
        prediction = pipe.predict(pd.DataFrame(columns=['state', 'district', 'commodity', 'variety'],
                                               data=np.array([state, district, commodity, variety]).reshape(1, 4)))
        # printing prediction

        st.title("₹" + str(np.round(prediction[0], 2)))




