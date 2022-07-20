# 1. All the feature values for 1 song + a thumbnail (choose the most popular song). We will play some of this song as an example. Also make sure it has no swear words or sexy terms.
# 2. An index of features (send features to Josh and he'll make them up)
# 3. The sliders for each feature is great. For each run, output the popularity score and the rank of that hypothetical sample in the dataset (e.g. 10th most popular). We can get the student to tinker with the parameters to try find the most popular song.
# 4. Plot of model accuracy.

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
from scipy.special import expit
# from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import classification_report, roc_curve, roc_auc_score
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('tiktok.csv')

def prediction(sentence, saved_model):
    observation = pd.DataFrame({'Message': [sentence]})
    yhat = saved_model.predict(observation)
    return yhat


st.markdown('# Model Building Demo')
st.image('Emanuel.png', width=300)
st.image('datasoc.png', width=300)
model = st.selectbox('What type of model would you like to build?', ('TikTok Song Popularity Model', 'Spam Detector'))
if model == 'TikTok Song Popularity Model':
    st.write('Here is the data we will use to build our model:')
    st.dataframe(df.select_dtypes(include=np.number).head())
    duration = st.slider('duration', min_value=None, max_value=None)
    popularity = st.slider('popularity', min_value=None, max_value=None)
    danceability = st.slider('danceability', min_value=None, max_value=None)
    energy = st.slider('energy', min_value=None, max_value=None)
    liveness = st.slider('Liveness', min_value=None, max_value=None)

    st.write(2*duration + np.log(1 + energy) + liveness + expit(danceability))

else:
    saved_pipeline = joblib.load('spam_detector_full_pipeline.joblib')
    text_input = st.text_input(label='Enter example email...')
    if st.button('PREDICT'):
        with st.spinner('The model is now making a prediction based off your email...'):
            time.sleep(3.5)
        yhat = prediction(text_input, saved_pipeline)
        if yhat == 0:
            st.success('This email is not spam')
        else:
            st.error('Spam message identified!')




# model = st.text_input('Type of model')
# if model:
#     if model == 'tiktok':
#         if st.button('Train Model'):
#             with st.spinner('Wait for it...'):
#                 time.sleep(3)
#             st.success('Done!')
#
#             liveness = st.slider('liveness', 0, 100, 50)
#             tempo = st.slider('tempo', 0, 100, 50)
#             f3 = st.text_input('f3', 0, 100, 50)
#             submit_button = st.button('Make prediction')
#
#             if submit_button:
#                 st.write(np.random.randint(0, 100))
#
#
#             # st.write('No longer looking at the previous shit lol')
#             # st.write(np.random.randint(20))
#     else:
#         saved_pipeline = joblib.load('spam_detector_full_pipeline.joblib')
#         text_input = st.text_input(label='Enter some text')
#         if st.button('PREDICT'):
#             st.write('The model is now making a prediction based off your sentence')
#             yhat = prediction(text_input, saved_pipeline)
#             if yhat == 0:
#                 st.write('Not spam')
#             else:
#                 st.write('spam')

