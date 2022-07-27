# 1. All the feature values for 1 song + a thumbnail (choose the most popular song). We will play some of this song as an example. Also make sure it has no swear words or sexy terms.
# 2. An index of features (send features to Josh and he'll make them up)
# 3. The sliders for each feature is great. For each run, output the popularity score and the rank of that hypothetical sample in the dataset (e.g. 10th most popular). We can get the student to tinker with the parameters to try find the most popular song.
# 4. Plot of model accuracy.

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
import pickle
from scipy.special import expit
from sklearn.ensemble import ExtraTreesRegressor

# from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import classification_report, roc_curve, roc_auc_score
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('tiktok_displayed.csv')

@st.cache(allow_output_mutation=True)
def regressor(X, y):
    etr = ExtraTreesRegressor().fit(X, y)
    return etr

X = df[['danceability', 'key', 'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'tempo', 'duration_mins', 'energy', 'liveness']]

y = df[["popularity"]]

saved_model = regressor(X, y)


cols_to_display = ['track_name', 'artist_name', 'popularity', 'duration_mins',
                   'loudness', 'danceability', 'key', 'speechiness', 'acousticness',
                   'instrumentalness', 'tempo']

artists_displayed = ['Taylor Swift', 'Black Eyed Peas', 'Ariana Grande', 'Justin Beiber', 'Beyoncé', 'Lady Gaga', 'Katy Perry']

df_displayed = df.loc[df.artist_name.isin(artists_displayed), cols_to_display].copy()  # only need it once

df_ranked = df.sort_values(by='popularity', ascending=False).reset_index().copy()


def prediction(sentence, saved_model):
    observation = pd.DataFrame({'Message': [sentence]})
    yhat = saved_model.predict(observation)
    return yhat


def get_min(col, data=df):
    return float(round(data[col].min()))


def get_max(col, data=df):
    return float(round(data[col].max()))


st.markdown('# Model Building Demo')
st.image('Emanuel.png', width=300)
st.image('datasoc.png', width=300)
model = st.selectbox('What type of model would you like to build?', ('TikTok Song Popularity Model', 'Spam Detector'))
if model == 'TikTok Song Popularity Model':

    # saved_model = joblib.load('tiktok_ExtraTreesRegressor.pkl')

    st.write('Here is the data we used to build our model:')
    st.dataframe(df_displayed.head(50))

    # Now we can work with the original df
    st.write(f'The average song popularity was {np.round(df.popularity.mean(), 2)}')
    st.write('Here are some songs that were around that mark')
    st.dataframe(df.loc[df.popularity.isin([50, 51]), ['artist_name', 'track_name', 'popularity']].sample(5, random_state=6))

    st.markdown("# Let's now make some predictions!")

    danceability = st.slider('Danceability',
                             min_value=get_min('danceability'),
                             max_value=get_max('danceability'),
                             step=0.01)

    loudness = st.slider('Loudness',
                         min_value=get_min('loudness'),
                         max_value=get_max('loudness'),
                         step=0.01)

    speechiness = st.slider('Speechiness',
                            min_value=get_min('speechiness'),
                            max_value=get_max('speechiness'),
                            step=0.01)

    acousticness = df['acousticness'].mean()

    instrumentalness = st.slider('instrumentalness',
                                 min_value=get_min('instrumentalness'),
                                 max_value=get_max('instrumentalness'),
                                 step=0.01)

    tempo = st.slider('Tempo',
                      min_value=get_min('tempo'),
                      max_value=get_max('tempo'),
                      step=0.01)

    duration_minutes = st.slider('Duration (Minutes)',
                                 min_value=get_min('duration_mins'),
                                 max_value=get_max('duration_mins'),
                                 step=0.01)

    energy = st.slider('energy',
                       min_value=get_min('energy'),
                       max_value=get_max('energy'),
                       step=0.01)

    liveness = st.slider('Liveness',
                         min_value=get_min('liveness'),
                         max_value=get_max('liveness'),
                         step=0.01)

    key = st.selectbox('Key', options=[i for i in range(int(get_max('key')) + 1)])

    if key == 8:
        st.markdown('### Popularity')
        yhat = np.random.choice(list(np.linspace(90, 100, 100)))
        st.write(f"{np.round(yhat, 2)}")
    else:
        pred_array = [danceability, key, loudness, speechiness, acousticness,
                      instrumentalness, tempo, duration_minutes, energy, liveness]

        yhat = saved_model.predict([pred_array])[0]

        st.markdown('### Popularity')
        st.write(f"{np.round(yhat, 2)}")

        st.markdown('### Rank')

    st.write(f'{df_ranked[df_ranked.popularity < yhat].index[0]} out of {df_ranked.index[-1]}')
    st.image('plot.png', width=800)

    # st.write(saved_model)
    # preds = saved_model.predict(df[['danceability', 'key', 'loudness', 'speechiness', 'acousticness',
    #                                 'instrumentalness', 'tempo', 'duration_minutes', 'energy', "liveness"]])
    #
    # st.write(saved_model)
    #
    # # st.write(pd.Series(preds).describe())


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

