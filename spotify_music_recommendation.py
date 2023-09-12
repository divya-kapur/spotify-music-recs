#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:14:13 2021

@author: divyakapur
"""

#!pip install spotipy

import spotipy
import numpy as np
import pandas as pd
import seaborn as sns
import spotipy.util as util
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.layers import BatchNormalization
from keras.callbacks import LearningRateScheduler


username = '1241819642'
scope = 'user-library-read user-top-read user-read-recently-played'
token = util.prompt_for_user_token(username,scope,client_id='66c32e0c40454f9ca545df8d023e5c1a', client_secret='b25e3fb90ef049a3bf153c6c66fbe498',redirect_uri='http://localhost:8888/callback')

spotify = spotipy.Spotify(auth=token)

# Custom function that allows us to retrieve audio features of more than 100 tracks
def get_audio_features(track_ids):
    init_index = 0
    next_index = 100
    results = pd.DataFrame(spotify.audio_features(track_ids[init_index:next_index]))
    while len(track_ids) > next_index:
        init_index = init_index + 100
        next_index = next_index + 100
        results2 = pd.DataFrame(spotify.audio_features(track_ids[init_index:next_index]))
        results = results.append(results2).reset_index(drop=True)
    return results

# Custom function that allows us to read more than 100 songs from playlist
def get_playlist_tracks(user_id,playlist_id):
    results = spotify.user_playlist_tracks(user_id,playlist_id)
    while results['next']:
        results = spotify.next(results)
        results['items'].extend(results['items'])
    return results

# Generate 'liked' (1) songs 

train = pd.DataFrame(columns=['track_name', 'artist', 'track_id'])

saved_tracks = spotify.current_user_saved_tracks(limit=50)
for item in saved_tracks['items']:
    saved_track = item['track']
    #print(saved_track['name'] + ' - ' + saved_track['artists'][0]['name'])
    train = train.append({'track_name':saved_track['name'], 
                          'artist':saved_track['artists'][0]['name'], 
                          'track_id':saved_track['id']}, ignore_index=True)
train
    
recently_played_tracks = spotify.current_user_recently_played(limit=50)
for item in recently_played_tracks['items']:
    recently_played_track = item['track']
    #print(recently_played_track['name'] + ' - ' + recently_played_track['artists'][0]['name'])
    train = train.append({'track_name':recently_played_track['name'], 
                          'artist':recently_played_track['artists'][0]['name'], 
                          'track_id':recently_played_track['id']}, ignore_index=True)
train

top_tracks = spotify.current_user_top_tracks(limit=50)
for item in top_tracks['items']:
    #print(item['name'] + ' - ' + item['artists'][0]['name'])
    train = train.append({'track_name':item['name'], 
                          'artist':item['artists'][0]['name'], 
                          'track_id':item['id']}, ignore_index=True)
train

# More playlists

liked_playlists_df = pd.DataFrame(columns=['playlist_name','playlist_id'])

liked_playlists = spotify.user_playlists(username)

for playlist in liked_playlists['items']:
    liked_playlists_df = liked_playlists_df.append({'playlist_name':playlist['name'], 
                          'playlist_id':playlist['id']}, ignore_index=True)
liked_playlists_df

for i,row in liked_playlists_df.iterrows():
    liked_playlist = get_playlist_tracks(username, row['playlist_id'])
    for item in liked_playlist['items']:
        liked_track = item['track']
        train = train.append({'track_name':liked_track['name'], 
                              'artist':liked_track['artists'][0]['name'], 
                              'track_id':liked_track['id']}, ignore_index=True)
        
train

train = train.drop_duplicates('track_id', keep='last').reset_index(drop=True)
train

train['liked'] = 1

train1 = train
train1

# Generate 'disliked' (0) songs

train2 = pd.DataFrame(columns=['track_name', 'artist', 'track_id'])

spotify_user_id = 'wizzler'
disliked_playlist1_id = '37i9dQZF1DX1lVhptIYRda'
disliked_playlist2_id = '37i9dQZF1DXcF6B6QPhFDv'
disliked_playlist3_id = '37i9dQZF1DX6xZZEgC9Ubl'
disliked_playlist4_id = '37i9dQZF1DX5M59nhwFlWl'

disliked_playlist1 = get_playlist_tracks(spotify_user_id, disliked_playlist1_id)
disliked_playlist1

for item in disliked_playlist1['items']:
    disliked_track = item['track']
    train2 = train2.append({'track_name':disliked_track['name'], 
                          'artist':disliked_track['artists'][0]['name'], 
                          'track_id':disliked_track['id']}, ignore_index=True)
train2

disliked_playlist2 = get_playlist_tracks(spotify_user_id, disliked_playlist2_id)
disliked_playlist2

for item in disliked_playlist2['items']:
    disliked_track = item['track']
    train2 = train2.append({'track_name':disliked_track['name'], 
                          'artist':disliked_track['artists'][0]['name'], 
                          'track_id':disliked_track['id']}, ignore_index=True)
train2

disliked_playlist3 = get_playlist_tracks(spotify_user_id, disliked_playlist3_id)
disliked_playlist3

for item in disliked_playlist3['items']:
    disliked_track = item['track']
    train2 = train2.append({'track_name':disliked_track['name'], 
                          'artist':disliked_track['artists'][0]['name'], 
                          'track_id':disliked_track['id']}, ignore_index=True)
train2

disliked_playlist4 = get_playlist_tracks(spotify_user_id, disliked_playlist4_id)
disliked_playlist4

for item in disliked_playlist4['items']:
    disliked_track = item['track']
    train2 = train2.append({'track_name':disliked_track['name'], 
                          'artist':disliked_track['artists'][0]['name'], 
                          'track_id':disliked_track['id']}, ignore_index=True)
train2

train2 = train2.drop_duplicates('track_id', keep='last').reset_index(drop=True)

train2['liked'] = 0

# Compiling liked and disliked tracks into one big dataframe

train = train.append(train2).reset_index(drop=True)
train


# Adding audio features to train our model

features = get_audio_features(train['track_id'])

features

features.drop(['type', 'id', 'uri', 'track_href', 'analysis_url'], axis=1, inplace=True)

train = train.join(features)

pd.set_option('max_columns', None)
train.head()
# Visualizing the data

# Remove these columns for visualization
#features.drop(['tempo', 'loudness', 'key', 'duration_ms', 'time_signature'], axis=1, inplace=True)
features_labels = list(features)[:]
features_labels
train_liked = train[train['liked'] == 1]
train_disliked = train[train['liked'] == 0]
liked_features_list = train_liked[features_labels].mean().tolist()
disliked_features_list = train_disliked[features_labels].mean().tolist()

angles = np.linspace(0, 2*np.pi, len(features_labels), endpoint=False)
fig = plt.figure(figsize = (18,18))

ax = fig.add_subplot(221, polar=True)
ax.plot(angles, liked_features_list, 'o-', linewidth=2, label = "Liked", color= 'blue')
ax.fill(angles, liked_features_list, alpha=0.25, facecolor='blue')
ax.set_thetagrids(angles * 180/np.pi, features_labels, fontsize = 13)


ax.set_rlabel_position(250)

ax.plot(angles, disliked_features_list, 'o-', linewidth=2, label = "Not Liked", color= 'orange')
ax.fill(angles, disliked_features_list, alpha=0.25, facecolor='orange')
ax.set_title('Mean Values')
ax.grid(True)

plt.legend(loc='best', bbox_to_anchor=(0.1, 0.1))

#pd.set_option('max_columns', None)

features_labels += ['liked']
sns.pairplot(train[features_labels], hue = 'liked')


# Generating tracks from large playlist 

track_bank = pd.DataFrame(columns=['track_name', 'artist', 'track_id', 'liked_predictions'])

large_playlist = get_playlist_tracks('b3ea802488ad4a3d','2sRZldX6n9oaII70OoO3zB')
large_playlist

for item in large_playlist['items']:
    playlist_track = item['track']
    track_bank = track_bank.append({'track_name':playlist_track['name'], 
                          'artist':playlist_track['artists'][0]['name'], 
                          'track_id':playlist_track['id']}, ignore_index=True)

playlist_features = get_audio_features(track_bank['track_id'])

playlist_features

playlist_features.drop(['type', 'id', 'uri', 'track_href', 'analysis_url'], axis=1, inplace=True)

track_bank = track_bank.join(playlist_features)
track_bank

# Building the model
x_labels = list(playlist_features)[:]
x_train = train[x_labels].values
y_train = train[['liked']].values

y_train
x_test_bank = track_bank[x_labels].values

model = keras.models.Sequential([
            keras.layers.Dense(10, activation="leaky_relu",name="Hidden1"),
            keras.layers.Dense(10, activation="leaky_relu",name="Hidden2"),
            keras.layers.Dense(20, activation="leaky_relu",name="Hidden3"),
            BatchNormalization(),
            keras.layers.Dense(2, activation="sigmoid")
            ])

model.compile(loss="mean_squared_error",
	optimizer='adam',
	metrics=["accuracy"])

# Learning rate scheduler
initial_learning_rate = 0.01
epochs = 5
decay = initial_learning_rate / epochs

def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)


model.fit(x_train,y_train, epochs = 30, callbacks = [LearningRateScheduler(lr_time_based_decay, verbose=1)])


x_test = train.sample(frac=0.2)
x_test_final = x_test[x_labels].values
y_test_final = x_test[['liked']].values

model.evaluate(x_test_final, y_test_final)

predictions = model.predict(x_test_bank)

track_bank['liked_predictions'] = pd.Series
for i in range(len(predictions)):
    if (predictions[i] >= 0.5):
        track_bank['liked_predictions'][i] = 1 
    else:
        track_bank['liked_predictions'][i] = 0


liked_predictions = shuffle(track_bank[track_bank['liked_predictions'] == 1])

new_liked_songs = liked_predictions[['track_name', 'artist', 'track_id']].merge(train1['track_id'].drop_duplicates(), on=['track_id'], how='left')
new_liked_songs = new_liked_songs[['track_name', 'artist']]

new_liked_songs.head()
