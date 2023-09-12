# spotify-music-recs

This project focuses on creating a personalized music recommendation system using Spotify's API. The primary objective is to harness the capabilities of deep learning neural networks to provide users with music recommendations tailored to their listening history and preferences.

**Summary:** This project uses Spotify's API and the spotipy wrapper to collect user data, including liked and disliked tracks. The collected data is then used to generate audio features for each track. These features include danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, and time_signature. These features are used to in a deep neural network to build a classification model. The output of the model is a list of recommended songs that are carefully cross-checked to ensure they are not already familiar to the user. This project is a bridge between data science and web development, showcasing the potential of deep learning neural networks in personalizing music recommendations.

**Input Data**

The input data for this project is derived from various sources:

- Liked Tracks: These are obtained from a user's recently played tracks, top played tracks, and created playlists.

- Disliked Tracks: To counter the lack of a 'dislike' feature on Spotify, disliked tracks are selected from publicly available playlists.

- Audio Features: Audio features essential for model training are provided by the Spotify API.

![Picture1](https://github.com/divya-kapur/spotify-music-recs/assets/47482776/c140f22a-86cd-4f69-b078-06c530ac0f7a)

This chart illustrates mean values of liked vs. disliked songs for various audio features.

![Picture2](https://github.com/divya-kapur/spotify-music-recs/assets/47482776/4376ca6d-293e-4a47-82f0-45801bb90287)

The seaborn pairs plot highlights similarities and differences between liked and disliked songs in their audio feature distributions.

**Methods/Results**

The core of this project relies on a multi-layer Perceptron deep neural network. To optimize model performance, extensive hyperparameter tuning was performed. Key model configurations include:

- Activation functions: leaky_relu & sigmoid
- Optimizer: adam
- Three hidden layers with varying numbers of neurons
- Batch normalization
- 30 epochs with a time-based decay learning rate scheduler
- Initial alpha = 0.01

The final model achieves an accuracy rate of approximately 73%.

**Limitations**

- Limited dataset size, comprising approximately 1000 songs.
- Spotify API limitations on data retrieval.
- Absence of a 'dislike' feature on Spotify.

**Future Directions**

- Implement mood-based recommendations using word vectorization and audio features
- Explore trends in audio features across different music genres



