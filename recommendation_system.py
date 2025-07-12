# recommendation.py

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Collaborative Filtering Preparation
# ---------------------------
ratings_df = pd.read_csv('ratings.csv')

# Create user-item matrix
user_item_matrix = ratings_df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
item_user_matrix = user_item_matrix.T

# Train KNN model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(item_user_matrix)

# Save KNN model and matrix
pickle.dump(model, open('knn_model.pkl', 'wb'))
pickle.dump(item_user_matrix, open('item_user_matrix.pkl', 'wb'))
print('✅ Collaborative filtering model and matrix saved.')

# ---------------------------
# Content-Based Filtering Preparation
# ---------------------------
movies_df = pd.read_csv('movies_metadata.csv')
movies_df['genres'] = movies_df['genres'].fillna('')

# TF-IDF on genres for content-based filtering
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies_df['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save content-based data
pickle.dump(movies_df, open('movies_df.pkl', 'wb'))
pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl', 'wb'))
pickle.dump(cosine_sim, open('cosine_sim.pkl', 'wb'))
print('✅ Content-based filtering matrices saved.')

print('\n✅ recommendation.py: All model files prepared for FastAPI deployment.')