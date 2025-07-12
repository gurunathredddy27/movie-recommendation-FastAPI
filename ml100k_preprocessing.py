import pandas as pd

# Paths
ratings_path = 'ml-100k/u.data'
items_path = 'ml-100k/u.item'
genres_path = 'ml-100k/u.genre'
users_path = 'ml-100k/u.user'

# Load Ratings Data (Collaborative Filtering)
ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings_df = pd.read_csv(ratings_path, sep='\t', names=ratings_cols, encoding='latin-1')
ratings_df.to_csv('ratings.csv', index=False)
print('Done: Saved ratings.csv')

# Load Items Data (Content-Based Filtering)
items_cols = [
    'item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
items_df = pd.read_csv(items_path, sep='|', names=items_cols, encoding='latin-1')
genre_cols = items_cols[5:]
items_df['genres'] = items_df[genre_cols].apply(lambda row: ', '.join([genre for genre, val in zip(genre_cols, row) if val == 1]), axis=1)
items_df_simple = items_df[['item_id', 'title', 'release_date', 'IMDb_URL', 'genres']]
items_df_simple.to_csv('movies_metadata.csv', index=False)
print('Done: Saved movies_metadata.csv')

# Load Users Data (optional)
users_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users_df = pd.read_csv(users_path, sep='|', names=users_cols, encoding='latin-1')
users_df.to_csv('users.csv', index=False)
print('Done: Saved users.csv')

print('\\nDone: Preprocessing complete! Now you can run recommendation.py.')
