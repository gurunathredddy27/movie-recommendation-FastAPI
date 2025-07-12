# fastapi_app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = FastAPI(title="Movie Recommendation API (Title Based)")

# Load models and data
model = pickle.load(open('knn_model.pkl', 'rb'))
item_user_matrix = pickle.load(open('item_user_matrix.pkl', 'rb'))
movies_df = pickle.load(open('movies_df.pkl', 'rb'))
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))

class TitleRequest(BaseModel):
    title: str
    top_n: int = 5

@app.get("/")
def read_root():
    return {"message": "🎬 Movie Recommendation API is running 🚀"}

@app.post("/recommend/collaborative")
def recommend_collaborative(request: TitleRequest):
    title = request.title.strip().lower()
    top_n = request.top_n

    # Find movie by title
    matches = movies_df[movies_df['title'].str.lower().str.contains(title)]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Movie title not found.")

    item_id = matches.iloc[0]['item_id']

    # Collaborative filtering using KNN
    if item_id not in item_user_matrix.index:
        raise HTTPException(status_code=404, detail="Item ID not found in collaborative model.")
    item_vector = item_user_matrix.loc[item_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(item_vector, n_neighbors=top_n + 1)
    recommended_ids = item_user_matrix.iloc[indices[0][1:]].index.tolist()

    recommended_titles = movies_df[movies_df['item_id'].isin(recommended_ids)]['title'].tolist()

    return {
        "input_title": matches.iloc[0]['title'],
        "recommended_titles": recommended_titles
    }

@app.post("/recommend/content")
def recommend_content(request: TitleRequest):
    title = request.title.strip().lower()
    top_n = request.top_n

    # Find movie by title
    matches = movies_df[movies_df['title'].str.lower().str.contains(title)]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Movie title not found.")

    idx = matches.index[0]

    # Content-based filtering using cosine similarity
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    recommended_titles = movies_df.iloc[movie_indices]['title'].tolist()

    return {
        "input_title": matches.iloc[0]['title'],
        "recommended_titles": recommended_titles
    }

print("✅ FastAPI app ready. Run using: uvicorn fastapi_app:app --reload")
