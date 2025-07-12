# streamlit_app.py

import streamlit as st
import requests

st.title("🎬 Movie Recommendation System (FastAPI Frontend)")

API_URL = "http://127.0.0.1:8000"

# User input
movie_title = st.text_input("Enter a movie title (partial or full):", "")
top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

# Recommendation type selection
recommendation_type = st.selectbox(
    "Select recommendation type:",
    ["Collaborative Filtering", "Content-Based Filtering"]
)

if st.button("Get Recommendations"):
    if movie_title.strip() == "":
        st.warning("Please enter a movie title.")
    else:
        endpoint = "/recommend/collaborative" if recommendation_type == "Collaborative Filtering" else "/recommend/content"
        payload = {
            "title": movie_title,
            "top_n": top_n
        }
        try:
            response = requests.post(f"{API_URL}{endpoint}", json=payload)
            if response.status_code == 200:
                data = response.json()
                st.success(f"Recommendations for '{data['input_title']}':")
                for idx, rec_title in enumerate(data['recommended_titles'], start=1):
                    st.write(f"{idx}. {rec_title}")
            else:
                st.error(f"Error {response.status_code}: {response.json()['detail']}")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Failed to connect to FastAPI. Make sure your FastAPI server is running before using this app.")

st.info("Ensure FastAPI is running:\n```bash\nuvicorn fastapi_app:app --reload\n```")
