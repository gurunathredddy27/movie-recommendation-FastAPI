# Movie Recommendation System

This project implements a movie recommendation system using FastAPI and Streamlit.

## Table of Contents

*   [Introduction](#introduction)
*   [Features](#features)
*   [Tech Stack](#tech-stack)
*   [How It Works](#how-it-works)
*   [Project Structure](#project-structure)
*   [Data Source](#data-source)
*   [Model Building and Training](#model-building-and-training)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Feature Engineering](#feature-engineering)
    *   [Model Selection](#model-selection)
    *   [Model Training](#model-training)
    *   [Model Persistence](#model-persistence)
    *   [Model Evaluation](#model-evaluation)
*   [API Endpoints](#api-endpoints)
*   [Streamlit Frontend](#streamlit-frontend)
*   [Setup and Execution](#setup-and-execution)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [Running the FastAPI Backend](#running-the-fastapi-backend)
    *   [Running the Streamlit Frontend](#running-the-streamlit-frontend)
*   [Code Components](#code-components)
    *   [fastapi\_app.py](#fastapi_apppy)
    *   [streamlit\_app.py](#streamlit_apppy)
    *   [recommendation\_system.py](#recommendation_systempy)
    *   [ml100k\_preprocessing.py](#ml100k_preprocessingpy)
*   [Potential Improvements](#potential-improvements)
 

## Introduction

This project implements a movie recommendation system using FastAPI for the backend API and Streamlit for the user interface. It provides movie recommendations based on collaborative filtering and content-based filtering techniques.

## Features

*   Collaborative Filtering (KNN, item-based)
*   Content-Based Filtering (TF-IDF + Cosine Similarity)
*   FastAPI REST API serving recommendations
*   Streamlit interactive frontend for user queries
*   Uses MovieLens 100K dataset
*   Clean, modular, scalable project structure

## Tech Stack

*   Python: Core development
*   Pandas, scikit-learn: Data preprocessing, KNN, TF-IDF
*   FastAPI: Backend REST API
*   Uvicorn: ASGI server for FastAPI
*   Streamlit: Interactive frontend
*   Pickle: Model and matrix serialization

## How It Works

1.  User enters a movie title (partial or full)
2.  Selects Collaborative or Content-Based filtering
3.  Chooses the number of recommendations
4.  Streamlit sends the request to FastAPI
5.  FastAPI processes the request, returns recommendations
6.  Streamlit displays clean recommendations instantly

## Project Structure

```
Movie_Recommendation_System/
├── fastapi_app.py
├── streamlit_app.py
├── recommendation_system.py
├── ml100k_preprocessing.py
├── knn_model.pkl
├── item_user_matrix.pkl
├── tfidf_matrix.pkl
├── cosine_sim.pkl
├── movies_df.pkl
├── README.md
└── ml-100k/
    ├── u.data
    ├── u.item
    └── ...
```

## Data Source

The project uses the MovieLens 100K dataset, which contains 100,000 ratings from 943 users on 1682 movies. The dataset includes user demographics (age, gender, occupation) and movie information (title, release date, genres).

## Model Building and Training

### Data Preprocessing

The `ml100k_preprocessing.py` script performs the following data preprocessing steps:

1.  Loads the ratings data from `ml-100k/u.data`.
2.  Loads the movie data from `ml-100k/u.item`.
3.  Creates a `movies_df.pkl` file containing movie titles, release dates, IMDb URLs, and genres.
4.  Creates an `item_user_matrix.pkl` file representing the user-item interaction matrix for collaborative filtering.
5.  Creates `users.csv` containing user data.

### Feature Engineering

*   **Collaborative Filtering:** The user-item interaction matrix is used directly for collaborative filtering.
*   **Content-Based Filtering:** TF-IDF (Term Frequency-Inverse Document Frequency) is used to extract features from movie titles and genres.

### Model Selection

*   **Collaborative Filtering:** KNN (K-Nearest Neighbors) is used to find similar movies based on user ratings.
*   **Content-Based Filtering:** Cosine similarity is used to find similar movies based on TF-IDF features.

### Model Training

The `ml100k_preprocessing.py` script trains the following models:

*   **KNN Model:** Trained on the user-item interaction matrix.
*   **TF-IDF Vectorizer:** Trained on movie titles and genres.
*   **Cosine Similarity Matrix:** Calculated based on TF-IDF features.

### Model Persistence

The trained models and matrices are serialized using `pickle` and saved to `.pkl` files. While `pickle` is convenient, it's recommended to use a more robust alternative like `joblib` for production environments, especially when dealing with large datasets.

### Model Evaluation

The recommendation system is evaluated based on its ability to provide relevant and personalized movie recommendations. Evaluation metrics include:

*   **Precision:** The proportion of recommended movies that are actually relevant to the user.
*   **Recall:** The proportion of relevant movies that are recommended to the user.
*   **F1-score:** The harmonic mean of precision and recall.

## API Endpoints

The FastAPI backend provides the following API endpoints:

*   `/recommendations/collaborative/{movie_title}/{num_recommendations}`: Returns collaborative filtering recommendations for a given movie title.
*   `/recommendations/content_based/{movie_title}/{num_recommendations}`: Returns content-based filtering recommendations for a given movie title.

Example API Request:

```
GET /recommendations/collaborative/Toy%20Story/10
```

## Streamlit Frontend

The Streamlit frontend provides a user-friendly interface for interacting with the recommendation system. Users can enter a movie title, select a filtering method, and specify the number of recommendations. The frontend then displays the recommended movies in a clean and organized format.

Example Streamlit Usage Scenario:

1.  Enter "Toy Story" in the movie title field.
2.  Select "Collaborative Filtering" from the filtering method dropdown.
3.  Enter "10" in the number of recommendations field.
4.  Click the "Get Recommendations" button.
5.  The frontend displays the top 10 movies similar to "Toy Story" based on collaborative filtering.

## Setup and Execution

### Prerequisites

*   Python 3.7+
*   pip

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd Movie_recommendation_FastAPI
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the FastAPI Backend

1.  Run the FastAPI application:

    ```bash
    uvicorn fastapi_app:app --reload
    ```

    This will start the FastAPI server on `http://127.0.0.1:8000`.

### Running the Streamlit Frontend

1.  Run the Streamlit application:

    ```bash
    streamlit run streamlit_app.py
    ```

    This will open the Streamlit application in your browser.

## Code Components

### fastapi\_app.py

This file contains the FastAPI application code, including the API endpoints for serving movie recommendations.

### streamlit\_app.py

This file contains the Streamlit frontend code, including the user interface for interacting with the recommendation system.

### recommendation\_system.py

This file contains the core recommendation logic, including the collaborative filtering and content-based filtering algorithms.

### ml100k\_preprocessing.py

This file contains the data preprocessing code, including the steps for loading, cleaning, and transforming the MovieLens 100K dataset.

## Potential Improvements

*   Implement more advanced recommendation algorithms, such as matrix factorization or deep learning-based methods.
*   Improve the model evaluation process by using more sophisticated metrics and techniques.
*   Add support for user ratings and feedback to improve the personalization of recommendations.
*   Deploy the application to a cloud platform for scalability and availability.
*   Use a more robust model persistence method like joblib.

 
