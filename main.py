from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import difflib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
songs_data = pd.read_csv("songs.csv")

if "index" not in songs_data.columns:
    songs_data.reset_index(inplace=True)

selected_features = ["artist", "song", "text"]

for feature in selected_features:
    songs_data[feature] = songs_data[feature].fillna("")

combined_features = (
    songs_data["artist"] + " " +
    songs_data["song"] + " " +
    songs_data["text"]
)

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)


similarity = cosine_similarity(feature_vectors)

list_of_all_titles = songs_data["title"].tolist()


class SongRequest(BaseModel):
    song: str

@app.get("/")
def root():
    return {"message": "Song Recommendation API"}

@app.post("/recommendation")
def recommend(request: SongRequest):
    song_name = request.song

    close_matches = difflib.get_close_matches(song_name, list_of_all_titles)
    
    if not close_matches:
        return {"error": "Song not found"}

    close_match = close_matches[0]

    index_of_song = songs_data[songs_data.title == close_match]["index"].values[0]

    similarity_scores = list(enumerate(similarity[index_of_song]))

    sorted_similar_songs = sorted(
        similarity_scores, key=lambda x: x[1], reverse=True
    )

    recommendations = []

    for song in sorted_similar_songs[1:21]:
        index = song[0]
        title = songs_data[songs_data.index == index]["song"].values[0]
        recommendations.append(title)

    return {
        "matched_song": close_match,
        "recommendations": recommendations
    }
