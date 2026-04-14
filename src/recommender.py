from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    instrumentalness: float
    speechiness: float
    liveness: float
    popularity: int

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    prefers_instrumental: bool
    target_speechiness: float
    target_liveness: float
    min_popularity: int

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Initialize the recommender with a list of songs."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Recommend k songs based on user profile."""
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain why a song was recommended for the user."""
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    import csv

    int_fields = {"id", "popularity"}
    float_fields = {"energy", "tempo_bpm", "valence", "danceability",
                    "acousticness", "instrumentalness", "speechiness", "liveness"}

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in int_fields:
                if field in row:
                    row[field] = int(row[field])
            for field in float_fields:
                if field in row:
                    row[field] = float(row[field])
            songs.append(row)
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Returns (-1.0, reasons) if the song is filtered out by min_popularity.
    Required by recommend_songs() and src/main.py
    """
    # Popularity filter — applied before scoring per the Algorithm Recipe
    if song["popularity"] < user_prefs["min_popularity"]:
        return (-1.0, [f"Filtered: popularity {song['popularity']} < min {user_prefs['min_popularity']}"])

    score = 0.0
    reasons = []

    # Genre match: +2.0
    if song["genre"] == user_prefs["favorite_genre"]:
        score += 2.0
        reasons.append("genre match (+2.0)")

    # Mood match: +1.0
    if song["mood"] == user_prefs["favorite_mood"]:
        score += 1.0
        reasons.append("mood match (+1.0)")

    # Energy proximity: 1.0 - abs(song.energy - user.target_energy) → 0.0–1.0
    energy_pts = round(1.0 - abs(song["energy"] - user_prefs["target_energy"]), 2)
    score += energy_pts
    reasons.append(f"energy proximity (+{energy_pts})")

    # Acoustic fit: +0.5 if song's acoustic character matches user preference (threshold 0.6)
    if (song["acousticness"] >= 0.6) == user_prefs["likes_acoustic"]:
        score += 0.5
        reasons.append("acoustic fit (+0.5)")

    # Instrumental fit: +0.5 if song's instrumental character matches user preference (threshold 0.6)
    if (song["instrumentalness"] >= 0.6) == user_prefs["prefers_instrumental"]:
        score += 0.5
        reasons.append("instrumental fit (+0.5)")

    # Speechiness proximity: 1.0 - abs(song.speechiness - user.target_speechiness) → 0.0–1.0
    speechiness_pts = round(1.0 - abs(song["speechiness"] - user_prefs["target_speechiness"]), 2)
    score += speechiness_pts
    reasons.append(f"speechiness proximity (+{speechiness_pts})")

    # Liveness proximity: 1.0 - abs(song.liveness - user.target_liveness) → 0.0–1.0
    liveness_pts = round(1.0 - abs(song["liveness"] - user_prefs["target_liveness"]), 2)
    score += liveness_pts
    reasons.append(f"liveness proximity (+{liveness_pts})")

    return (round(score, 3), reasons)

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored = [
        (song, score, "; ".join(reasons))
        for song in songs
        for score, reasons in [score_song(user_prefs, song)]
        if score >= 0
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
