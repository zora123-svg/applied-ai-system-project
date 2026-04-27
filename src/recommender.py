from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


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
    instrumentalness: float = 0.0
    speechiness: float = 0.1
    liveness: float = 0.1
    popularity: int = 50
    era: str = "2020s"
    explicit: int = 0
    loudness: float = -8.0


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
    prefers_instrumental: bool = False
    target_speechiness: float = 0.1
    target_liveness: float = 0.1
    target_valence: float = 0.5
    min_popularity: int = 0
    target_danceability: float = 0.5
    target_tempo: float = 100.0
    preferred_era: str = ""
    avoid_explicit: bool = False
    target_loudness: float = -8.0


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5, mode: str = "relevance") -> List[Song]:
        """Recommend k songs based on user profile. mode: 'relevance' or 'discovery'."""
        user_dict = asdict(user)
        scored = []
        for song in self.songs:
            song_dict = asdict(song)
            score, reasons = score_song(user_dict, song_dict)
            if score >= 0:
                scored.append((song_dict, score, "; ".join(reasons)))

        scored = sorted(scored, key=lambda x: x[1], reverse=True)
        if mode == "discovery":
            scored = _apply_discovery_boost(scored)
        scored = _apply_artist_diversity(scored)

        song_lookup = {s.id: s for s in self.songs}
        return [song_lookup[s["id"]] for s, _, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain why a song was recommended for the user."""
        _, reasons = score_song(asdict(user), asdict(song))
        return "; ".join(reasons)


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    import csv

    int_fields = {"id", "popularity", "explicit"}
    float_fields = {
        "energy", "tempo_bpm", "valence", "danceability",
        "acousticness", "instrumentalness", "speechiness", "liveness", "loudness",
    }

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
    Returns (-1.0, reasons) if filtered by min_popularity.
    Maximum possible score: 12.0
    """
    if song["popularity"] < user_prefs["min_popularity"]:
        return (-1.0, [f"Filtered: popularity {song['popularity']} < min {user_prefs['min_popularity']}"])

    score = 0.0
    reasons = []

    # Genre match: +1.0
    if song["genre"] == user_prefs["favorite_genre"]:
        score += 1.0
        reasons.append("genre match (+1.0)")

    # Mood match: +1.0
    if song["mood"] == user_prefs["favorite_mood"]:
        score += 1.0
        reasons.append("mood match (+1.0)")

    # Valence proximity: 0.0–1.0
    valence_pts = round(1.0 - abs(song["valence"] - user_prefs["target_valence"]), 2)
    score += valence_pts
    reasons.append(f"valence proximity (+{valence_pts})")

    # Energy proximity: 0.0–2.0
    energy_pts = round(2.0 * (1.0 - abs(song["energy"] - user_prefs["target_energy"])), 2)
    score += energy_pts
    reasons.append(f"energy proximity (+{energy_pts})")

    # Danceability proximity: 0.0–1.0
    dance_pts = round(1.0 - abs(song["danceability"] - user_prefs["target_danceability"]), 2)
    score += dance_pts
    reasons.append(f"danceability proximity (+{dance_pts})")

    # Tempo proximity: 0.0–1.0 (normalized over 100 BPM window)
    tempo_pts = round(max(0.0, 1.0 - abs(song["tempo_bpm"] - user_prefs["target_tempo"]) / 100.0), 2)
    score += tempo_pts
    reasons.append(f"tempo proximity (+{tempo_pts})")

    # Era match: +0.5 (only scored when user specifies a preferred era)
    if user_prefs["preferred_era"] and song.get("era") == user_prefs["preferred_era"]:
        score += 0.5
        reasons.append("era match (+0.5)")

    # Explicit fit: +0.5 unless user avoids explicit content and song is explicit
    if not (user_prefs["avoid_explicit"] and song.get("explicit", 0) == 1):
        score += 0.5
        reasons.append("explicit fit (+0.5)")

    # Loudness proximity: 0.0–1.0 (normalized over 15 dB window)
    loudness_pts = round(
        max(0.0, 1.0 - abs(song.get("loudness", -8.0) - user_prefs["target_loudness"]) / 15.0), 2
    )
    score += loudness_pts
    reasons.append(f"loudness proximity (+{loudness_pts})")

    # Acoustic fit: +0.5 if song acoustic character matches user preference (threshold 0.6)
    if (song["acousticness"] >= 0.6) == user_prefs["likes_acoustic"]:
        score += 0.5
        reasons.append("acoustic fit (+0.5)")

    # Instrumental fit: +0.5 if song instrumental character matches user preference (threshold 0.6)
    if (song["instrumentalness"] >= 0.6) == user_prefs["prefers_instrumental"]:
        score += 0.5
        reasons.append("instrumental fit (+0.5)")

    # Speechiness proximity: 0.0–1.0
    speechiness_pts = round(1.0 - abs(song["speechiness"] - user_prefs["target_speechiness"]), 2)
    score += speechiness_pts
    reasons.append(f"speechiness proximity (+{speechiness_pts})")

    # Liveness proximity: 0.0–1.0
    liveness_pts = round(1.0 - abs(song["liveness"] - user_prefs["target_liveness"]), 2)
    score += liveness_pts
    reasons.append(f"liveness proximity (+{liveness_pts})")

    return (round(score, 3), reasons)


def _apply_artist_diversity(
    scored: List[Tuple[Dict, float, str]]
) -> List[Tuple[Dict, float, str]]:
    """
    Fairness pass: apply a 20% score penalty to any song whose artist already
    appears higher in the ranking. Prevents one artist from monopolizing the top-K.
    Re-sorts after penalties are applied.
    """
    seen: set = set()
    result = []
    for song, score, explanation in scored:
        artist = song["artist"]
        if artist in seen:
            score = round(score * 0.8, 3)
            explanation += "; artist diversity penalty (x0.8)"
        else:
            seen.add(artist)
        result.append((song, score, explanation))
    return sorted(result, key=lambda x: x[1], reverse=True)


def _apply_discovery_boost(
    scored: List[Tuple[Dict, float, str]], threshold: int = 65
) -> List[Tuple[Dict, float, str]]:
    """
    Discovery mode: boost scores of songs below the popularity threshold by 15%
    to surface hidden gems that relevance mode might bury under popular tracks.
    """
    result = []
    for song, score, explanation in scored:
        if song["popularity"] < threshold:
            score = round(score * 1.15, 3)
            explanation += f"; discovery boost (+15%, popularity {song['popularity']})"
        result.append((song, score, explanation))
    return sorted(result, key=lambda x: x[1], reverse=True)


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5, mode: str = "relevance"
) -> List[Tuple[Dict, float, str]]:
    """
    Functional recommendation interface. Returns top-k (song, score, explanation) tuples.
    mode: "relevance" — sort purely by score (default)
          "discovery" — boost low-popularity songs before sorting
    Artist diversity penalty is always applied after mode-based ranking.
    """
    scored = [
        (song, score, "; ".join(reasons))
        for song in songs
        for score, reasons in [score_song(user_prefs, song)]
        if score >= 0
    ]
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    if mode == "discovery":
        scored = _apply_discovery_boost(scored)
    scored = _apply_artist_diversity(scored)
    return scored[:k]
