import re
from typing import Dict, List, Optional, Set

from src.api_client import LastFmAPIError, LastFmClient


def _normalize_text(value: str) -> str:
    return value.strip().lower()


def _collect_artist_names(profile: Dict, api_client: Optional[LastFmClient]) -> Set[str]:
    artists = set()
    if profile.get("seed_artist"):
        artists.add(_normalize_text(profile["seed_artist"]))
        if api_client:
            try:
                similar = api_client.get_similar_artists(profile["seed_artist"], limit=10)
                artists.update(_normalize_text(name) for name in similar)
            except LastFmAPIError:
                pass

    if profile.get("favorite_genre") and api_client:
        try:
            tag_artists = api_client.get_top_artists_by_tag(profile["favorite_genre"], limit=10)
            artists.update(_normalize_text(name) for name in tag_artists)
        except LastFmAPIError:
            pass

    if profile.get("favorite_mood") and api_client:
        try:
            mood_artists = api_client.get_top_artists_by_tag(profile["favorite_mood"], limit=5)
            artists.update(_normalize_text(name) for name in mood_artists)
        except LastFmAPIError:
            pass

    return artists


def build_candidate_songs(
    profile: Dict,
    songs: List[Dict],
    api_client: Optional[LastFmClient] = None,
    min_candidates: int = 12,
) -> List[Dict]:
    candidates: List[Dict] = []
    normalized_genre = _normalize_text(profile.get("favorite_genre", ""))
    normalized_mood = _normalize_text(profile.get("favorite_mood", ""))
    artist_names = _collect_artist_names(profile, api_client)

    for song in songs:
        song_genre = _normalize_text(song.get("genre", ""))
        song_mood = _normalize_text(song.get("mood", ""))
        song_artist = _normalize_text(song.get("artist", ""))

        if song_artist in artist_names:
            candidates.append(song)
            continue

        if normalized_genre and song_genre == normalized_genre:
            candidates.append(song)
            continue

        if normalized_mood and song_mood == normalized_mood:
            candidates.append(song)
            continue

        if artist_names and any(part in song_artist for part in artist_names if len(part) > 3):
            candidates.append(song)

    if len(candidates) < min_candidates:
        candidates = list({song["id"]: song for song in candidates + songs}.values())

    return candidates
