from src.api_client import LastFmAPIError
from src.retriever import build_candidate_songs


class DummyApiClient:
    def get_similar_artists(self, artist_name: str, limit: int = 10):
        return ["Similar Artist", "Related Band"]

    def get_top_artists_by_tag(self, tag: str, limit: int = 10):
        if tag == "rock":
            return ["Rock Star"]
        if tag == "happy":
            return ["Happy Singer"]
        return []


class ErrorApiClient:
    def get_similar_artists(self, artist_name: str, limit: int = 10):
        raise LastFmAPIError("simulated error")

    def get_top_artists_by_tag(self, tag: str, limit: int = 10):
        raise LastFmAPIError("simulated error")


def make_songs():
    return [
        {"id": 1, "title": "Seed Track", "artist": "Seed Artist", "genre": "pop", "mood": "happy"},
        {"id": 2, "title": "Similar Artist Track", "artist": "Similar Artist", "genre": "jazz", "mood": "chill"},
        {"id": 3, "title": "Rock Star Track", "artist": "Rock Star", "genre": "rock", "mood": "sad"},
        {"id": 4, "title": "Happy Singer Track", "artist": "Happy Singer", "genre": "classical", "mood": "happy"},
        {"id": 5, "title": "Other Track", "artist": "Other Artist", "genre": "electronic", "mood": "moody"},
    ]


def test_build_candidate_songs_includes_matching_seed_and_api_artists():
    profile = {
        "seed_artist": "Seed Artist",
        "favorite_genre": "rock",
        "favorite_mood": "happy",
    }
    songs = make_songs()
    candidates = build_candidate_songs(profile, songs, api_client=DummyApiClient(), min_candidates=5)

    candidate_ids = {song["id"] for song in candidates}
    assert {1, 2, 3, 4}.issubset(candidate_ids)
    assert len(candidates) == len(songs)


def test_build_candidate_songs_falls_back_to_full_list_when_insufficient():
    profile = {"seed_artist": "Unknown", "favorite_genre": "classical", "favorite_mood": "romantic"}
    songs = make_songs()
    candidates = build_candidate_songs(profile, songs, api_client=None, min_candidates=10)

    assert len(candidates) == len(songs)
    assert set(song["id"] for song in candidates) == set(song["id"] for song in songs)


def test_build_candidate_songs_handles_api_errors_gracefully():
    profile = {
        "seed_artist": "Seed Artist",
        "favorite_genre": "rock",
        "favorite_mood": "happy",
    }
    songs = make_songs()
    candidates = build_candidate_songs(profile, songs, api_client=ErrorApiClient(), min_candidates=5)

    assert len(candidates) == len(songs)
    assert any(song["artist"] == "Rock Star" for song in candidates)


def test_build_candidate_songs_matches_artist_substrings_longer_than_three_chars():
    profile = {
        "seed_artist": "John Doe",
        "favorite_genre": "", 
        "favorite_mood": "",
    }
    songs = [
        {"id": 1, "title": "John Doe Track", "artist": "John Doe", "genre": "rock", "mood": "happy"},
        {"id": 2, "title": "Doe Tribute", "artist": "John Doe Band", "genre": "pop", "mood": "happy"},
        {"id": 3, "title": "No Match", "artist": "Random Artist", "genre": "jazz", "mood": "sad"},
    ]

    candidates = build_candidate_songs(profile, songs, api_client=None, min_candidates=2)
    candidate_names = [song["artist"] for song in candidates]

    assert "John Doe Band" in candidate_names
    assert "John Doe" in candidate_names
