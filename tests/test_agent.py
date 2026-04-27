from pathlib import Path

from src.agent import RecommenderAgent


def make_songs():
    return [
        {
            "id": 1,
            "title": "Neon Echo Song",
            "artist": "Neon Echo",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "tempo_bpm": 110,
            "valence": 0.8,
            "danceability": 0.8,
            "acousticness": 0.2,
            "instrumentalness": 0.0,
            "speechiness": 0.1,
            "liveness": 0.1,
            "popularity": 70,
            "era": "2020s",
            "explicit": 0,
            "loudness": -5.0,
        },
        {
            "id": 2,
            "title": "Quiet Folk",
            "artist": "The Hollow Oaks",
            "genre": "folk",
            "mood": "nostalgic",
            "energy": 0.4,
            "tempo_bpm": 90,
            "valence": 0.6,
            "danceability": 0.5,
            "acousticness": 0.9,
            "instrumentalness": 0.0,
            "speechiness": 0.05,
            "liveness": 0.2,
            "popularity": 60,
            "era": "2010s",
            "explicit": 0,
            "loudness": -10.0,
        },
    ]


def test_extract_profile_infers_keywords_and_seed_artist():
    agent = RecommenderAgent(api_client=None)
    query = "I want upbeat acoustic pop like Neon Echo"
    profile = agent.extract_profile(query, make_songs())

    assert profile["seed_artist"] == "Neon Echo"
    assert profile["favorite_genre"] == "pop"
    assert profile["likes_acoustic"] is True
    assert profile["target_energy"] == 0.85
    assert profile["preferred_era"] == "2020s"


class DummyApiClient:
    def search_artist(self, artist_name: str, limit: int = 1):
        return ["R. Kelly"] if artist_name.lower().strip() in {"r.kelly", "r kelly", "r"} else []

    def get_artist_tags(self, artist_name: str, limit: int = 5):
        if artist_name.lower().strip() == "radiohead":
            return ["alternative rock", "art rock"]
        return []


def test_guess_seed_artist_handles_dotted_artist_names():
    agent = RecommenderAgent(api_client=DummyApiClient())
    query = "I want something like R.Kelly but more upbeat"
    seed_artist = agent._guess_seed_artist(query.lower(), make_songs())

    assert seed_artist == "R. Kelly"


def test_extract_profile_uses_api_tags_for_unknown_artist_genre():
    agent = RecommenderAgent(api_client=DummyApiClient())
    profile = agent.extract_profile("I want something like Radiohead", make_songs())

    assert profile["seed_artist"] == "Radiohead"
    assert profile["favorite_genre"] == "rock"


def test_merge_profile_clamps_target_values():
    agent = RecommenderAgent(api_client=None)
    profile = {"target_energy": 0.9, "target_valence": 0.5}
    refined = {"target_energy": 1.2, "target_valence": -0.2}

    merged = agent.merge_profile(profile, refined)
    assert merged["target_energy"] == 1.0
    assert merged["target_valence"] == 0.0


def test_evaluate_results_returns_low_quality_for_empty_recommendations():
    agent = RecommenderAgent(api_client=None)
    result = agent.evaluate_results("I want upbeat music", {}, [])

    assert result["quality_score"] == 1
    assert result["should_refine"] is False
    assert "No recommendations" in result["feedback"]


def test_evaluate_results_upbeat_query_refines_energy_when_needed():
    agent = RecommenderAgent(api_client=None, quality_threshold=8)
    profile = {"target_energy": 0.5, "target_valence": 0.7}
    recommendations = [
        ({"energy": 0.3, "valence": 0.7}, 8.0, "reason"),
        ({"energy": 0.35, "valence": 0.72}, 7.8, "reason"),
    ]

    result = agent.evaluate_results("I want upbeat music", profile, recommendations)
    assert result["should_refine"] is True
    assert result["refined_profile"]["target_energy"] == 0.65
    assert "Increase target energy" in result["feedback"]


def test_run_stops_when_quality_threshold_is_met():
    agent = RecommenderAgent(api_client=None, max_iterations=2, quality_threshold=1)
    recommendations, reasoning = agent.run("I want relaxed music", make_songs(), k=1, mode="relevance")

    assert len(recommendations) == 1
    assert "SATISFIED" in reasoning
    assert Path("logs") .exists()


def test_evaluate_results_refines_even_when_quality_score_is_high():
    agent = RecommenderAgent(api_client=None, quality_threshold=7)
    profile = {"target_energy": 0.5, "target_valence": 0.7}
    recommendations = [
        ({"energy": 0.3, "valence": 0.7}, 9.0, "reason"),
        ({"energy": 0.35, "valence": 0.72}, 8.8, "reason"),
    ]

    result = agent.evaluate_results("I want upbeat music", profile, recommendations)
    assert result["quality_score"] >= 7
    assert result["should_refine"] is True
    assert result["refined_profile"]["target_energy"] == 0.65
