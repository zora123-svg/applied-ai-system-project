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
    assert profile["target_energy"] == 0.83
    assert profile["preferred_era"] == "2020s"


class DummyApiClient:
    def search_artist(self, artist_name: str, limit: int = 1):
        normalized = artist_name.lower().strip()
        if normalized in {"r.kelly", "r kelly", "r"}:
            return ["R. Kelly"]
        if normalized == "radiohead":
            return ["Radiohead"]
        if normalized == "prince":
            return ["Prince"]
        return []

    def get_artist_tags(self, artist_name: str, limit: int = 5):
        if artist_name.lower().strip() == "radiohead":
            return ["alternative rock", "art rock"]
        if artist_name.lower().strip() == "prince":
            return ["funk", "r&b", "soul"]
        return []

    def get_similar_artists(self, artist_name: str, limit: int = 10):
        if artist_name.lower().strip() == "prince":
            return ["The Time", "Morris Day"]
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


def test_merge_profile_clamps_tempo_and_loudness_with_field_specific_ranges():
    agent = RecommenderAgent(api_client=None)
    profile = {"target_tempo": 110.0, "target_loudness": -8.0}
    refined = {"target_tempo": 300.0, "target_loudness": -90.0}

    merged = agent.merge_profile(profile, refined)
    assert merged["target_tempo"] == 220.0
    assert merged["target_loudness"] == -60.0


def test_evaluate_results_returns_low_quality_for_empty_recommendations():
    agent = RecommenderAgent(api_client=None)
    result = agent.evaluate_results("I want upbeat music", {}, [])

    assert result["quality_score"] == 1
    assert result["should_refine"] is False
    assert result["coverage_gap"] is True
    assert result["confidence_score"] == 0.0
    assert "No recommendations" in result["feedback"]
    assert result["reliability_verdict"] == "uncertain"


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
    assert "ITERATION 2" in reasoning
    assert "SATISFIED" not in reasoning
    assert agent.last_evaluation["confidence_score"] >= 0.0
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


def test_evaluate_results_forces_refinement_when_artist_style_is_missing():
    agent = RecommenderAgent(api_client=DummyApiClient(), quality_threshold=7)
    profile = {
        "seed_artist": "Prince",
        "favorite_genre": "rock",
        "target_energy": 0.65,
        "target_valence": 0.7,
        "target_danceability": 0.65,
    }
    recommendations = [
        ({"artist": "Neon Echo", "genre": "rock", "energy": 0.66, "valence": 0.71}, 9.2, "reason"),
        ({"artist": "Indigo Parade", "genre": "rock", "energy": 0.64, "valence": 0.69}, 9.0, "reason"),
    ]

    result = agent.evaluate_results("I want something like Prince music", profile, recommendations)
    assert result["quality_score"] >= 7
    assert result["should_refine"] is True
    assert result["refined_profile"]["favorite_genre"] == "pop"
    assert result["refined_profile"]["target_danceability"] > 0.65
    assert "not style-aligned with Prince" in result["feedback"]


def test_evaluate_results_flags_coverage_gap_for_kpop_request():
    agent = RecommenderAgent(api_client=None, quality_threshold=7)
    profile = {
        "favorite_genre": "k-pop",
        "target_energy": 0.8,
        "target_valence": 0.75,
        "target_danceability": 0.85,
    }
    recommendations = [
        ({"artist": "Neon Echo", "genre": "pop", "energy": 0.82, "valence": 0.8}, 9.0, "reason"),
        ({"artist": "Indigo Parade", "genre": "indie pop", "energy": 0.78, "valence": 0.77}, 8.7, "reason"),
    ]

    result = agent.evaluate_results("I want kpop music", profile, recommendations)
    assert result["coverage_gap"] is True
    assert result["confidence_score"] < 0.65
    assert result["feedback"].startswith("Not reasonable:")
    assert "Dataset coverage gap" in result["feedback"]
    assert "Low confidence result" in result["safe_response"]


def test_evaluate_results_does_not_infer_pop_from_kpop_token():
    agent = RecommenderAgent(api_client=None, quality_threshold=7)
    profile = {
        "favorite_genre": "k-pop",
        "target_energy": 0.8,
        "target_valence": 0.75,
        "target_danceability": 0.85,
    }
    recommendations = [
        ({"artist": "Nova Nine", "genre": "k-pop", "energy": 0.83, "valence": 0.79}, 9.1, "reason"),
        ({"artist": "Luna Byte", "genre": "k-pop", "energy": 0.76, "valence": 0.7}, 8.7, "reason"),
    ]

    result = agent.evaluate_results("I want kpop music", profile, recommendations)
    assert result["coverage_gap"] is False
    assert result["reliability_verdict"] == "reliable"


def test_run_stops_unsatisfied_when_not_refining_and_low_confidence():
    class LowConfidenceNoRefineEvaluator:
        def evaluate(self, query, profile, recommendations, quality_threshold):
            return {
                "quality_score": 7,
                "feedback": "insufficient coverage",
                "should_refine": False,
                "refined_profile": {},
                "confidence_score": 0.4,
                "coverage_gap": True,
                "safe_response": "Low confidence",
            }

    agent = RecommenderAgent(api_client=None, max_iterations=3, evaluator=LowConfidenceNoRefineEvaluator())
    _, reasoning = agent.run("I want kpop music", make_songs(), k=1, mode="relevance")
    assert "UNSATISFIED" in reasoning


def test_run_outputs_profile_delta_and_inference_diagnostics():
    agent = RecommenderAgent(api_client=None, max_iterations=1)
    _, reasoning = agent.run("I want upbeat acoustic pop like Neon Echo", make_songs(), k=1, mode="relevance")
    assert "profile_delta" in reasoning
    assert "inference" in reasoning
    assert "seed_artist" in reasoning


def test_relational_heuristic_adjusts_upbeat_jazz_profile():
    agent = RecommenderAgent(api_client=None)
    profile = agent.extract_profile("I want upbeat jazz", make_songs())
    assert profile["favorite_genre"] == "jazz"
    assert profile["target_energy"] < 0.84
    assert profile["target_tempo"] < 120.0


def test_reflection_heuristic_refines_tempo_for_upbeat_when_tempo_is_slow():
    agent = RecommenderAgent(api_client=None, quality_threshold=7)
    profile = {"target_energy": 0.85, "target_valence": 0.7, "target_tempo": 110.0}
    recommendations = [
        ({"energy": 0.84, "valence": 0.7, "tempo_bpm": 92, "genre": "pop", "artist": "A"}, 9.0, "reason"),
        ({"energy": 0.81, "valence": 0.72, "tempo_bpm": 95, "genre": "pop", "artist": "B"}, 8.8, "reason"),
    ]

    result = agent.evaluate_results("I want upbeat music", profile, recommendations)
    assert result["should_refine"] is True
    assert result["refined_profile"]["target_tempo"] > 110.0
    assert "faster tempo" in result["feedback"]


def test_feedback_is_brutally_honest_for_unresolved_artist_and_era_intent():
    agent = RecommenderAgent(api_client=None, quality_threshold=7)
    profile = {
        "favorite_genre": "pop",
        "target_energy": 0.65,
        "target_valence": 0.7,
        "target_danceability": 0.65,
        "target_tempo": 110.0,
        "seed_artist": "",
    }
    recommendations = [
        ({"artist": "Neon Echo", "genre": "pop", "era": "2020s", "energy": 0.8, "valence": 0.82, "tempo_bpm": 120}, 9.0, "reason"),
        ({"artist": "Indigo Parade", "genre": "indie pop", "era": "2020s", "energy": 0.78, "valence": 0.8, "tempo_bpm": 118}, 8.8, "reason"),
    ]

    result = agent.evaluate_results("I want tupac music in 1950's", profile, recommendations)
    assert result["coverage_gap"] is True
    assert result["feedback"].startswith("Not reasonable:")
    assert "Artist intent unresolved" in result["feedback"]
    assert "Era intent unresolved" in result["feedback"]
