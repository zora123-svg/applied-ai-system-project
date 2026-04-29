from src.agent import RecommenderAgent
from src.ai_inference import AnthropicInferenceClient, AIInferenceError, LLMEvaluationProvider, LLMProfileExtractor


def make_songs():
    return [
        {
            "id": 1,
            "title": "Signal Bloom",
            "artist": "Nova Thread",
            "genre": "hyperpop",
            "mood": "euphoric",
            "energy": 0.9,
            "tempo_bpm": 136,
            "valence": 0.78,
            "danceability": 0.85,
            "acousticness": 0.08,
            "instrumentalness": 0.0,
            "speechiness": 0.12,
            "liveness": 0.13,
            "popularity": 61,
            "era": "2020s",
            "explicit": 0,
            "loudness": -5.3,
        }
    ]


class FakeInferenceClient:
    def __init__(self, payload):
        self.payload = payload

    def chat_json(self, system_prompt: str, user_prompt: str):
        _ = system_prompt
        _ = user_prompt
        return self.payload


def test_llm_profile_extractor_accepts_out_of_scope_genre():
    payload = {
        "favorite_genre": "hyperpop",
        "favorite_mood": "euphoric",
        "target_energy": 0.92,
        "target_valence": 0.77,
        "target_danceability": 0.86,
        "likes_acoustic": False,
        "target_tempo": 138,
        "preferred_era": "2020s",
        "seed_artist": "100 gecs",
        "confidence_score": 0.88,
        "out_of_scope_reason": "",
    }
    extractor = LLMProfileExtractor(FakeInferenceClient(payload))
    base = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.65,
        "likes_acoustic": False,
        "target_speechiness": 0.1,
        "target_liveness": 0.15,
        "target_valence": 0.7,
        "min_popularity": 0,
        "target_danceability": 0.65,
        "target_tempo": 110.0,
        "preferred_era": "2020s",
        "avoid_explicit": False,
        "target_loudness": -8.0,
        "seed_artist": "",
    }

    inferred = extractor.infer_profile("I want glitchy hyperpop", base, make_songs())
    assert inferred["favorite_genre"] == "hyperpop"
    assert inferred["target_energy"] == 0.92
    assert inferred["seed_artist"] == "100 gecs"


def test_llm_evaluation_provider_validates_malformed_payload():
    payload = {
        "quality_score": "bad-value",
        "feedback": 123,
        "should_refine": "yes",
        "refined_profile": [],
        "confidence_score": "NaN",
        "evidence_count": -3,
        "coverage_gap": True,
        "safe_response": "",
    }
    provider = LLMEvaluationProvider(FakeInferenceClient(payload))
    recommendations = [({"title": "x", "artist": "y", "genre": "pop"}, 7.5, "reason")]

    result = provider.evaluate("obscure request", {"favorite_genre": "pop"}, recommendations, quality_threshold=7)
    assert result["quality_score"] == 1
    assert result["feedback"] == "Evaluation feedback unavailable."
    assert result["refined_profile"] == {}
    assert result["coverage_gap"] is True
    assert "out of dataset scope" in result["safe_response"]


def test_agent_uses_external_profile_extractor_when_available():
    payload = {
        "favorite_genre": "hyperpop",
        "favorite_mood": "euphoric",
        "target_energy": 0.9,
        "target_valence": 0.75,
        "target_danceability": 0.83,
        "likes_acoustic": False,
        "target_tempo": 134,
        "preferred_era": "2020s",
        "seed_artist": "Frost Circuit",
        "confidence_score": 0.9,
        "out_of_scope_reason": "",
    }
    extractor = LLMProfileExtractor(FakeInferenceClient(payload))
    agent = RecommenderAgent(api_client=None, profile_extractor=extractor)
    profile = agent.extract_profile("Give me obscure cyber hyperpop", make_songs())
    assert profile["favorite_genre"] == "hyperpop"


def test_anthropic_content_parser_reads_json_text_blocks():
    client = AnthropicInferenceClient(api_key="key", model="claude")
    parsed = client._parse_content_blocks([{"type": "text", "text": '{"quality_score": 8, "coverage_gap": false}'}])
    assert parsed["quality_score"] == 8
    assert parsed["coverage_gap"] is False


def test_anthropic_content_parser_raises_on_non_json():
    client = AnthropicInferenceClient(api_key="key", model="claude")
    try:
        client._parse_content_blocks([{"type": "text", "text": "not-json"}])
        assert False, "Expected AIInferenceError"
    except AIInferenceError:
        assert True
