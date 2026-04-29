import json
import os
from typing import Dict, List, Protocol, Tuple

import requests


class AIInferenceError(Exception):
    pass


class InferenceClient(Protocol):
    def chat_json(self, system_prompt: str, user_prompt: str) -> Dict:
        ...


class OpenAICompatibleInferenceClient:
    def __init__(self, api_key: str, model: str, base_url: str, timeout_seconds: int = 20):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "OpenAICompatibleInferenceClient":
        api_key = os.environ.get("AI_INFERENCE_API_KEY", "").strip()
        model = os.environ.get("AI_INFERENCE_MODEL", "").strip()
        base_url = os.environ.get("AI_INFERENCE_BASE_URL", "").strip()
        timeout_raw = os.environ.get("AI_INFERENCE_TIMEOUT_SECONDS", "20").strip()
        if not api_key or not model or not base_url:
            raise AIInferenceError(
                "Missing AI inference config. Set AI_INFERENCE_API_KEY, AI_INFERENCE_MODEL, and AI_INFERENCE_BASE_URL."
            )
        try:
            timeout_seconds = int(timeout_raw)
        except ValueError as exc:
            raise AIInferenceError("AI_INFERENCE_TIMEOUT_SECONDS must be an integer.") from exc
        return cls(api_key=api_key, model=model, base_url=base_url, timeout_seconds=timeout_seconds)

    def chat_json(self, system_prompt: str, user_prompt: str) -> Dict:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AIInferenceError(f"AI inference request failed: {exc}") from exc

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise AIInferenceError("AI inference returned no choices.")
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise AIInferenceError("AI inference returned empty content.")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise AIInferenceError("AI inference content is not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise AIInferenceError("AI inference JSON response must be an object.")
        return parsed


class AnthropicInferenceClient:
    def __init__(self, api_key: str, model: str, timeout_seconds: int = 20):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.base_url = "https://api.anthropic.com/v1/messages"

    @classmethod
    def from_env(cls) -> "AnthropicInferenceClient":
        api_key = os.environ.get("AI_INFERENCE_API_KEY", "").strip()
        model = os.environ.get("AI_INFERENCE_MODEL", "").strip()
        timeout_raw = os.environ.get("AI_INFERENCE_TIMEOUT_SECONDS", "20").strip()
        if not api_key or not model:
            raise AIInferenceError("Missing Anthropic config. Set AI_INFERENCE_API_KEY and AI_INFERENCE_MODEL.")
        try:
            timeout_seconds = int(timeout_raw)
        except ValueError as exc:
            raise AIInferenceError("AI_INFERENCE_TIMEOUT_SECONDS must be an integer.") from exc
        return cls(api_key=api_key, model=model, timeout_seconds=timeout_seconds)

    def chat_json(self, system_prompt: str, user_prompt: str) -> Dict:
        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "temperature": 0.1,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AIInferenceError(f"Anthropic request failed: {exc}") from exc

        data = response.json()
        content_blocks = data.get("content", [])
        parsed = self._parse_content_blocks(content_blocks)
        if not isinstance(parsed, dict):
            raise AIInferenceError("Anthropic JSON response must be an object.")
        return parsed

    def _parse_content_blocks(self, content_blocks: List[Dict]) -> Dict:
        if not isinstance(content_blocks, list) or not content_blocks:
            raise AIInferenceError("Anthropic response contained no content blocks.")
        text_parts = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_value = block.get("text", "")
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value.strip())
        if not text_parts:
            raise AIInferenceError("Anthropic response did not include text content.")
        combined = "\n".join(text_parts).strip()
        try:
            payload = json.loads(combined)
        except json.JSONDecodeError as exc:
            raise AIInferenceError("Anthropic response text is not valid JSON.") from exc
        return payload


def create_inference_client_from_env() -> InferenceClient:
    provider = os.environ.get("AI_INFERENCE_PROVIDER", "openai_compatible").strip().lower()
    if provider == "anthropic":
        return AnthropicInferenceClient.from_env()
    if provider == "openai_compatible":
        return OpenAICompatibleInferenceClient.from_env()
    raise AIInferenceError("Unsupported AI_INFERENCE_PROVIDER. Use 'openai_compatible' or 'anthropic'.")


class LLMProfileExtractor:
    def __init__(self, client: InferenceClient):
        self.client = client

    def infer_profile(self, query: str, default_profile: Dict, songs: List[Dict]) -> Dict:
        songs_preview = [
            {
                "title": song.get("title", ""),
                "artist": song.get("artist", ""),
                "genre": song.get("genre", ""),
                "energy": song.get("energy", 0.0),
                "valence": song.get("valence", 0.0),
                "danceability": song.get("danceability", 0.0),
                "acousticness": song.get("acousticness", 0.0),
            }
            for song in songs[:20]
        ]
        system_prompt = (
            "You are a music preference extractor. Return ONLY JSON with keys: "
            "favorite_genre, favorite_mood, target_energy, target_valence, target_danceability, likes_acoustic, "
            "target_tempo, preferred_era, seed_artist, confidence_score, out_of_scope_reason. "
            "Never invent songs or artists not in query context. If query is out of scope, set confidence_score low and explain out_of_scope_reason."
        )
        user_prompt = (
            f"Query: {query}\n"
            f"Current default profile: {json.dumps(default_profile)}\n"
            f"Songs preview: {json.dumps(songs_preview)}"
        )
        payload = self.client.chat_json(system_prompt, user_prompt)
        return self._validate_profile_payload(payload, default_profile)

    def _validate_profile_payload(self, payload: Dict, default_profile: Dict) -> Dict:
        result = default_profile.copy()
        genre = payload.get("favorite_genre")
        if isinstance(genre, str) and genre.strip():
            result["favorite_genre"] = genre.strip().lower()

        mood = payload.get("favorite_mood")
        if isinstance(mood, str) and mood.strip():
            result["favorite_mood"] = mood.strip().lower()

        seed_artist = payload.get("seed_artist")
        if isinstance(seed_artist, str):
            result["seed_artist"] = seed_artist.strip()

        era = payload.get("preferred_era")
        if isinstance(era, str) and era.strip():
            result["preferred_era"] = era.strip()

        result["target_energy"] = self._coerce_float(payload.get("target_energy"), result["target_energy"], 0.0, 1.0)
        result["target_valence"] = self._coerce_float(payload.get("target_valence"), result["target_valence"], 0.0, 1.0)
        result["target_danceability"] = self._coerce_float(
            payload.get("target_danceability"), result["target_danceability"], 0.0, 1.0
        )
        result["target_tempo"] = self._coerce_float(payload.get("target_tempo"), result["target_tempo"], 40.0, 220.0)

        likes_acoustic = payload.get("likes_acoustic")
        if isinstance(likes_acoustic, bool):
            result["likes_acoustic"] = likes_acoustic
        return result

    def _coerce_float(self, value, fallback: float, minimum: float, maximum: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return fallback
        return max(minimum, min(maximum, parsed))


class LLMEvaluationProvider:
    def __init__(self, client: InferenceClient):
        self.client = client

    def evaluate(self, query: str, profile: Dict, recommendations: List[Tuple[Dict, float, str]], quality_threshold: int) -> Dict:
        rec_preview = [
            {
                "title": song.get("title", ""),
                "artist": song.get("artist", ""),
                "genre": song.get("genre", ""),
                "score": score,
                "reason": reason,
            }
            for song, score, reason in recommendations[:5]
        ]
        system_prompt = (
            "You are a strict evaluator for a music recommendation agent. Return JSON only with keys: "
            "quality_score, feedback, should_refine, refined_profile, confidence_score, evidence_count, coverage_gap, safe_response. "
            "Set coverage_gap true when query intent is not represented by candidate results."
        )
        user_prompt = (
            f"Query: {query}\nProfile: {json.dumps(profile)}\nTop recommendations: {json.dumps(rec_preview)}\n"
            f"Quality threshold: {quality_threshold}"
        )
        payload = self.client.chat_json(system_prompt, user_prompt)
        return self._validate_evaluation_payload(payload, quality_threshold)

    def _validate_evaluation_payload(self, payload: Dict, quality_threshold: int) -> Dict:
        quality_score = self._coerce_int(payload.get("quality_score"), 1, 10, default=1)
        confidence_score = self._coerce_float(payload.get("confidence_score"), 0.0, 0.0, 1.0)
        evidence_count = self._coerce_int(payload.get("evidence_count"), 0, 100, default=0)
        coverage_gap = bool(payload.get("coverage_gap", False))
        should_refine = bool(payload.get("should_refine", quality_score < quality_threshold))

        refined_profile_raw = payload.get("refined_profile", {})
        refined_profile = refined_profile_raw if isinstance(refined_profile_raw, dict) else {}
        feedback = payload.get("feedback", "")
        safe_response = payload.get("safe_response", "")
        if not isinstance(feedback, str):
            feedback = "Evaluation feedback unavailable."
        if not isinstance(safe_response, str):
            safe_response = ""

        if coverage_gap and not safe_response:
            safe_response = "Low confidence result: request appears out of dataset scope."

        return {
            "quality_score": quality_score,
            "feedback": feedback.strip() or "Evaluation feedback unavailable.",
            "should_refine": should_refine,
            "refined_profile": refined_profile,
            "confidence_score": confidence_score,
            "evidence_count": evidence_count,
            "coverage_gap": coverage_gap,
            "safe_response": safe_response,
        }

    def _coerce_float(self, value, default: float, minimum: float, maximum: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, parsed))

    def _coerce_int(self, value, minimum: int, maximum: int, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, parsed))
