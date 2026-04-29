import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.api_client import LastFmAPIError, LastFmClient
from src.ai_inference import LLMProfileExtractor
from src.evaluators import EvaluationProvider, HeuristicEvaluator
from src.recommender import recommend_songs
from src.retriever import build_candidate_songs


DEFAULT_PROFILE = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.65,
    "likes_acoustic": False,
    "prefers_instrumental": False,
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

GENRE_KEYWORDS = {
    "rock": "rock",
    "alternative": "rock",
    "alt": "rock",
    "indie rock": "rock",
    "indie": "indie pop",
    "pop": "pop",
    "hip-hop": "hip-hop",
    "hip hop": "hip-hop",
    "electronic": "electronic",
    "edm": "electronic",
    "synthpop": "electronic",
    "lofi": "lofi",
    "lo-fi": "lofi",
    "dream pop": "electronic",
    "jazz": "jazz",
    "folk": "folk",
    "singer-songwriter": "folk",
    "singer songwriter": "folk",
    "classical": "classical",
    "blues": "blues",
    "metal": "metal",
    "country": "country",
    "synthwave": "synthwave",
    "indie pop": "indie pop",
    "kpop": "k-pop",
    "k-pop": "k-pop",
    "korean pop": "k-pop",
    "afrobeats": "afrobeats",
    "afrobeat": "afrobeats",
    "reggaeton": "reggaeton",
    "latin pop": "latin pop",
    "house": "house",
    "techno": "techno",
    "drill": "drill",
    "funk": "funk",
    "soul": "soul",
    "r&b": "r&b",
    "rnb": "r&b",
    "punk": "punk",
    "alternative rock": "rock",
}

MOOD_KEYWORDS = {
    "happy": "happy",
    "joyful": "happy",
    "confident": "confident",
    "relaxed": "chill",
    "chill": "chill",
    "calm": "chill",
    "sad": "sad",
    "nostalgic": "nostalgic",
    "moody": "moody",
    "romantic": "romantic",
    "focused": "focused",
    "intense": "intense",
    "euphoric": "euphoric",
    "angry": "angry",
    "melancholic": "melancholic",
}

ENERGY_KEYWORDS = {
    "upbeat": 0.85,
    "energetic": 0.9,
    "high energy": 0.9,
    "lively": 0.8,
    "intense": 0.9,
    "heavy": 0.9,
    "driving": 0.85,
    "fast": 0.85,
    "quick": 0.8,
    "chill": 0.35,
    "relaxed": 0.35,
    "mellow": 0.4,
    "slow": 0.3,
    "low energy": 0.25,
}

VALENCE_KEYWORDS = {
    "happy": 0.85,
    "joyful": 0.9,
    "bright": 0.85,
    "confident": 0.8,
    "romantic": 0.75,
    "sad": 0.25,
    "melancholic": 0.25,
    "moody": 0.35,
    "angry": 0.2,
    "chill": 0.55,
}

TEMPO_KEYWORDS = {
    "slow": 70.0,
    "laid-back": 72.0,
    "relaxed": 80.0,
    "midtempo": 100.0,
    "medium tempo": 100.0,
    "upbeat": 120.0,
    "fast": 130.0,
    "energetic": 130.0,
    "driving": 135.0,
}

ERA_KEYWORDS = {
    "1960s": "1960s",
    "1970s": "1970s",
    "1980s": "1980s",
    "1990s": "1990s",
    "2000s": "2000s",
    "2010s": "2010s",
    "2020s": "2020s",
    "classic": "classic",
}

FUZZY_ENERGY_RANGES = {
    "upbeat": (0.75, 0.92),
    "energetic": (0.82, 0.95),
    "high energy": (0.85, 0.97),
    "chill": (0.25, 0.45),
    "relaxed": (0.25, 0.42),
    "mellow": (0.30, 0.48),
}

GENRE_RELATIONAL_RULES = {
    "jazz": {"energy_multiplier": 0.8, "tempo_offset": -8.0, "danceability_offset": -0.05},
    "metal": {"energy_multiplier": 1.08, "tempo_offset": 10.0, "danceability_offset": -0.06},
    "folk": {"energy_multiplier": 0.82, "tempo_offset": -12.0, "danceability_offset": -0.08},
    "hip-hop": {"energy_multiplier": 0.95, "tempo_offset": -6.0, "danceability_offset": 0.08},
    "r&b": {"energy_multiplier": 0.86, "tempo_offset": -10.0, "danceability_offset": 0.04},
}


class RecommenderAgent:
    def __init__(
        self,
        api_client: LastFmClient = None,
        max_iterations: int = 3,
        quality_threshold: int = 7,
        min_confidence: float = 0.65,
        evaluator: Optional[EvaluationProvider] = None,
        profile_extractor: Optional[LLMProfileExtractor] = None,
    ):
        self.api_client = api_client
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.min_confidence = min_confidence
        self.log_path = Path("logs") / "agent.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.evaluator = evaluator or HeuristicEvaluator(api_client=self.api_client, logger=self.append_log)
        self.profile_extractor = profile_extractor
        self.last_evaluation: Dict = {}

    def extract_profile(self, query: str, songs: List[Dict]) -> Dict:
        normalized = query.lower()
        profile = DEFAULT_PROFILE.copy()
        profile["seed_artist"] = self._guess_seed_artist(normalized, songs)
        profile["favorite_genre"] = self._infer_genre(normalized, profile["seed_artist"]) or profile["favorite_genre"]
        profile["favorite_mood"] = self._infer_mood(normalized) or profile["favorite_mood"]
        profile["target_energy"] = self._infer_energy(normalized, profile["target_energy"])
        profile["target_valence"] = self._infer_valence(normalized, profile["target_valence"])
        profile["target_tempo"] = self._infer_tempo(normalized, profile["target_tempo"])
        profile["likes_acoustic"] = self._infer_acoustic(normalized, profile["likes_acoustic"])
        profile["avoid_explicit"] = self._infer_avoid_explicit(normalized)
        profile["preferred_era"] = self._infer_era(normalized) or profile["preferred_era"]
        profile["target_danceability"] = self._infer_danceability(normalized, profile["target_danceability"])
        profile = self._apply_relational_heuristics(normalized, profile)
        if self.profile_extractor:
            try:
                profile = self.profile_extractor.infer_profile(query, profile, songs)
            except Exception as exc:
                self.append_log(f"AI PROFILE ERROR: {exc}")
        return profile

    def search_songs(self, profile: Dict, songs: List[Dict], k: int, mode: str = "relevance") -> List[Tuple[Dict, float, str]]:
        try:
            candidates = build_candidate_songs(profile, songs, api_client=self.api_client, min_candidates=max(k * 3, 10))
        except LastFmAPIError as exc:
            self.append_log(f"API ERROR: {exc}")
            candidates = songs

        recommendations = recommend_songs(profile, candidates, k=k, mode=mode)
        return recommendations

    def evaluate_results(self, query: str, profile: Dict, recommendations: List[Tuple[Dict, float, str]]) -> Dict:
        return self.evaluator.evaluate(query, profile, recommendations, quality_threshold=self.quality_threshold)

    def merge_profile(self, profile: Dict, refined: Dict) -> Dict:
        merged = profile.copy()
        bounded_fields = {
            "target_energy": (0.0, 1.0),
            "target_valence": (0.0, 1.0),
            "target_danceability": (0.0, 1.0),
            "target_speechiness": (0.0, 1.0),
            "target_liveness": (0.0, 1.0),
            "target_tempo": (40.0, 220.0),
            "target_loudness": (-60.0, 0.0),
        }
        for key, value in refined.items():
            if isinstance(value, float):
                bounds = bounded_fields.get(key)
                if bounds:
                    value = max(bounds[0], min(bounds[1], value))
            merged[key] = value
        return merged

    def append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{timestamp} {message}\n")

    def run(self, query: str, songs: List[Dict], k: int = 5, mode: str = "relevance") -> Tuple[List[Tuple[Dict, float, str]], str]:
        profile = self.extract_profile(query, songs)
        self.append_log(f'QUERY: "{query}"')

        trace_lines = [f'QUERY: "{query}"']
        recommendations = []

        for iteration in range(1, self.max_iterations + 1):
            recommendations = self.search_songs(profile, songs, k=k, mode=mode)
            evaluation = self.evaluate_results(query, profile, recommendations)

            profile_delta = self._profile_delta_from_default(profile)
            inference_diagnostics = self._build_inference_diagnostics(profile, profile_delta)
            self.append_log(f"ITERATION {iteration} | profile_delta: {profile_delta}")
            self.append_log(f"ITERATION {iteration} | inference: {inference_diagnostics}")
            if recommendations:
                self.append_log(
                    f"ITERATION {iteration} | top_result: {recommendations[0][0]['title']} ({recommendations[0][0]['artist']}) score={recommendations[0][1]:.2f}"
                )
            else:
                self.append_log(f"ITERATION {iteration} | top_result: none")
            self.append_log(
                "ITERATION "
                f"{iteration} | quality: {evaluation['quality_score']}/10 | confidence: {evaluation.get('confidence_score', 0.0):.2f} "
                f"| coverage_gap: {evaluation.get('coverage_gap', False)} | verdict: {evaluation.get('reliability_verdict', 'unknown')} "
                f"| feedback: {evaluation['feedback']}"
            )

            trace_lines.append(f"ITERATION {iteration} | profile_delta: {profile_delta}")
            trace_lines.append(f"ITERATION {iteration} | inference: {inference_diagnostics}")
            trace_lines.append(
                "ITERATION "
                f"{iteration} | quality: {evaluation['quality_score']}/10 | confidence: {evaluation.get('confidence_score', 0.0):.2f} "
                f"| coverage_gap: {evaluation.get('coverage_gap', False)} | verdict: {evaluation.get('reliability_verdict', 'unknown')} "
                f"| feedback: {evaluation['feedback']}"
            )

            self.last_evaluation = evaluation
            is_confident = evaluation.get("confidence_score", 0.0) >= self.min_confidence
            has_gap = evaluation.get("coverage_gap", False)
            if not evaluation["should_refine"] and is_confident and not has_gap:
                trace_lines.append(f"SATISFIED — stopping after {iteration} iteration(s)")
                break
            if not evaluation["should_refine"] and (not is_confident or has_gap):
                trace_lines.append(
                    f"UNSATISFIED — stopping after {iteration} iteration(s); insufficient evidence for a reliable answer."
                )
                break

            profile = self.merge_profile(profile, evaluation["refined_profile"])
            self.append_log(f"REFINED PROFILE: {evaluation['refined_profile']}")
            trace_lines.append(f"REFINED PROFILE: {evaluation['refined_profile']}")

        self.append_log(f"FINAL: {len(recommendations)} candidate(s) returned")
        return recommendations, "\n".join(trace_lines)

    def _infer_genre(self, query: str, seed_artist: str = "") -> str:
        if self.api_client and seed_artist:
            try:
                tags = self.api_client.get_artist_tags(seed_artist, limit=5)
            except LastFmAPIError as exc:
                self.append_log(f"API ERROR: {exc}")
                tags = []

            normalized_tags = [tag.lower().strip() for tag in tags if tag]
            for tag in normalized_tags:
                mapped = self._map_tag_to_genre(tag)
                if mapped:
                    return mapped

        for keyword, genre in GENRE_KEYWORDS.items():
            if keyword in query:
                return genre
        return ""

    def _map_tag_to_genre(self, tag: str) -> str:
        normalized_tag = tag.lower().strip()
        if normalized_tag in GENRE_KEYWORDS:
            return GENRE_KEYWORDS[normalized_tag]
        if "k-pop" in normalized_tag or "kpop" in normalized_tag or "korean pop" in normalized_tag:
            return "k-pop"
        if "rock" in normalized_tag:
            return "rock"
        if "pop" in normalized_tag:
            return "pop"
        if "electronic" in normalized_tag:
            return "electronic"
        if "hip hop" in normalized_tag or "hip-hop" in normalized_tag:
            return "hip-hop"
        if "folk" in normalized_tag:
            return "folk"
        if "r&b" in normalized_tag or "rnb" in normalized_tag or "soul" in normalized_tag:
            return "pop"
        if "funk" in normalized_tag:
            return "pop"
        return ""

    def _infer_mood(self, query: str) -> str:
        for keyword, mood in MOOD_KEYWORDS.items():
            if keyword in query:
                return mood
        return ""

    def _infer_energy(self, query: str, default: float) -> float:
        for keyword, (lower, upper) in FUZZY_ENERGY_RANGES.items():
            if keyword in query:
                # Fuzzy midpoint keeps deterministic behavior while avoiding rigid single-value mapping.
                return round((lower + upper) / 2.0, 2)
        for keyword, energy in ENERGY_KEYWORDS.items():
            if keyword in query:
                return energy
        return default

    def _infer_valence(self, query: str, default: float) -> float:
        for keyword, valence in VALENCE_KEYWORDS.items():
            if keyword in query:
                return valence
        return default

    def _infer_tempo(self, query: str, default: float) -> float:
        for keyword, tempo in TEMPO_KEYWORDS.items():
            if keyword in query:
                return tempo
        return default

    def _infer_danceability(self, query: str, default: float) -> float:
        if self._contains_any(query, ["dance", "groovy", "club", "rhythm"]):
            return 0.8
        if self._contains_any(query, ["slow", "calm", "relaxed"]):
            return 0.45
        return default

    def _infer_acoustic(self, query: str, default: bool) -> bool:
        if self._contains_any(query, ["acoustic", "unplugged", "folky"]):
            return True
        if self._contains_any(query, ["electronic", "synth", "edm", "pop"]):
            return False
        return default

    def _infer_avoid_explicit(self, query: str) -> bool:
        return self._contains_any(query, ["clean", "radio friendly", "no explicit"])

    def _infer_era(self, query: str) -> str:
        for keyword, era in ERA_KEYWORDS.items():
            if keyword in query:
                return era
        return ""

    def _guess_seed_artist(self, query: str, songs: List[Dict]) -> str:
        normalized_query = query.lower()
        for song in songs:
            artist = song.get("artist", "").lower()
            if artist and artist in normalized_query:
                return song["artist"]

        pattern = re.compile(
            r"(?:like|sounds like|sound like|similar to|in the style of|as if|by|from)\s+([a-z0-9 .&']+?)(?:\s+(?:but|and|with|who|for|to|as|from|in|on|at|by|about|or|the|a|an|but|so)\b|$)",
            re.IGNORECASE,
        )
        match = pattern.search(query)
        if match and self.api_client:
            candidate = match.group(1).strip()
            candidate = self._normalize_artist_name(candidate)
            found = self.api_client.search_artist(candidate, limit=1)
            if found:
                return found[0]

        return ""

    def _normalize_artist_name(self, candidate: str) -> str:
        candidate = candidate.strip().lower()
        candidate = re.sub(r"[^a-z0-9 .&']+", " ", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        return candidate

    def _contains_any(self, query: str, keywords: List[str]) -> bool:
        return any(keyword in query for keyword in keywords)

    def _profile_delta_from_default(self, profile: Dict) -> Dict:
        delta = {}
        for key, default_value in DEFAULT_PROFILE.items():
            if profile.get(key) != default_value and key != "seed_artist":
                delta[key] = profile.get(key)
        if profile.get("seed_artist"):
            delta["seed_artist"] = profile.get("seed_artist")
        return delta

    def _build_inference_diagnostics(self, profile: Dict, profile_delta: Dict) -> Dict:
        diagnostics = {
            "used_default_baseline": len(profile_delta) == 0,
            "has_seed_artist": bool(profile.get("seed_artist")),
            "changed_fields_count": len(profile_delta),
        }
        if not profile_delta:
            diagnostics["reason"] = "No strong intent signals detected; profile remains at default baseline."
        return diagnostics

    def _apply_relational_heuristics(self, query: str, profile: Dict) -> Dict:
        updated = profile.copy()
        genre = updated.get("favorite_genre", "").lower().strip()
        rules = GENRE_RELATIONAL_RULES.get(genre)
        if not rules:
            return updated

        if self._contains_any(query, ["upbeat", "energetic", "high energy", "lively", "chill", "relaxed", "mellow"]):
            base_energy = float(updated.get("target_energy", 0.65))
            updated["target_energy"] = max(0.0, min(1.0, round(base_energy * rules["energy_multiplier"], 2)))
        updated["target_tempo"] = max(40.0, min(220.0, round(float(updated.get("target_tempo", 110.0)) + rules["tempo_offset"], 1)))
        updated["target_danceability"] = max(
            0.0, min(1.0, round(float(updated.get("target_danceability", 0.65)) + rules["danceability_offset"], 2))
        )
        return updated
