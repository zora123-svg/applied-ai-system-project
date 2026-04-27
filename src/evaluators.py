from typing import Callable, Dict, List, Optional, Protocol, Tuple

from src.api_client import LastFmAPIError, LastFmClient


class EvaluationProvider(Protocol):
    def evaluate(
        self, query: str, profile: Dict, recommendations: List[Tuple[Dict, float, str]], quality_threshold: int
    ) -> Dict:
        ...


class HeuristicEvaluator:
    def __init__(self, api_client: Optional[LastFmClient] = None, logger: Optional[Callable[[str], None]] = None):
        self.api_client = api_client
        self.logger = logger

    def evaluate(
        self, query: str, profile: Dict, recommendations: List[Tuple[Dict, float, str]], quality_threshold: int
    ) -> Dict:
        if not recommendations:
            return {
                "quality_score": 1,
                "feedback": "No recommendations were returned.",
                "should_refine": False,
                "refined_profile": {},
            }

        scores = [score for _, score, _ in recommendations]
        avg_score = sum(scores) / len(scores)
        quality_score = min(10, max(1, int(round(avg_score / 12.0 * 10.0))))

        energies = [song["energy"] for song, _, _ in recommendations]
        valences = [song["valence"] for song, _, _ in recommendations]
        avg_energy = sum(energies) / len(energies)
        avg_valence = sum(valences) / len(valences)

        refined_profile: Dict = {}
        feedback = []
        normalized = query.lower()

        if self._contains_any(normalized, ["upbeat", "energetic", "high energy", "lively"]):
            if avg_energy < profile["target_energy"] + 0.1:
                refined_profile["target_energy"] = min(1.0, profile["target_energy"] + 0.15)
                feedback.append("Increase target energy to better match upbeat requests.")
        if self._contains_any(normalized, ["chill", "relaxed", "mellow"]):
            if avg_energy > profile["target_energy"] - 0.1:
                refined_profile["target_energy"] = max(0.0, profile["target_energy"] - 0.15)
                feedback.append("Lower target energy to match chill requests.")
        if self._contains_any(normalized, ["happy", "joyful", "bright"]):
            if avg_valence < profile["target_valence"] + 0.1:
                refined_profile["target_valence"] = min(1.0, profile["target_valence"] + 0.15)
                feedback.append("Raise target valence for happier results.")
        if self._contains_any(normalized, ["sad", "moody", "melancholic"]):
            if avg_valence > profile["target_valence"] - 0.1:
                refined_profile["target_valence"] = max(0.0, profile["target_valence"] - 0.15)
                feedback.append("Lower target valence for sad or moody requests.")

        if quality_score < quality_threshold and not refined_profile and avg_score < 6.0:
            refined_profile["target_energy"] = min(1.0, profile["target_energy"] + 0.1)
            feedback.append("Refining energy target to improve recommendation quality.")

        artist_reference = profile.get("seed_artist", "")
        if self._is_artist_style_query(normalized, artist_reference):
            style_targets = self._get_artist_style_targets(artist_reference)
            if not self._has_artist_style_alignment(recommendations, artist_reference, style_targets):
                artist_genre = style_targets.get("genre", "")
                if artist_genre and artist_genre != profile.get("favorite_genre", ""):
                    refined_profile["favorite_genre"] = artist_genre
                refined_profile["target_danceability"] = min(1.0, profile.get("target_danceability", 0.65) + 0.08)
                feedback.append(
                    f"Top results are not style-aligned with {artist_reference}; refining genre and groove targets."
                )

        reason = " | ".join(feedback) if feedback else "Results appear reasonable, no further refinement suggested."
        return {
            "quality_score": quality_score,
            "feedback": reason,
            "should_refine": bool(refined_profile),
            "refined_profile": refined_profile,
        }

    def _map_tag_to_genre(self, tag: str) -> str:
        normalized_tag = tag.lower().strip()
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

    def _is_artist_style_query(self, normalized_query: str, seed_artist: str) -> bool:
        if not seed_artist:
            return False
        return self._contains_any(normalized_query, ["like", "similar to", "in the style of", "sounds like", "sound like"])

    def _get_artist_style_targets(self, seed_artist: str) -> Dict:
        targets = {"genre": "", "similar_artists": []}
        if not self.api_client or not seed_artist:
            return targets

        try:
            tags = self.api_client.get_artist_tags(seed_artist, limit=5)
            for tag in tags:
                mapped = self._map_tag_to_genre(tag)
                if mapped:
                    targets["genre"] = mapped
                    break
        except LastFmAPIError as exc:
            self._log(f"API ERROR: {exc}")

        try:
            similar_artists = self.api_client.get_similar_artists(seed_artist, limit=10)
            targets["similar_artists"] = [artist.lower().strip() for artist in similar_artists]
        except LastFmAPIError as exc:
            self._log(f"API ERROR: {exc}")
        return targets

    def _has_artist_style_alignment(
        self, recommendations: List[Tuple[Dict, float, str]], seed_artist: str, style_targets: Dict
    ) -> bool:
        normalized_seed = seed_artist.lower().strip()
        similar_artists = set(style_targets.get("similar_artists", []))
        target_genre = style_targets.get("genre", "")
        for song, _, _ in recommendations:
            song_artist = song.get("artist", "").lower().strip()
            song_genre = song.get("genre", "").lower().strip()
            if normalized_seed and (song_artist == normalized_seed or normalized_seed in song_artist):
                return True
            if song_artist in similar_artists:
                return True
            if target_genre and song_genre == target_genre:
                return True
        return False

    def _contains_any(self, query: str, keywords: List[str]) -> bool:
        return any(keyword in query for keyword in keywords)

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)


class SemanticEvaluator:
    """
    Adapter for external model-based evaluation.
    evaluator_fn should return:
    {"quality_score": int, "feedback": str, "should_refine": bool, "refined_profile": dict}
    """

    def __init__(self, evaluator_fn: Callable[[str, Dict, List[Tuple[Dict, float, str]]], Dict]):
        self.evaluator_fn = evaluator_fn

    def evaluate(
        self, query: str, profile: Dict, recommendations: List[Tuple[Dict, float, str]], quality_threshold: int
    ) -> Dict:
        _ = quality_threshold  # threshold can be handled inside evaluator_fn.
        return self.evaluator_fn(query, profile, recommendations)
