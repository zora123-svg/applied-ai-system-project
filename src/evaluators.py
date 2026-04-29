import re
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
                "confidence_score": 0.0,
                "evidence_count": 0,
                "coverage_gap": True,
                "safe_response": "I don't have enough evidence to recommend songs for that request yet.",
            }

        scores = [score for _, score, _ in recommendations]
        avg_score = sum(scores) / len(scores)
        quality_score = min(10, max(1, int(round(avg_score / 12.0 * 10.0))))

        energies = [song["energy"] for song, _, _ in recommendations]
        valences = [song["valence"] for song, _, _ in recommendations]
        tempos = [song["tempo_bpm"] for song, _, _ in recommendations if song.get("tempo_bpm") is not None]
        avg_energy = sum(energies) / len(energies)
        avg_valence = sum(valences) / len(valences)
        avg_tempo = (sum(tempos) / len(tempos)) if tempos else profile.get("target_tempo", 110.0)

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

        # Reflection heuristic: upbeat intent should satisfy both energy and tempo coherence.
        if self._contains_any(normalized, ["upbeat", "energetic", "high energy", "lively"]):
            if avg_energy >= 0.75 and avg_tempo < 105.0:
                refined_profile["target_tempo"] = min(220.0, profile.get("target_tempo", 110.0) + 10.0)
                feedback.append("Upbeat request needs faster tempo; increasing target tempo for next iteration.")
        if self._contains_any(normalized, ["chill", "relaxed", "mellow"]):
            if avg_energy <= 0.45 and avg_tempo > 100.0:
                refined_profile["target_tempo"] = max(40.0, profile.get("target_tempo", 100.0) - 8.0)
                feedback.append("Chill request needs slower pacing; lowering target tempo for next iteration.")

        if quality_score < quality_threshold and not refined_profile and avg_score < 6.0:
            refined_profile["target_energy"] = min(1.0, profile["target_energy"] + 0.1)
            feedback.append("Refining energy target to improve recommendation quality.")

        requested_genres = self._extract_requested_genres(normalized)
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

        recommended_genres = {song.get("genre", "").lower().strip() for song, _, _ in recommendations}
        missing_requested_genres = [genre for genre in requested_genres if genre not in recommended_genres]
        coverage_gap = bool(missing_requested_genres)
        if coverage_gap:
            feedback.append(
                f"Dataset coverage gap: requested genre(s) not found in top candidates ({', '.join(missing_requested_genres)})."
            )

        unresolved_artist_intent = self._has_unresolved_artist_intent(normalized, profile, requested_genres)
        if unresolved_artist_intent:
            coverage_gap = True
            feedback.append(
                "Artist intent unresolved: query requests artist-specific music but no seed artist was identified."
            )

        if self._has_unresolved_era_intent(normalized, recommendations):
            coverage_gap = True
            feedback.append("Era intent unresolved: requested era is not represented in top recommendations.")

        evidence_count = self._count_supported_signals(profile, recommendations, requested_genres)
        confidence_score = self._compute_confidence_score(
            quality_score=quality_score,
            evidence_count=evidence_count,
            has_coverage_gap=coverage_gap,
            needs_refinement=bool(refined_profile),
        )
        safe_response = ""
        if coverage_gap or confidence_score < 0.65:
            safe_response = (
                "Low confidence result: I may not have enough matching songs for this request. "
                "Try broadening the genre or add more songs in the requested style."
            )
        if feedback:
            reason = "Not reasonable: " + " | ".join(feedback)
        else:
            reason = "Reasonable and evidence-backed; no further refinement suggested."
        return {
            "quality_score": quality_score,
            "feedback": reason,
            "should_refine": bool(refined_profile),
            "refined_profile": refined_profile,
            "confidence_score": confidence_score,
            "evidence_count": evidence_count,
            "coverage_gap": coverage_gap,
            "safe_response": safe_response,
        }

    def _map_tag_to_genre(self, tag: str) -> str:
        normalized_tag = tag.lower().strip()
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

    def _has_unresolved_artist_intent(self, query: str, profile: Dict, requested_genres: List[str]) -> bool:
        has_seed_artist = bool(profile.get("seed_artist", "").strip())
        if has_seed_artist:
            return False
        if requested_genres:
            return False
        artist_intent_patterns = [
            r"\bsimilar to\b",
            r"\blike\b",
            r"\bin the style of\b",
            r"\bsounds like\b",
            r"\bwant\s+[a-z0-9 .&']+\s+music\b",
        ]
        return any(re.search(pattern, query) for pattern in artist_intent_patterns)

    def _has_unresolved_era_intent(self, query: str, recommendations: List[Tuple[Dict, float, str]]) -> bool:
        era_patterns = [
            r"\b(19|20)\d0'?s\b",
            r"\b(19|20)\d{2}'s\b",
            r"\b\d{2}s\b",
        ]
        if not any(re.search(pattern, query) for pattern in era_patterns):
            return False
        recommended_eras = {song.get("era", "").lower().strip() for song, _, _ in recommendations}
        normalized_query = query.replace("'", "")
        if "1950s" in normalized_query and "1950s" not in recommended_eras:
            return True
        if "1960s" in normalized_query and "1960s" not in recommended_eras:
            return True
        if "1970s" in normalized_query and "1970s" not in recommended_eras:
            return True
        if "1980s" in normalized_query and "1980s" not in recommended_eras:
            return True
        if "1990s" in normalized_query and "1990s" not in recommended_eras:
            return True
        if "2000s" in normalized_query and "2000s" not in recommended_eras:
            return True
        if "2010s" in normalized_query and "2010s" not in recommended_eras:
            return True
        if "2020s" in normalized_query and "2020s" not in recommended_eras:
            return True
        return False

    def _extract_requested_genres(self, query: str) -> List[str]:
        normalized_query = query.lower()
        genre_aliases = {
            "kpop": "k-pop",
            "k-pop": "k-pop",
            "korean pop": "k-pop",
            "hip hop": "hip-hop",
            "hip-hop": "hip-hop",
            "rnb": "r&b",
            "r&b": "r&b",
        }
        known_genres = [
            "pop",
            "rock",
            "hip-hop",
            "electronic",
            "lofi",
            "jazz",
            "folk",
            "classical",
            "blues",
            "metal",
            "country",
            "synthwave",
            "indie pop",
            "k-pop",
            "afrobeats",
            "reggaeton",
            "latin pop",
            "house",
            "techno",
            "drill",
            "funk",
            "soul",
            "r&b",
            "punk",
            "ambient",
        ]
        extracted = []
        for alias, canonical in genre_aliases.items():
            alias_pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            if re.search(alias_pattern, normalized_query) and canonical not in extracted:
                extracted.append(canonical)

        # Prevent broad genre false positives (e.g., "pop" inside "kpop").
        scrubbed_query = normalized_query
        for alias in genre_aliases:
            alias_pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            scrubbed_query = re.sub(alias_pattern, " ", scrubbed_query)

        for genre in known_genres:
            genre_pattern = rf"(?<![a-z0-9]){re.escape(genre)}(?![a-z0-9])"
            if re.search(genre_pattern, scrubbed_query) and genre not in extracted:
                extracted.append(genre)
        return extracted

    def _count_supported_signals(self, profile: Dict, recommendations: List[Tuple[Dict, float, str]], requested_genres: List[str]) -> int:
        evidence = 0
        target_genre = profile.get("favorite_genre", "").lower().strip()
        top_genres = [song.get("genre", "").lower().strip() for song, _, _ in recommendations[:3]]
        if target_genre and any(genre == target_genre for genre in top_genres):
            evidence += 1
        if requested_genres and any(req in top_genres for req in requested_genres):
            evidence += 1
        if profile.get("seed_artist"):
            seed = profile["seed_artist"].lower().strip()
            top_artists = [song.get("artist", "").lower().strip() for song, _, _ in recommendations[:3]]
            if any(seed in artist for artist in top_artists):
                evidence += 1
        return evidence

    def _compute_confidence_score(
        self, quality_score: int, evidence_count: int, has_coverage_gap: bool, needs_refinement: bool
    ) -> float:
        quality_component = quality_score / 10.0
        evidence_component = min(1.0, evidence_count / 2.0)
        confidence = round(0.6 * quality_component + 0.4 * evidence_component, 2)
        if has_coverage_gap:
            confidence = max(0.0, round(confidence - 0.35, 2))
        if needs_refinement:
            confidence = max(0.0, round(confidence - 0.15, 2))
        return confidence

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
