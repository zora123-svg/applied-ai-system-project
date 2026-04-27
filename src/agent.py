import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from src.api_client import LastFmAPIError, LastFmClient
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


class RecommenderAgent:
    def __init__(self, api_client: LastFmClient = None, max_iterations: int = 3, quality_threshold: int = 7):
        self.api_client = api_client
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.log_path = Path("logs") / "agent.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

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

        refined_profile = {}
        feedback = []
        normalized = query.lower()

        if self._contains_any(normalized, ["upbeat", "energetic", "high energy", "lively"]):
            if avg_energy < profile["target_energy"] + 0.1:
                new_energy = min(1.0, profile["target_energy"] + 0.15)
                refined_profile["target_energy"] = new_energy
                feedback.append("Increase target energy to better match upbeat requests.")
        if self._contains_any(normalized, ["chill", "relaxed", "mellow"]):
            if avg_energy > profile["target_energy"] - 0.1:
                new_energy = max(0.0, profile["target_energy"] - 0.15)
                refined_profile["target_energy"] = new_energy
                feedback.append("Lower target energy to match chill requests.")
        if self._contains_any(normalized, ["happy", "joyful", "bright"]):
            if avg_valence < profile["target_valence"] + 0.1:
                new_valence = min(1.0, profile["target_valence"] + 0.15)
                refined_profile["target_valence"] = new_valence
                feedback.append("Raise target valence for happier results.")
        if self._contains_any(normalized, ["sad", "moody", "melancholic"]):
            if avg_valence > profile["target_valence"] - 0.1:
                new_valence = max(0.0, profile["target_valence"] - 0.15)
                refined_profile["target_valence"] = new_valence
                feedback.append("Lower target valence for sad or moody requests.")

        if quality_score < self.quality_threshold and not refined_profile:
            if avg_score < 6.0:
                new_energy = min(1.0, profile["target_energy"] + 0.1)
                refined_profile["target_energy"] = new_energy
                feedback.append("Refining energy target to improve recommendation quality.")

        reason = " | ".join(feedback) if feedback else "Results appear reasonable, no further refinement suggested."
        should_refine = bool(refined_profile)
        return {
            "quality_score": quality_score,
            "feedback": reason,
            "should_refine": should_refine,
            "refined_profile": refined_profile,
        }

    def merge_profile(self, profile: Dict, refined: Dict) -> Dict:
        merged = profile.copy()
        for key, value in refined.items():
            if isinstance(value, float):
                if key.startswith("target_"):
                    value = max(0.0, min(1.0, value))
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

            profile_snapshot = {k: v for k, v in profile.items() if k != "seed_artist"}
            self.append_log(f"ITERATION {iteration} | profile: {profile_snapshot}")
            self.append_log(
                f"ITERATION {iteration} | top_result: {recommendations[0][0]['title']} ({recommendations[0][0]['artist']}) score={recommendations[0][1]:.2f}"
            )
            self.append_log(
                f"ITERATION {iteration} | quality: {evaluation['quality_score']}/10 | feedback: {evaluation['feedback']}"
            )

            trace_lines.append(f"ITERATION {iteration} | profile: {profile_snapshot}")
            trace_lines.append(
                f"ITERATION {iteration} | quality: {evaluation['quality_score']}/10 | feedback: {evaluation['feedback']}"
            )

            if not evaluation["should_refine"]:
                trace_lines.append(f"SATISFIED — stopping after {iteration} iteration(s)")
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
                if tag in GENRE_KEYWORDS:
                    return GENRE_KEYWORDS[tag]
                if "rock" in tag:
                    return "rock"
                if "pop" in tag:
                    return "pop"
                if "electronic" in tag:
                    return "electronic"
                if "hip hop" in tag or "hip-hop" in tag:
                    return "hip-hop"
                if "folk" in tag:
                    return "folk"

        for keyword, genre in GENRE_KEYWORDS.items():
            if keyword in query:
                return genre
        return ""

    def _infer_mood(self, query: str) -> str:
        for keyword, mood in MOOD_KEYWORDS.items():
            if keyword in query:
                return mood
        return ""

    def _infer_energy(self, query: str, default: float) -> float:
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
