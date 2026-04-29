"""
Command line runner for the Agentic Music Recommendation System.

Supports the original static profile mode and a new --agent mode that
accepts natural language requests, retrieves metadata via Last.fm, and
uses an iterative evaluation loop to refine recommendations.
"""

import argparse
from src.api_client import LastFmClient
from src.agent import RecommenderAgent
from src.ai_inference import AIInferenceError, LLMEvaluationProvider, LLMProfileExtractor, create_inference_client_from_env
from src.recommender import load_songs, recommend_songs

# ---------------------------------------------------------------------------
# Three distinct user profiles
# ---------------------------------------------------------------------------
PROFILES = {
    "Happy Pop Listener": {
        "favorite_genre":      "pop",
        "favorite_mood":       "happy",
        "target_energy":       0.8,
        "likes_acoustic":      False,
        "prefers_instrumental": False,
        "target_speechiness":  0.1,
        "target_liveness":     0.1,
        "target_valence":      0.8,
        "min_popularity":      40,
        "target_danceability": 0.75,
        "target_tempo":        120.0,
        "preferred_era":       "2020s",
        "avoid_explicit":      False,
        "target_loudness":     -5.0,
    },
    "Hip-Hop Fan": {
        "favorite_genre":      "hip-hop",
        "favorite_mood":       "confident",
        "target_energy":       0.85,
        "likes_acoustic":      False,
        "prefers_instrumental": False,
        "target_speechiness":  0.3,
        "target_liveness":     0.15,
        "target_valence":      0.7,
        "min_popularity":      50,
        "target_danceability": 0.9,
        "target_tempo":        95.0,
        "preferred_era":       "2020s",
        "avoid_explicit":      False,
        "target_loudness":     -4.0,
    },
    "Acoustic Chill Listener": {
        "favorite_genre":      "folk",
        "favorite_mood":       "nostalgic",
        "target_energy":       0.4,
        "likes_acoustic":      True,
        "prefers_instrumental": False,
        "target_speechiness":  0.1,
        "target_liveness":     0.4,
        "target_valence":      0.6,
        "min_popularity":      0,
        "target_danceability": 0.55,
        "target_tempo":        100.0,
        "preferred_era":       "2010s",
        "avoid_explicit":      True,
        "target_loudness":     -13.0,
    },
}


# ---------------------------------------------------------------------------
# Visual table helpers
# ---------------------------------------------------------------------------
def _highlights(explanation: str, n: int = 3) -> str:
    """Return up to n non-zero scoring reasons as a compact string."""
    parts = [r.strip() for r in explanation.split(";")]
    keep = [
        r for r in parts
        if "(+0.0)" not in r
        and "penalty" not in r
        and "boost" not in r
        and "Filtered" not in r
    ]
    return "  |  ".join(keep[:n])


def print_table(profile_name: str, recommendations: list, mode: str) -> None:
    """Print recommendations as a formatted ASCII table."""
    W = 84
    TW, AW = 22, 16
    print(f"\nProfile: {profile_name}  |  Mode: {mode}")
    print("=" * W)
    print(f" {'#':<3} {'Title':<{TW}} {'Artist':<{AW}} {'Score':>6}  Top Reasons")
    print("-" * W)
    for rank, (song, score, explanation) in enumerate(recommendations, 1):
        title  = song["title"][:TW]
        artist = song["artist"][:AW]
        top    = _highlights(explanation, n=3)
        print(f" {rank:<3} {title:<{TW}} {artist:<{AW}} {score:>6.3f}  {top}")
    print("=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the music recommender in static profile mode or as an agentic workflow."
    )
    parser.add_argument("--agent", type=str, help="Natural language query for the recommender agent.")
    parser.add_argument(
        "--profile",
        type=str,
        choices=list(PROFILES.keys()),
        help="Run a predefined profile instead of the agent.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("relevance", "discovery"),
        default="relevance",
        help="Ranking mode for recommendations.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of recommendations to return.")
    parser.add_argument(
        "--use-external-ai",
        action="store_true",
        help="Use external AI inference for profile extraction and evaluation.",
    )
    args = parser.parse_args()

    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs\n")

    if args.agent:
        try:
            api_client = LastFmClient.from_env()
        except ValueError as exc:
            print(f"Error: {exc}")
            print("Set LASTFM_API_KEY in .env or your environment to use --agent mode.")
            return

        evaluator = None
        profile_extractor = None
        if args.use_external_ai:
            try:
                inference_client = create_inference_client_from_env()
                evaluator = LLMEvaluationProvider(inference_client)
                profile_extractor = LLMProfileExtractor(inference_client)
            except AIInferenceError as exc:
                print(f"AI inference disabled: {exc}")
                print("Falling back to built-in heuristic inference.")

        agent = RecommenderAgent(api_client=api_client, evaluator=evaluator, profile_extractor=profile_extractor)
        recommendations, reasoning = agent.run(args.agent, songs, k=args.k, mode=args.mode)
        print_table("Agent Query", recommendations, args.mode)
        if agent.last_evaluation.get("safe_response"):
            print("\nReliability note:")
            print(agent.last_evaluation["safe_response"])
            print(
                "confidence="
                f"{agent.last_evaluation.get('confidence_score', 0.0):.2f}, "
                f"coverage_gap={agent.last_evaluation.get('coverage_gap', False)}"
            )
        print("\nReasoning chain:\n")
        print(reasoning)
        return

    if args.profile:
        profile = PROFILES[args.profile]
        print(f"Running profile: {args.profile} | Mode: {args.mode}\n")
        recs = recommend_songs(profile, songs, k=args.k, mode=args.mode)
        print_table(args.profile, recs, args.mode)
        return

    # Default behavior preserves existing static profile mode.
    print("=" * 84)
    print("  RECOMMENDATIONS  —  Relevance Mode")
    print("=" * 84)
    for name, profile in PROFILES.items():
        recs = recommend_songs(profile, songs, k=5, mode="relevance")
        print_table(name, recs, "relevance")

    print("\n\n" + "=" * 84)
    print("  RANKING MODE COMPARISON  —  Acoustic Chill Listener")
    print("=" * 84)
    for mode in ("relevance", "discovery"):
        recs = recommend_songs(PROFILES["Acoustic Chill Listener"], songs, k=5, mode=mode)
        print_table("Acoustic Chill Listener", recs, mode)


if __name__ == "__main__":
    main()
