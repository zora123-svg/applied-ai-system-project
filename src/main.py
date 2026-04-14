"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    # Starter example profile
    user_prefs = {
        "favorite_genre":    "pop",
        "favorite_mood":     "happy",
        "target_energy":     0.8,
        "likes_acoustic":    False,
        "prefers_instrumental": False,
        "target_speechiness": 0.1,
        "target_liveness":   0.1,
        "min_popularity":    40,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    divider = "-" * 44
    print("\nTop Recommendations")
    print("=" * 44)
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f" #{rank}  {song['title']} by {song['artist']}")
        print(f"      Score: {score:.3f}")
        print(divider)
        for reason in explanation.split("; "):
            print(f"      {reason}")
        print()


if __name__ == "__main__":
    main()
