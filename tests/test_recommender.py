from src.recommender import Song, UserProfile, Recommender, recommend_songs


def make_sample_songs():
    return [
        Song(
            id=1,
            title="Happy Pop Hit",
            artist="Top Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=110,
            valence=0.8,
            danceability=0.8,
            acousticness=0.2,
            instrumentalness=0.0,
            speechiness=0.1,
            liveness=0.1,
            popularity=90,
            loudness=-5.0,
        ),
        Song(
            id=2,
            title="Low Popularity Pop",
            artist="Indie Star",
            genre="pop",
            mood="happy",
            energy=0.6,
            tempo_bpm=100,
            valence=0.7,
            danceability=0.7,
            acousticness=0.2,
            instrumentalness=0.0,
            speechiness=0.1,
            liveness=0.1,
            popularity=40,
            loudness=-5.0,
        ),
        Song(
            id=3,
            title="Duplicate Artist A",
            artist="Top Artist",
            genre="pop",
            mood="happy",
            energy=0.7,
            tempo_bpm=105,
            valence=0.75,
            danceability=0.75,
            acousticness=0.2,
            instrumentalness=0.0,
            speechiness=0.1,
            liveness=0.1,
            popularity=80,
            loudness=-5.0,
        ),
        Song(
            id=4,
            title="Different Artist",
            artist="Other Artist",
            genre="pop",
            mood="happy",
            energy=0.75,
            tempo_bpm=108,
            valence=0.76,
            danceability=0.76,
            acousticness=0.2,
            instrumentalness=0.0,
            speechiness=0.1,
            liveness=0.1,
            popularity=85,
            loudness=-5.0,
        ),
    ]


def test_recommend_returns_songs_sorted_by_score():
    songs = make_sample_songs()
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    recommender = Recommender(songs)
    results = recommender.recommend(user, k=2)

    assert len(results) == 2
    assert results[0].title == "Happy Pop Hit"
    assert results[1].title in {"Different Artist", "Duplicate Artist A", "Low Popularity Pop"}


def test_recommend_filters_by_min_popularity():
    songs = make_sample_songs()
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
        min_popularity=80,
    )
    recommender = Recommender(songs)

    results = recommender.recommend(user, k=5)
    assert all(song.popularity >= 80 for song in results)
    assert all(song.genre == "pop" for song in results)
    assert len(results) >= 1


def test_discovery_mode_boosts_low_popularity_candidates():
    songs = [
        {
            "id": 1,
            "title": "Mainstream Hit",
            "artist": "Popular Band",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "tempo_bpm": 120,
            "valence": 0.8,
            "danceability": 0.8,
            "acousticness": 0.2,
            "instrumentalness": 0.0,
            "speechiness": 0.1,
            "liveness": 0.1,
            "popularity": 90,
            "era": "2020s",
            "explicit": 0,
            "loudness": -5.0,
        },
        {
            "id": 2,
            "title": "Indie Gem",
            "artist": "Indie Star",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.6,
            "tempo_bpm": 100,
            "valence": 0.7,
            "danceability": 0.7,
            "acousticness": 0.2,
            "instrumentalness": 0.0,
            "speechiness": 0.1,
            "liveness": 0.1,
            "popularity": 40,
            "era": "2020s",
            "explicit": 0,
            "loudness": -5.0,
        },
    ]
    profile = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
        "prefers_instrumental": False,
        "target_speechiness": 0.1,
        "target_liveness": 0.1,
        "target_valence": 0.75,
        "min_popularity": 0,
        "target_danceability": 0.75,
        "target_tempo": 110.0,
        "preferred_era": "2020s",
        "avoid_explicit": False,
        "target_loudness": -5.0,
    }

    relevance_order = [song[0]["id"] for song in recommend_songs(profile, songs, k=2, mode="relevance")]
    discovery_order = [song[0]["id"] for song in recommend_songs(profile, songs, k=2, mode="discovery")]

    assert relevance_order[0] == 1
    assert discovery_order[0] == 2
    assert relevance_order != discovery_order


def test_explain_recommendation_returns_non_empty_string():
    songs = make_sample_songs()
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    recommender = Recommender(songs)
    explanation = recommender.explain_recommendation(user, songs[0])

    assert "genre match" in explanation
    assert explanation.strip() != ""


def test_artist_diversity_penalty_is_applied_to_second_duplicate_artist():
    songs = make_sample_songs()
    profile = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
        "prefers_instrumental": False,
        "target_speechiness": 0.1,
        "target_liveness": 0.1,
        "target_valence": 0.8,
        "min_popularity": 0,
        "target_danceability": 0.75,
        "target_tempo": 110.0,
        "preferred_era": "2020s",
        "avoid_explicit": False,
        "target_loudness": -5.0,
    }
    songs = [
        {
            "id": 1,
            "title": "Happy Pop Hit",
            "artist": "Top Artist",
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
            "popularity": 90,
            "era": "2020s",
            "explicit": 0,
            "loudness": -5.0,
        },
        {
            "id": 2,
            "title": "Duplicate Artist A",
            "artist": "Top Artist",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.7,
            "tempo_bpm": 105,
            "valence": 0.75,
            "danceability": 0.75,
            "acousticness": 0.2,
            "instrumentalness": 0.0,
            "speechiness": 0.1,
            "liveness": 0.1,
            "popularity": 80,
            "era": "2020s",
            "explicit": 0,
            "loudness": -5.0,
        },
        {
            "id": 3,
            "title": "Different Artist",
            "artist": "Other Artist",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.75,
            "tempo_bpm": 108,
            "valence": 0.76,
            "danceability": 0.76,
            "acousticness": 0.2,
            "instrumentalness": 0.0,
            "speechiness": 0.1,
            "liveness": 0.1,
            "popularity": 85,
            "era": "2020s",
            "explicit": 0,
            "loudness": -5.0,
        },
    ]
    scored = recommend_songs(profile, songs, k=3, mode="relevance")
    explanations = [explanation for _, _, explanation in scored]

    assert any("artist diversity penalty" in explanation for explanation in explanations)
