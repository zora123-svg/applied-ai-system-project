# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**VibeFit 1.0**  
A rule-based music recommender that scores songs against your stated taste profile and ranks them by how well they fit.

---

## 2. Intended Use  

**Goal:** Given a listener's stated preferences — favorite genre, mood, energy level, and a few production style choices — VibeFit 1.0 ranks a small catalog of songs and returns the top five that best match that profile.

**Who it is for:** This is a classroom simulation built to demonstrate how scoring-based recommender systems work. It is not connected to any streaming service and does not learn from listening history.

**What it assumes:** The user already knows what genre and mood they want. The system takes those answers at face value and does not try to infer taste from behavior.

**Non-intended use:** VibeFit 1.0 should not be used for production music discovery, personalized playlists, or any context where recommendations influence purchasing or content moderation decisions. The catalog is too small, the scoring is hand-tuned, and the system has no concept of fairness across artists or genres.

---

## 3. How the Model Works  

Imagine a judge at a talent show with a scorecard. For every song in the catalog, the judge checks it against your preferences and awards points in seven categories:

- **Genre match (+1.0):** Does the song's genre match exactly what you said? If yes, it gets a point. If it is close but not exact — say "indie pop" when you asked for "pop" — it gets nothing.
- **Mood match (+1.0):** Is the song's tagged mood the same as yours? Happy, sad, intense, chill — it either matches or it does not.
- **Valence fit (up to +1.0):** Valence measures how musically positive or happy a song sounds on a 0–1 scale. The closer the song's valence to your target, the more points it earns.
- **Energy fit (up to +2.0):** How close is the song's energy to your target? A perfect match earns the full 2 points; the further away it is, the fewer points it gets.
- **Danceability fit (up to +1.0):** How closely does the song's groove match what you said you want? High-danceability fans get rewarded for tracks that are built to move to.
- **Tempo fit (up to +1.0):** How close is the song's BPM to your preferred tempo? Normalized over a 100 BPM window so that a 20 BPM difference costs less than a 100 BPM difference.
- **Era match (+0.5):** If you said you prefer a specific era (classic, 2010s, 2020s), a song from that era earns a small bonus. If you have no era preference, this dimension is skipped entirely.
- **Explicit content fit (+0.5):** If you want to avoid explicit songs, any song flagged explicit earns no credit here. Otherwise all songs earn the half point.
- **Loudness fit (up to +1.0):** How close is the song's average loudness (in decibels) to what you prefer? Normalized over a 15 dB window.
- **Acoustic character (+0.5):** If you said you do not like acoustic music and the song is not acoustic, it gets half a point. Same logic if you do like acoustic and the song is.
- **Instrumental preference (+0.5):** Does the song lean instrumental or vocal, and does that match what you said you want?
- **Speechiness fit (up to +1.0):** How close is the amount of talking or rapping in the song to what you prefer?
- **Liveness fit (up to +1.0):** Does the song sound like a live concert recording, or a clean studio track? Points for closeness to your preference.

The judge adds up all the points (maximum 12.0), sorts every song from highest to lowest, and hands you the top five.

**Key design changes from the original:** The genre bonus started at +2.0, which meant it dominated everything else. After testing, it was reduced to +1.0 and the energy bonus was doubled to compensate. Five new scoring dimensions were added (valence, danceability, tempo, era, explicit content, and loudness) to give the system a richer picture of each song. An artist diversity pass runs last: if the same artist would appear twice in the top five, the second song takes a 20% score penalty so another artist can surface instead.

----

## 4. Data  

The catalog contains **18 songs** spread across **15 genres** and **9 moods**. Genres include pop, lofi, rock, indie pop, blues, hip-hop, electronic, ambient, jazz, synthwave, r&b, metal, country, classical, and folk. Moods include happy, chill, intense, focused, moody, relaxed, sad, romantic, nostalgic, confident, euphoric, melancholic, and angry.

Each song has 14 numeric and categorical features: energy, tempo in BPM, valence (emotional brightness on a 0–1 scale), danceability, acousticness, instrumentalness, speechiness, liveness, popularity, era (release decade: classic / 2010s / 2020s), explicit (0 or 1), and loudness (average dB level), plus the text labels for genre and mood. Three new attributes — era, explicit, and loudness — were added to the original dataset to expand the scoring surface.

**Significant gaps:** Most genres have only one song, which means genre-based filtering can produce no real variety. The catalog has no representation for genres like reggae, Latin, or K-pop. The catalog also has no songs that combine high energy with a sad or melancholic mood, so listeners with that combination of preferences will always receive mismatched results.

---

## 5. Diversity and Fairness

**Artist diversity penalty:** After all songs are scored and sorted, the system checks whether any artist appears more than once in the top-K results. If so, every song after that artist's first appearance takes a 20% score reduction (`score × 0.8`). The list is then re-sorted. This prevents a scenario where a single prolific artist dominates all five recommendation slots — which would happen in the current catalog because Neon Echo has two songs (Sunrise City and Night Drive Loop) that both score well for energetic pop listeners.

**Impact:** For a happy pop listener, without the penalty both Neon Echo songs could appear in the top five. With the penalty, Night Drive Loop drops enough in score that other artists' songs surface instead, producing a more varied recommendation list.

**Limitation of this approach:** The penalty is applied after scoring, not during. This means the artist's best song still gets its full score and cannot be bumped by a different artist's song. The diversity improvement only affects positions 2–K, not position 1. A stronger approach would be to explicitly constrain the selection step to ensure a minimum number of distinct artists.

**Discovery mode:** A second ranking mode boosts songs with popularity below 65 by 15% before sorting. This addresses the mainstream bias in the catalog — popular songs naturally score well on many dimensions simply because they were designed for broad appeal. The discovery boost gives less-known but well-matched songs a chance to surface in the top results. The fairness motivation: listeners who prefer niche or independent music should not always receive the same five popular tracks just because popularity correlates with other features the scorer rewards.

---

## 6. Strengths

The system works best when a listener's preferences are internally consistent with how music is typically produced. A "happy pop" fan who wants moderate energy, vocal-forward songs, and a clean studio sound will get a near-perfect top recommendation (Sunrise City scored 6.89 out of 7.0) because those preferences naturally cluster together in the catalog.

Every recommendation comes with a point-by-point explanation of why the song was chosen. This transparency is a genuine advantage over black-box systems — a listener can read the explanation and immediately understand whether the system understood them correctly or not.

The scoring also handles the distinction between acoustic and electric, and between vocal and instrumental, cleanly. A listener who explicitly wants non-acoustic, vocal-forward music will never receive an ambient instrumental track at the top of their list, because those preferences each carry their own filtering weight.

---

## 7. Limitations and Bias

The most significant weakness discovered during testing is that genre matching uses strict string equality, which causes it to treat semantically similar genres as completely unrelated. For example, when a user prefers "pop," a song tagged as "indie pop" receives zero genre credit even though the two genres share nearly identical production style, tempo, and audience. This became visible in experiments where "Gym Hero" (pop, intense mood) outranked "Rooftop Lights" (indie pop, happy mood) for a user who explicitly wanted happy pop music — the exact-match genre bonus outweighed the mood mismatch entirely. Halving the genre weight from +2.0 to +1.0 corrected that specific ranking, but the underlying issue remains: any user whose preferred genre is labeled slightly differently in the catalog will receive systematically worse recommendations than a user whose genre label matches exactly. Additionally, with only one or two songs per genre in the current 18-song dataset, a user who prefers a rare genre such as blues, metal, or jazz will always receive the same single song as their top recommendation regardless of how well it actually fits their other preferences.

---

## 8. Evaluation

Three user profiles were tested during evaluation. The first was a standard "happy pop" listener targeting high energy (0.8), no acoustic preference, and low speechiness — essentially someone who wants upbeat radio-friendly pop. The second was an adversarial profile that combined a high energy target (0.9) with a "sad" mood preference, designed to see how the system handles emotionally contradictory requests that real listeners do make. The third was an edge case: a popularity floor set to 100, which silently emptied the catalog and returned zero recommendations with no warning to the user.

The most surprising result came from the happy pop profile. "Gym Hero," a workout-pump track tagged as intense, consistently ranked above "Rooftop Lights," a genuinely happy indie pop song. Nothing about Gym Hero sounds like what a happy pop listener is asking for — it is fast, driving, and built for the gym — yet the recommender chose it because "pop" matched the genre label exactly, earning a full bonus that "indie pop" never could. Rooftop Lights, despite matching the mood perfectly and sitting closer to the target energy, lost every time simply because its genre tag contained two extra words. After reducing the genre weight and increasing the energy weight, the ranking corrected itself and Rooftop Lights moved to the second spot where it belongs. That single change confirmed the original weights were miscalibrated rather than the scoring logic being fundamentally broken.

---

## 9. Future Work

**1. Add valence as a scored signal.**
Valence is already in every song's data and it directly measures how emotionally positive or negative a song sounds. Including it would make the mood matching far more accurate without requiring any new data. A happy-mood listener should prefer songs with high valence; a sad-mood listener should prefer low valence. Right now the system ignores this entirely.

**2. Replace the hard acoustic and instrumental thresholds with smooth proximity scores.**
Currently, a song with acousticness 0.59 and a song with acousticness 0.61 are treated completely differently — one earns +0.5 and the other earns nothing, despite sounding nearly identical. Using a continuous formula instead of a yes/no cutoff at 0.6 would produce much more stable and fair rankings.

**3. Expand the catalog to at least five songs per genre.**
No amount of scoring improvement helps a jazz fan when there is only one jazz song in the catalog. Adding genre variety is the single highest-leverage change available — it would immediately benefit any user whose preferred genre is currently underrepresented.

---

## 10. Personal Reflection

**Biggest learning moment**

My biggest learning moment was discovering that the Gym Hero problem was not a bug — it was the weights working exactly as designed, just in the wrong direction. I spent time staring at the output wondering why a gym pump-up track was appearing for someone who asked for happy music, and the answer turned out to be one number: the genre bonus was set to +2.0. That single value was large enough to override a completely wrong mood match. Once I understood that, I realized that in any scoring system, the weights are the real design decisions. Getting the logic right is necessary, but it is not enough. You can have perfectly correct code and still ship a bad recommender if your numbers are off. That shifted how I think about this kind of engineering work — the math matters as much as the structure.

**How AI tools helped, and when I double-checked**

Using AI during this project helped me move faster through analysis I would have otherwise done manually: tracing score calculations step by step, identifying which songs would be affected by a weight change before running the code, and naming patterns like the threshold cliff effect at the 0.6 acousticness cutoff. Where I had to be careful was in trusting the analysis without verifying it against the actual data. For example, when the valence column was identified as unused, I went back and checked the CSV and the score_song function myself to confirm it was genuinely not scored — because "it looks like it might be used" and "it is actually used" are easy to confuse when you are reading code quickly. The rule I developed was: use AI reasoning for direction, but read the actual file before acting on anything specific.

**What surprised me about simple algorithms feeling like recommendations**

The most surprising thing was that a system with only seven scoring rules, no machine learning, and no listening history could still produce a result that felt personally relevant. When Sunrise City came back as the top recommendation with a 6.89 out of 7.0, and the explanation listed every dimension it matched, it felt like the system had understood the request — even though all it did was arithmetic. That made me realize that "feeling like a recommendation" does not require intelligence or learning. It requires transparency and alignment. The moment you can see why something was chosen and the reason makes sense, the result feels earned. The moment the reason does not make sense — like Gym Hero showing up — it breaks the illusion immediately. The explainability is what separates a system that feels trustworthy from one that feels random.

**What I would try next**

The first thing I would add is valence as a scored signal, since it is already in the dataset and is the most direct numeric measure of emotional mood. After that, I would replace the binary threshold checks for acousticness and instrumentalness with soft proximity scores — the same formula used for energy — so that a song at 0.59 and a song at 0.61 acousticness are not treated as fundamentally different. Finally, I would expand the catalog to at least five songs per genre before doing any further tuning, because right now the dataset is the binding constraint. Better weights cannot help a jazz fan when there is only one jazz song to recommend.
