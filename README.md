# 🎵 Agentic Music Recommendation System

## Project Summary

This repository implements an agentic music recommendation system that converts conversational user requests into structured listening profiles, retrieves real music metadata from an external API, and refines recommendations through iterative evaluation. It is built as an extension of the original CodePath AI 110 “Applied AI Music Recommender System” from Modules 1-3.

The original project was a content-based recommendation simulator that represented songs and listener preferences as structured data, scored songs by genre, mood, energy, tempo, and other features, and ranked matching tracks for predefined user profiles. The current agentic version advances that foundation by introducing natural language understanding, API-driven retrieval, an evaluation loop, and self-correction to improve alignment with user intent.

This system is relevant because it demonstrates practical experience with modern AI architecture: clean separation of deterministic scoring logic, language-model-driven interpretation, external data retrieval, and transparent decision tracing. It is also relevant for users because it shows how AI can make recommendation systems more intuitive and trustworthy by accepting conversational requests, adapting to feedback, and surfacing results that are easier to understand and refine. The design highlights the broader impacts of agentic systems: improving human-machine collaboration, reducing the risk of opaque or misleading recommendations, and enabling more responsible, user-centered AI behavior.

---

## Documentation Coverage (Rubric Section 3)

This README is organized to satisfy the required project documentation items:

- **Original project and extension summary:** Covered in `## Project Summary`, including a 2-3 sentence description of the Modules 1-3 baseline and how this final system extends it.
- **Title and summary:** Provided at the top of this document with project purpose and impact.
- **Architecture overview:** Covered in `## How The System Works` and the diagram sections. The latest full system diagram source is stored at `assets/system_diagram.mmd`.
- **Setup instructions:** Covered in `## Getting Started` with environment, dependency installation, and run commands.
- **Sample interactions:** Included in `## Terminal Output — Three User Profiles` and `### Sample Interactions` with concrete commands and expected outputs.
- **Design decisions and trade-offs:** Covered in `## Design Decisions`.
- **Testing summary:** Covered in `### Running Tests`, `### Reliability and Evaluation`, and `### Testing Summary` with current test command and results.
- **Reflection:** Covered in `## Reflection` and `### Responsible AI Reflection`.

---

## How The System Works

This system is designed as an agentic music recommender rather than a static playlist generator. A user begins with a natural language request, and the system converts that request into structured preferences, retrieves relevant music metadata, scores candidates using a deterministic engine, and evaluates whether the results match the original intent.

The workflow is:

- **Input:** a conversational query such as “I want something like Radiohead but more upbeat.”
- **Profile extraction:** the agent translates language into a structured `UserProfile` containing fields like genre, mood, energy, tempo, acoustic preference, and explicit content preference.
- **Retrieval:** the system uses the user profile to fetch and normalize music metadata from an external API, expanding the candidate pool beyond the local catalog.
- **Recommendation:** the core engine scores each candidate by comparing song attributes to the profile and ranks the best matches.
- **Evaluation:** the agent reviews the returned recommendations against the original query, assigns a quality score, and decides whether to refine the profile and retry.
- **Output:** final ranked recommendations plus a reasoning chain that explains each decision and iteration.

The main components are:

- **RecommenderAgent:** orchestrates the agentic loop, calls tools, and enforces guardrails.
- **Profile extractor:** converts natural language into structured preference data.
- **Retriever/API client:** fetches real music metadata and normalizes it for scoring.
- **Recommender engine:** deterministic scoring logic that ranks songs by feature match.
- **Evaluator:** assesses result quality and triggers self-correction when needed.

This architecture preserves a deterministic core while adding AI-powered interpretation and evaluation. The system is designed so that the recommendation logic remains transparent and testable, while the agent adds the ability to accept user intent directly and recover from weak matches.

---

## Design Decisions

This implementation was built to balance a clear recommendation model with an agentic workflow layer.

- **Deterministic scoring core:** The existing `src/recommender.py` engine remains the authoritative ranking mechanism, so recommendations are explainable and easy to validate.
- **Agent orchestration:** `src/agent.py` provides the control loop, profile extraction, candidate retrieval, evaluation, and refinement. This separates intent handling from scoring.
- **External metadata retrieval:** `src/api_client.py` and `src/retriever.py` integrate Last.fm lookup and caching, enabling the system to expand beyond the local catalog without changing the core recommender.
- **Safe fallback behavior:** If API retrieval fails, the system gracefully falls back to the local song catalog and still produces recommendations.
- **Explicit guardrails:** The agent limits iterations, validates profile values, writes structured logs, and reports clear errors when the API key is missing.

### Trade-offs

- **Heuristic interpretation instead of an LLM backend:** Profile extraction and evaluation are currently implemented with deterministic rules to keep the system self-contained and reliable. This reduces flexibility compared to a full language model, but it minimizes external dependencies and complexity.
- **Candidate selection over data enrichment:** The retrieval layer selects relevant songs based on Last.fm artist and tag lookups, but it does not fully enrich every song feature. This maintains compatibility with the existing scoring schema while still improving relevance.
- **Quality evaluation is pragmatic:** The evaluator uses keyword-driven adjustments and average score thresholds. That makes the loop predictable, although it is less nuanced than a human or learned quality model.
- **Preservation of original functionality:** The original static profile mode remains available, which ensures reproducibility and makes the system easier to demo and test.

---

### Algorithm Recipe

| Rule | Points |
|---|---|
| `song.genre == user.favorite_genre` | **+1.0** |
| `song.mood == user.favorite_mood` | **+1.0** |
| Valence proximity: `1.0 - abs(song.valence - user.target_valence)` | **0.0 – 1.0** |
| Energy proximity: `2.0 × (1.0 - abs(song.energy - user.target_energy))` | **0.0 – 2.0** |
| Danceability proximity: `1.0 - abs(song.danceability - user.target_danceability)` | **0.0 – 1.0** |
| Tempo proximity: `max(0, 1.0 - abs(song.tempo_bpm - user.target_tempo) / 100)` | **0.0 – 1.0** |
| Era match: `song.era == user.preferred_era` (only when era preference is set) | **+0.5** |
| Explicit fit: `+0.5` unless user avoids explicit and song is explicit | **0.0 – 0.5** |
| Loudness proximity: `max(0, 1.0 - abs(song.loudness - user.target_loudness) / 15)` | **0.0 – 1.0** |
| Acoustic fit: matches `likes_acoustic` preference (threshold 0.6) | **+0.5** |
| Instrumental fit: matches `prefers_instrumental` preference (threshold 0.6) | **+0.5** |
| Speechiness proximity: `1.0 - abs(song.speechiness - user.target_speechiness)` | **0.0 – 1.0** |
| Liveness proximity: `1.0 - abs(song.liveness - user.target_liveness)` | **0.0 – 1.0** |
| **Maximum possible score** | **12.0** |
| Song below `min_popularity` | **filtered out** |
| Artist diversity penalty (duplicate artist in top-K) | **×0.8** |

**Two ranking modes:**
- `relevance` (default) — sort purely by feature-match score.
- `discovery` — applies a +15% score boost to songs with popularity below 65 before sorting, surfacing lesser-known tracks that score well but would otherwise be buried by popular ones.

The weights were chosen so that genre is the strongest discrete signal and energy carries the most continuous weight (up to 2.0). Five additional attributes — danceability, tempo, era, explicit content, and loudness — were added as scored dimensions to give the system a richer picture of each song. An artist diversity penalty ensures no single artist dominates the top-K results.

---

### Data Flow Diagram

graph TD
    %% User Input
    User((User Interface)) -->|Natural Language Query| Agent[Recommender Agent]

    %% Agentic Reasoning Loop
    subgraph Agentic_Loop [Agentic Intelligence]
        Agent --> Tool1[extract_profile]
        Tool1 -->|Structured Profile| Profile{User Profile}
        Profile --> Tool2[evaluate_results]
        Tool2 -->|Feedback/Refine| Agent
    end

    %% Retrieval Layer
    subgraph Data_Retrieval [Retrieval & Enrichment]
        Agent -->|Request Metadata| Client[API Client: Last.fm]
        Client -->|Fetch Similar Artists/Tags| Cache[(API Cache)]
        Cache -->|Normalized Data| Joiner[Data Normalizer]
    end

    %% Scoring Logic
    subgraph Scoring_Core [Recommender Engine]
        CSV[(songs.csv)] --> Joiner
        Joiner -->|Contextual Catalog| Engine[Scoring Function]
        Profile -->|Target Vectors| Engine
        Engine -->|Ranked Results| Results[Top Recommendations]
    end

    %% Output
    Results --> Tool2
    Results -->|Final Selection + Reasoning| Output[CLI Output / Logs]
    
    %% Styling
    style Agentic_Loop fill:#f9f,stroke:#333,stroke-width:2px
    style Scoring_Core fill:#bbf,stroke:#333,stroke-width:2px
    style Data_Retrieval fill:#dfd,stroke:#333,stroke-width:2px

---

### Expected Biases

- **Genre over-prioritization.** A genre match alone (2.0 pts) outscores a perfect mood + energy combination (1.0 + 1.0 = 2.0). A song that fits the vibe but belongs to the wrong genre will always rank at or below genre-matched songs, even if it would be a better listening experience.
- **Exact-string genre/mood matching.** "indie pop" and "pop" are treated as completely different genres, so similar-sounding categories receive no partial credit. This could cause the system to miss relevant songs.
- **Small catalog amplifies genre gaps.** With only 18 songs, some genres (e.g., blues, classical, country) have just one entry. A user who prefers those genres will receive genre-match points for at most one song, making energy/mood proximity the only differentiator for the rest of the list.
- **Popularity filter skews toward mainstream.** Setting `min_popularity` too high could silently exclude niche but well-matched songs (e.g., the ambient and classical tracks with popularity 54–59).

---

## Terminal Output — Three User Profiles

Run with `python -m src.main` from the project root.

### Profile 1 — Happy Pop Listener (relevance mode)
```
Profile: Happy Pop Listener  |  Mode: relevance
====================================================================================
 #   Title                  Artist            Score  Top Reasons
------------------------------------------------------------------------------------
 1   Sunrise City           Neon Echo        11.780  genre match (+1.0)  |  mood match (+1.0)  |  valence proximity (+0.96)
 2   Rooftop Lights         Indigo Parade    10.570  mood match (+1.0)  |  valence proximity (+0.99)  |  energy proximity (+1.92)
 3   Gym Hero               Max Pulse        10.370  genre match (+1.0)  |  valence proximity (+0.97)  |  energy proximity (+1.74)
 4   Neon Pulse             Circuit Drift     9.180  valence proximity (+0.92)  |  energy proximity (+1.76)  |  danceability proximity (+0.8)
 5   Gold Chain Anthem      Cipher Kings      9.040  valence proximity (+0.93)  |  energy proximity (+1.86)  |  danceability proximity (+0.84)
====================================================================================
```
Sunrise City (#1) wins because it is the only song that matches both genre and mood exactly. Rooftop Lights (#2) beats Gym Hero (#3) despite sharing neither the genre label nor the energy, because its valence and overall vibe are closer to what a happy listener wants — the genre weight change (from +2.0 to +1.0) made this possible.

---

### Profile 2 — Hip-Hop Fan (relevance mode)
```
Profile: Hip-Hop Fan  |  Mode: relevance
====================================================================================
 #   Title                  Artist            Score  Top Reasons
------------------------------------------------------------------------------------
 1   Gold Chain Anthem      Cipher Kings     11.890  genre match (+1.0)  |  mood match (+1.0)  |  valence proximity (+0.97)
 2   Gym Hero               Max Pulse         9.120  valence proximity (+0.93)  |  energy proximity (+1.84)  |  danceability proximity (+0.98)
 3   Sunrise City           Neon Echo         9.100  valence proximity (+0.86)  |  energy proximity (+1.94)  |  danceability proximity (+0.89)
 4   Rooftop Lights         Indigo Parade     8.900  valence proximity (+0.89)  |  energy proximity (+1.82)  |  danceability proximity (+0.92)
 5   Neon Pulse             Circuit Drift     8.800  valence proximity (+0.82)  |  energy proximity (+1.86)  |  danceability proximity (+0.95)
====================================================================================
```
Gold Chain Anthem is the only song that matches both hip-hop and confident mood, giving it a decisive lead. Positions 2–5 are decided by energy, danceability, and valence — all of which favor high-energy dance tracks, which is consistent with a hip-hop fan's preferences.

---

### Profile 3 — Acoustic Chill Listener (relevance mode)
```
Profile: Acoustic Chill Listener  |  Mode: relevance
====================================================================================
 #   Title                  Artist            Score  Top Reasons
------------------------------------------------------------------------------------
 1   Campfire Fable         The Hollow Oaks  11.720  genre match (+1.0)  |  mood match (+1.0)  |  valence proximity (+0.98)
 2   Wildflower Waltz       Creek & Stone     9.850  mood match (+1.0)  |  valence proximity (+0.92)  |  energy proximity (+1.8)
 3   Coffee Shop Stories    Slow Stereo       9.510  valence proximity (+0.89)  |  energy proximity (+1.94)  |  danceability proximity (+0.99)
 4   Rainy Porch Blues      Delta Hollow      9.030  valence proximity (+0.62)  |  energy proximity (+1.92)  |  danceability proximity (+0.9)
 5   Focus Flow             LoRoom            8.140  valence proximity (+0.99)  |  energy proximity (+2.0)  |  danceability proximity (+0.95)
====================================================================================
```
Campfire Fable leads with both a genre match (folk) and mood match (nostalgic), plus a near-perfect valence. The rest of the list shifts toward acoustic-friendly, low-energy tracks — a clear contrast from the pop and hip-hop profiles, demonstrating that the system genuinely adapts to different listener types.

---

### Ranking Mode Comparison — Acoustic Chill Listener
```
Profile: Acoustic Chill Listener  |  Mode: discovery
====================================================================================
 #   Title                  Artist            Score  Top Reasons
------------------------------------------------------------------------------------
 1   Campfire Fable         The Hollow Oaks  13.478  genre match (+1.0)  |  mood match (+1.0)  |  valence proximity (+0.98)
 2   Wildflower Waltz       Creek & Stone    11.327  mood match (+1.0)  |  valence proximity (+0.92)  |  energy proximity (+1.8)
 3   Coffee Shop Stories    Slow Stereo      10.936  valence proximity (+0.89)  |  energy proximity (+1.94)  |  danceability proximity (+0.99)
 4   Rainy Porch Blues      Delta Hollow     10.384  valence proximity (+0.62)  |  energy proximity (+1.92)  |  danceability proximity (+0.9)
 5   Focus Flow             LoRoom            9.361  valence proximity (+0.99)  |  energy proximity (+2.0)  |  danceability proximity (+0.95)
====================================================================================
```
Discovery mode boosts every song in the catalog with popularity below 65. All five results here have popularity < 65, so they all receive the +15% boost. The ranking order stays the same because they all qualify, but a song like Campfire Fable (popularity 61) now surfaces above popular alternatives that would otherwise crowd it out.

---

## Getting Started

## Screenshots
![alt text](<Screenshot 2026-04-14 173913.png>)
![alt text](<Screenshot 2026-04-14 175846.png>) 
![alt text](<Screenshot 2026-04-14 175856.png>)

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate       # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a local environment file:

   ```bash
   copy .env.example .env        # Windows
   cp .env.example .env          # Mac or Linux
   ```

4. Add your Last.fm API key to `.env`:

   ```text
   LASTFM_API_KEY=your_lastfm_api_key_here
   ```

5. Run the default CLI mode (static profiles):

   ```bash
   python -m src.main
   ```

6. Run the agentic Last.fm mode:

   ```bash
   python -m src.main --agent "I want something like Radiohead but more upbeat"
   ```

7. Run a specific predefined profile:

   ```bash
   python -m src.main --profile "Happy Pop Listener" --mode discovery --k 5
   ```

### Sample Interactions

#### Example 1 — Static profile recommendation

```bash
python -m src.main --profile "Happy Pop Listener" --mode relevance --k 5
```

Expected output:

- A formatted recommendation table for the `Happy Pop Listener` profile
- Top reasons like `genre match`, `mood match`, `valence proximity`, and `energy proximity`

#### Example 2 — Agentic natural language query

```bash
python -m src.main --agent "I want something like Radiohead but more upbeat"
```

Expected output:

- A recommendation table labeled `Agent Query`
- A reasoning chain describing profile extraction, evaluation, and any refinements
- A log entry appended to `logs/agent.log`

#### Example 3 — Discovery mode for a fixed profile

```bash
python -m src.main --profile "Acoustic Chill Listener" --mode discovery --k 5
```

Expected output:

- A ranked table with acoustic, nostalgic, low-energy songs
- Discovery boost details in the recommendation reasoning

### Running Tests

Run the full test suite with:

```bash
python -m pytest -q
```

Using `python -m pytest` ensures the `src` package is resolved consistently across environments.
You can add more tests in `tests/test_recommender.py`.

### Reliability and Evaluation

- **Automated tests:** 30 tests pass, covering the recommender engine, retrieval layer, agent profile extraction, inference adapters, fallback behavior, and edge-case scoring.
- **Quality scoring:** The agent computes an internal `quality_score` on a 1–10 scale and uses it to decide whether to refine recommendations.
- **Logging and error handling:** The agent writes structured logs to `logs/agent.log`, and the retrieval layer falls back to the local catalog when Last.fm calls fail.
- **Human review:** The README includes example outputs and reasoning chains so a reviewer can compare model decisions to expected behavior.

Summary: 30 out of 30 tests passed in the current suite; the AI is reliable for the implemented recommendation, retrieval, and agent orchestration cases, with validation rules and log tracing supporting debugging when external context is missing.

### Testing Summary

- **What worked:** The deterministic recommendation engine and agent orchestration were both stable, and the current suite validates sorting, popularity filtering, discovery boosts, artist diversity penalties, candidate retrieval fallback behavior, and profile refinement.
- **What didn’t:** The initial test setup had import-path issues and missing environment dependencies (`requests`), which were resolved by adding `src/__init__.py`, a `pytest.ini` config, and installing the required packages.
- **What I learned:** Reliable testing for an AI-enhanced system requires both deterministic unit coverage and environment sanity checks. Making the package importable and covering API fallback behavior early prevents the agent logic from failing silently when external metadata is unavailable.

---

## Experiments You Tried

- **Weight sensitivity test:** Reduced the genre bonus from +2.0 to +1.0 and doubled the energy bonus to max 2.0. This caused "Rooftop Lights" (indie pop, happy mood) to correctly outrank "Gym Hero" (pop, intense mood) for a happy pop listener. The original weights were mathematically valid but musically wrong.
- **Adversarial profile:** Tested a user who wanted high energy (0.9) and a sad mood simultaneously. Because no song in the catalog is both high-energy and sad, the system recommended high-energy songs with the wrong mood. Energy proximity dominated when mood matching had nothing to latch onto.
- **Impossible popularity floor:** Set `min_popularity` to 100. Every song was filtered out and the system returned an empty list with no warning — a silent failure that would confuse a real user.
- **Unknown genre test:** Set the genre to "jazz." Coffee Shop Stories became the permanent #1 regardless of any other preference, because it was the only jazz song in the catalog. Better scoring logic cannot compensate for a catalog with no variety.

---

## Limitations and Risks

- **Limited catalog breadth:** 48 songs across 25 genres still leaves uneven representation, with 9 genres having only one song. A user who prefers an underrepresented genre (for example, some niche styles with a single entry) may see that track repeatedly at the top regardless of broader preference fit.
- **Genre label brittleness:** Genre matching uses exact string equality. "Indie pop" and "pop" score as completely different, even though they describe nearly identical music. Users whose preferred genre is labeled slightly differently in the catalog receive systematically worse recommendations.
- **Artist diversity is post-hoc:** The artist penalty is applied after scoring, not during. A catalog dominated by one artist would still surface that artist's top song at #1 before the penalty kicks in for later results.
- **No listening history:** The system treats every session as a blank slate. It cannot learn that a user always skips high-liveness songs or always replays tracks with valence above 0.8.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

This project reinforced that applied AI is about balancing technical design with practical problem-solving. Building a recommender system is only part of the work; the more important challenge is ensuring that the system’s behavior is understandable, testable, and aligned with the user’s intent.

Key lessons learned:

- **Feature weighting is the core decision.** The system structure can be simple, but the scoring weights determine whether recommendations feel correct. A small change in genre or energy weighting can significantly alter the output.
- **Explainability improves trust.** Explicit explanation text for recommendations makes it much easier to see why a song was chosen and whether the result matches the request.
- **Handling edge cases is critical.** Missing API responses, fallback candidate generation, and exact string matching showed how fragile a recommender can be without robust fallback logic.
- **Iterative validation is necessary.** The project evolved through repeated testing and refinement rather than by assuming the first design was sufficient.
- **Practical solutions are often the best choice.** For this project, a heuristic-driven agent with clear fail-safes was more dependable than an overly complex black-box approach.

Overall, this work showed me that effective AI projects require engineering discipline, careful validation, and a strong focus on making system behavior visible and manageable. That combination is what makes a recommendation system useful, not just technically interesting.

### Responsible AI Reflection

- **Limitations and biases:** The system is limited by a small, hand-curated song catalog, exact-string genre and mood matching, and a popularity threshold that can favor mainstream tracks. It also lacks personalized listening history and relies on heuristic profile inference rather than learned taste models, so it can misinterpret nuanced user intent.
- **Potential misuse:** This AI could be misused if users assume it represents a broad personal taste profile or if it is relied on to recommend content for sensitive contexts. To prevent that, the system includes explicit reasoning output, fallback behavior, and clear error handling, and it should be presented as a prototype with known constraints rather than a finished personalization engine.
- **Testing surprises:** The reliability tests showed that environment and import setup were the first real failure mode, not the scoring logic. It was also surprising how much the system’s behavior depended on small weight changes and how exact genre labels could shift recommendations unexpectedly.
- **AI collaboration:** I worked with an AI coding assistant to structure the README, define tests, and refine responsibilities. A helpful suggestion was adding a `pytest.ini` import fix; a flawed suggestion was an early test fixture that passed the wrong song object type into the recommender, which required correcting the test structure.



