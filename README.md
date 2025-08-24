1. Architecture
    - RAG-Powered Local Event Assistant
    - Layers:
    1. Data Layer: dataset.json containing posts with attributes (title, tags, coordinates, date, time, popularity_score, etc.)
    2. Semantic Layer: Sentence-BERT embeddings stored in FAISS for efficient similarity search.
    3. Filtering Layer:
        - Temporal filtering via DateTimeProcessor
        - Location matching and adjacency via LocationMatcher
        - Semantic enhancement via SemanticMatcher
    4. RAG Decision Layer:
        - Optional Groq LLM for dynamic fallback or varied summaries
        - Heuristic fallback if no LLM available
    5. API Layer: FastAPI with `/query` endpoint.

```bash
FastAPI API
   |
   v
RAGAssistant
   |-- LocationMatcher            # Extracts query location, adjacency support, distance calculations
   |-- DateTimeProcessor          # Parses query temporal info, recency scoring, time filtering
   |-- SemanticMatcher            # Topic-specific semantic enhancements
   |-- SentenceTransformer        # Embeddings for semantic search
   |-- FAISS Index                # Efficient nearest-neighbor search
   |-- Groq LLM                   # Decision-making for summary/fallback
   v
QueryResponse

```

2. Tech Stack
    - Backend: Python, FastAPI
    - Semantic Search:
        - sentence-transformers for embeddings (all-MiniLM-L6-v2)
        - faiss for vector similarity search
    - LLM: Groq API
    - Data Storage: JSON dataset (dataset.json)
    - Dependencies: `pydantic`, `uvicorn`, `numpy`, `math`, `re`

3. Prompt Strategy
For RAG/LLM assisted fallback:
    - Provide query, top posts, adjacency info, date targets.
    - Ask LLM to decide: "answer" if enough info, "fallback" if insufficient.
    - Output must be JSON with action, summary, reason.

        `{"action": "<answer|fallback>", "summary": "<user-facing text>", "reason": "<reason>"}`
    - Heuristics used when LLM unavailable:
        - Explicit date queries enforce strict date filtering
        - Explicit location queries enforce adjacency/proximity
        - Semantic relevance boosts topic-specific matches
        - Weighted scoring: semantic + recency + proximity + popularity

4. Challenges & Trade-offs
    - Temporal Filtering: Explicit queries (e.g., “tonight”, “this Saturday”) require strict matching; fallback logic needed if no events available.
    - Location Adjacency: Ensuring nearby locations are suggested without overextending to unrelated neighborhoods.
    - Semantic Relevance: Pure keyword matching missed topic intent; Sentence-BERT improved this but adds embedding computation overhead.
    - Fallback Strategy: Balancing between deterministic heuristics and LLM-based summaries to provide varied, human-like responses.
    - Popularity vs Freshness: Trending/popular events can outweigh slightly older posts; weights were tuned to prevent stale recommendations.
    - LLM Dependency: LLM improves summaries but system works fully with heuristic-only approach.