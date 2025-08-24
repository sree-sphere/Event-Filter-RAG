import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import math
import re
from collections import defaultdict
from contextlib import asynccontextmanager
import traceback
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from json import JSONDecodeError

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_POSTS_TO_RETRIEVE = 50
    TOP_POSTS_TO_RETURN = 5
    PROXIMITY_THRESHOLD_KM = 15 # km for 'nearby' proximity
    RECENCY_WEIGHT = 0.25
    PROXIMITY_WEIGHT = 0.35
    SEMANTIC_WEIGHT = 0.35
    POPULARITY_WEIGHT = 0.15

@dataclass
class Post:
    id: int = 0
    title: str = ""
    tags: List[str] = field(default_factory=list)
    location: str = ""
    coordinates: Dict[str, float] = field(default_factory=lambda: {"lat": 0.0, "lng": 0.0})
    date: str = ""  # expected "yyyy-mm-dd"
    time: str = ""  # human readable time string
    description: str = ""
    category: str = ""
    popularity_score: float = 0.0

class QueryRequest(BaseModel):
    query: str
    user_location: Optional[str] = "Anna Nagar"

class QueryResponse(BaseModel):
    summary: str
    top_posts: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class LocationMatcher:
    """Location matching with better adjacency support"""
    LOCATION_COORDS = {
        "anna nagar": {"lat": 13.0850, "lng": 80.2101},
        "t nagar": {"lat": 13.0418, "lng": 80.2341},
        "besant nagar": {"lat": 12.9988, "lng": 80.2666},
        "velachery": {"lat": 12.9755, "lng": 80.2201},
        "adyar": {"lat": 13.0067, "lng": 80.2206},
        "chromepet": {"lat": 12.9516, "lng": 80.1462},
        "mylapore": {"lat": 13.0338, "lng": 80.2619},
        "guindy": {"lat": 13.0103, "lng": 80.2209},
        "porur": {"lat": 13.0382, "lng": 80.1562},
        "tambaram": {"lat": 12.9249, "lng": 80.1000},
        "koyambedu": {"lat": 13.0732, "lng": 80.1986},
        "kingsley": {"lat": 13.0890, "lng": 80.2140}
    }

    # adjacency map
    DEFAULT_ADJACENCY = {
        "t nagar": ["mylapore", "guindy", "adyar"],
        "anna nagar": ["koyambedu", "kingsley"],
        "besant nagar": ["mylapore", "adyar"],
        "velachery": ["tambaram", "guindy"],
        "mylapore": ["t nagar", "besant nagar", "adyar"],
        "adyar": ["t nagar", "besant nagar", "mylapore"],
        "guindy": ["t nagar", "velachery"]
    }
    
    @classmethod
    def load_adjacency(cls, path: str = "adjacency.json") -> Dict[str, List[str]]:
        """Try load adjacency mapping from file, else use default"""
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {k.lower(): [v.lower() for v in vals] for k, vals in data.items()}
            except Exception:
                return {k: [v.lower() for v in vals] for k, vals in cls.DEFAULT_ADJACENCY.items()}
        else:
            return {k: [v.lower() for v in vals] for k, vals in cls.DEFAULT_ADJACENCY.items()}
    
    @classmethod
    def extract_location(cls, query: str) -> Optional[str]:
        """Extract location from query text using word boundaries and greedy check"""
        query_lower = query.lower()
        # direct word-boundary matches first
        for location in cls.LOCATION_COORDS.keys():
            if re.search(rf"\b{re.escape(location)}\b", query_lower):
                return location
        # handle around/in patterns with a capture
        patterns = [
            r"(?:around|near|in)\s+([a-z\s]+?)(?:\s|$|today|tonight|this)",
            r"(?:around|near|in)\s+([a-z]+(?:\s[a-z]+)?)"
        ]
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                candidate = match.group(1).strip()
                for location in cls.LOCATION_COORDS.keys():
                    if re.search(rf"\b{re.escape(location)}\b", candidate):
                        return location
        return None
    
    @classmethod
    def get_coordinates(cls, location: str) -> Optional[Dict[str, float]]:
        """Get coordinates for a location"""
        if not location:
            return None
        return cls.LOCATION_COORDS.get(location.lower())
    
    @classmethod
    def get_adjacent(cls, location: str, adjacency_map: Optional[Dict[str, List[str]]] = None) -> List[str]:
        """Return adjacent neighborhoods for a location (lowercased)"""
        if not adjacency_map:
            adjacency_map = cls.DEFAULT_ADJACENCY
        return adjacency_map.get(location.lower(), [])
    
    @classmethod
    def calculate_distance(cls, coord1: Dict[str, float], coord2: Dict[str, float]) -> float:
        """Calculate distance between two coordinates in km using Haversine formula"""
        lat1, lon1 = coord1.get("lat", 0.0), coord1.get("lng", 0.0)
        lat2, lon2 = coord2.get("lat", 0.0), coord2.get("lng", 0.0)
        
        R = 6371  # Earth's radius km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

class DateTimeProcessor:
    """Date/time processing with better temporal logic"""

    @staticmethod
    def parse_query_time(query: str) -> Tuple[Optional[str], Optional[str]]:
        """Lightweight return of date_ref and time_ref"""
        query_lower = query.lower()
        date_ref = None
        if "today" in query_lower:
            date_ref = "today"
        elif "tonight" in query_lower:
            date_ref = "tonight"
        elif "weekend" in query_lower or "saturday" in query_lower or "sunday" in query_lower:
            date_ref = "weekend"
        elif "this week" in query_lower:
            date_ref = "this_week"
        time_ref = None
        if "evening" in query_lower or "night" in query_lower:
            time_ref = "evening"
        elif "morning" in query_lower:
            time_ref = "morning"
        elif "afternoon" in query_lower:
            time_ref = "afternoon"
        return date_ref, time_ref

    @staticmethod
    def get_target_dates(query: str, ref_date: Optional[datetime] = None) -> Tuple[List[str], bool]:
        """Convert query time phrases into concrete target dates"""
        now = ref_date or datetime.now()
        q = query.lower()
        dates: List[str] = []
        explicit = False
        
        if "tonight" in q or "today" in q:
            dates = [now.strftime("%Y-%m-%d")]
            explicit = True
            return dates, explicit
        
        if "this weekend" in q or ("this" in q and "weekend" in q):
            saturday = now + timedelta((5 - now.weekday()) % 7)
            sunday = now + timedelta((6 - now.weekday()) % 7)
            dates = [saturday.strftime("%Y-%m-%d"), sunday.strftime("%Y-%m-%d")]
            explicit = True
            return dates, explicit

        # Handle current and next weekend queries
        if "weekend" in q and "this" not in q:
            saturday = now + timedelta((5 - now.weekday()) % 7)
            sunday = now + timedelta((6 - now.weekday()) % 7)
            next_saturday = saturday + timedelta(7)
            next_sunday = sunday + timedelta(7)
            dates = [saturday.strftime("%Y-%m-%d"), sunday.strftime("%Y-%m-%d"), next_saturday.strftime("%Y-%m-%d"), next_sunday.strftime("%Y-%m-%d")]
            explicit = False  # Less strict for general "weekend"
            return dates, explicit

        weekday_names = {"monday": 0, "tuesday":1, "wednesday":2, "thursday":3, "friday":4, "saturday":5, "sunday":6}
        for name, idx in weekday_names.items():
            if re.search(rf"\bthis\s+{name}\b", q):
                target = now + timedelta((idx - now.weekday()) % 7)
                dates = [target.strftime("%Y-%m-%d")]
                explicit = True
                return dates, explicit
        
        return [], False

    @staticmethod
    def calculate_recency_score(post_date: str, reference_date: str = None) -> float:
        """Calculate recency score with improved weighting"""
        try:
            post_date_clean = post_date.split("T")[0] if "T" in post_date else post_date
            post_dt = datetime.strptime(post_date_clean, "%Y-%m-%d")
            ref_dt = datetime.now() if not reference_date else datetime.strptime(reference_date, "%Y-%m-%d")
            days_diff = abs((post_dt - ref_dt).days)
            
            # recency scoring
            if days_diff == 0:
                return 1.0
            elif days_diff <= 1:
                return 0.9
            elif days_diff <= 3:
                return 0.8
            elif days_diff <= 7:
                return 0.6
            elif days_diff <= 14:
                return 0.4
            elif days_diff <= 30:
                return 0.2
            else:
                return 0.1
        except Exception:
            return 0.5

    @staticmethod
    def matches_time_filter(post: Post, date_list: List[str], time_ref: Optional[str]) -> bool:
        """Time filtering with better logic"""
        try:
            # if date_list provided explicitly, enforce it
            if date_list:
                post_date_str = (post.date or "").split("T")[0]
                if post_date_str not in date_list:
                    return False

            # time filtering
            if time_ref == "evening" or "tonight" in (time_ref or ""):
                post_time = (post.time or "").lower()
                evening_indicators = ["pm", "night", "evening"]
                has_evening_indicator = any(x in post_time for x in evening_indicators)
                
                if has_evening_indicator:
                    return True
                
                # Check for hours
                m = re.search(r"(\d{1,2})(?::\d{2})?\s*pm", post_time)
                if m:
                    hour = int(m.group(1))
                    if hour >= 6 or hour <= 2:  # 6PM onwards or midnight hours
                        return True
                
                # If no clear time info and it's a "tonight" query, be conservative
                if not post_time.strip() and "tonight" in (time_ref or ""):
                    return False
                    
            return True
        except Exception:
            return True

class SemanticMatcher:
    """Semantic matching for better topic-specific queries"""
    
    TOPIC_KEYWORDS = {
        "music": ["music", "concert", "live", "band", "jazz", "rock", "indie", "acoustic", "performance"],
        "food": ["food", "restaurant", "cafe", "dining", "cuisine", "feast", "festival", "potluck"],
        "sports": ["sports", "cricket", "football", "game", "match", "tournament", "fitness"],
        "entertainment": ["comedy", "theater", "movie", "show", "entertainment", "performance"],
        "social": ["meetup", "networking", "social", "community", "gathering", "storytelling"],
        "cultural": ["art", "cultural", "exhibition", "gallery", "poetry", "literature"]
    }
    
    @classmethod
    def enhance_semantic_score(cls, query: str, post: Post, base_score: float) -> float:
        """Enhance semantic score based on topic matching"""
        query_lower = query.lower()
        
        # extract topics from query
        query_topics = []
        for topic, keywords in cls.TOPIC_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                query_topics.append(topic)
        
        if not query_topics:
            return base_score
        
        # check post content for topic relevance
        post_content = f"{post.title} {' '.join(post.tags)} {post.description} {post.category}".lower()
        
        topic_boost = 0.0
        for topic in query_topics:
            topic_keywords = cls.TOPIC_KEYWORDS[topic]
            matches = sum(1 for keyword in topic_keywords if keyword in post_content)
            if matches > 0:
                topic_boost += min(0.3, matches * 0.1)  # Cap boost at 0.3
        
        return min(1.0, base_score + topic_boost)

class RAGAssistant:
    """RAG pipeline with improved filtering and response generation"""

    def __init__(self, data_path: str = "dataset.json"):
        self.config = Config()
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        try:
            if Groq and self.config.GROQ_API_KEY and "your-groq-api-key-here" not in self.config.GROQ_API_KEY:
                self.groq_client = Groq(api_key=self.config.GROQ_API_KEY)
            else:
                self.groq_client = None
        except Exception as e:
            print(f"Warning: could not initialize Groq client: {e}")
            self.groq_client = None
        
        self.location_matcher = LocationMatcher()
        self.datetime_processor = DateTimeProcessor()
        self.semantic_matcher = SemanticMatcher()
        
        # Load and process data
        self.posts = self._load_posts(data_path)
        self.embeddings, self.index = self._create_embeddings()
        
    def _safe_post_from_dict(self, item: Dict[str, Any]) -> Post:
        """Construct Post with defaults if some fields are missing"""
        try:
            return Post(
                id=int(item.get("id", 0)),
                title=str(item.get("title", "") or ""),
                tags=item.get("tags") if isinstance(item.get("tags"), list) else ([item.get("tags")] if item.get("tags") else []),
                location=str(item.get("location", "") or ""),
                coordinates=item.get("coordinates") if isinstance(item.get("coordinates"), dict) else item.get("coords", {"lat": 0.0, "lng": 0.0}),
                date=str(item.get("date", "") or ""),
                time=str(item.get("time", "") or ""),
                description=str(item.get("description", "") or ""),
                category=str(item.get("category", "") or ""),
                popularity_score=float(item.get("popularity_score", item.get("popularity", 0.0) or 0.0))
            )
        except Exception:
            return Post()
    
    def _load_posts(self, data_path: str) -> List[Post]:
        """Load posts from JSON file with robust handling"""
        if not os.path.exists(data_path):
            print(f"Data file not found at {data_path}. continuing with empty dataset.")
            return []
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                if "posts" in data and isinstance(data["posts"], list):
                    raw_posts = data["posts"]
                else:
                    lists = [v for v in data.values() if isinstance(v, list)]
                    raw_posts = lists[0] if lists else []
            elif isinstance(data, list):
                raw_posts = data
            else:
                raw_posts = []
            
            posts = []
            for item in raw_posts:
                if isinstance(item, dict):
                    posts.append(self._safe_post_from_dict(item))
            return posts
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def _create_embeddings(self) -> Tuple[np.ndarray, faiss.IndexFlatL2]:
        """Create embeddings and FAISS index"""
        texts = []
        for post in self.posts:
            title = post.title or ""
            tags = " ".join(post.tags or [])
            description = post.description or ""
            loc = post.location or ""
            text = f"{title} {tags} {description} {loc}".strip()
            if not text:
                text = "empty"
            texts.append(text)
        
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)
        
        if not texts:
            empty_embeddings = np.zeros((0, dimension), dtype='float32')
            return empty_embeddings, index
        
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        
        embeddings = embeddings.astype('float32')
        index.add(embeddings)
        
        return embeddings, index
    
    def _semantic_search(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        """Semantic search with more candidates"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')
        distances, indices = self.index.search(query_embedding, min(k, len(self.posts)))
        
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.posts) and idx >= 0:
                similarity = 1 / (1 + float(dist))
                # semantic enhancement
                enhanced_score = self.semantic_matcher.enhance_semantic_score(query, self.posts[idx], similarity)
                results.append((int(idx), enhanced_score))
        
        return results

    def _calculate_composite_score(self, post: Post, query: str, query_location: str, semantic_score: float) -> float:
        """Composite scoring with better weighting"""
        scores = {
            'semantic': semantic_score,
            'recency': self.datetime_processor.calculate_recency_score(post.date),
            'proximity': 0.5,  # default
            'popularity': min(1.0, post.popularity_score / 10.0)  # Normalize popularity
        }

        # Proximity calculation
        if query_location:
            query_coords = self.location_matcher.get_coordinates(query_location)
            if query_coords and isinstance(post.coordinates, dict):
                distance = self.location_matcher.calculate_distance(query_coords, post.coordinates)
                if distance <= 2:  # Same area
                    scores['proximity'] = 1.0
                elif distance <= 5:  # Very close
                    scores['proximity'] = 0.8
                elif distance <= self.config.PROXIMITY_THRESHOLD_KM:
                    scores['proximity'] = max(0.3, 1 - (distance / self.config.PROXIMITY_THRESHOLD_KM))
                else:
                    scores['proximity'] = 0.1  # worst case

        popularity_boost = self.config.POPULARITY_WEIGHT
        if any(word in query.lower() for word in ['trending', 'popular', 'hot', 'weekend']):
            popularity_boost *= 2  # Double popularity weight for trending queries
        
        # Weighted composite score
        composite = (
            scores['semantic'] * self.config.SEMANTIC_WEIGHT +
            scores['recency'] * self.config.RECENCY_WEIGHT +
            scores['proximity'] * self.config.PROXIMITY_WEIGHT +
            scores['popularity'] * popularity_boost
        )
        
        return composite
    
    def _filter_posts(self, posts: List[Post], query: str, 
                     query_location: str, date_list: List[str], 
                     explicit_date: bool, time_ref: Optional[str]) -> List[Post]:
        """Filtering with location adjacency support"""
        filtered_posts = []
        adjacency_map = self.location_matcher.load_adjacency()
        adjacent_locations = self.location_matcher.get_adjacent(query_location, adjacency_map) if query_location else []
        
        for post in posts:
            # Time/date filter
            if not self.datetime_processor.matches_time_filter(post, date_list, time_ref):
                continue

            # Location filtering
            if query_location:
                post_location = (post.location or "").lower()
                
                # Direct location match
                if query_location.lower() in post_location:
                    filtered_posts.append(post)
                    continue
                
                # Adjacent location match
                if any(adj_loc in post_location for adj_loc in adjacent_locations):
                    filtered_posts.append(post)
                    continue
                
                # Proximity-based filtering
                query_coords = self.location_matcher.get_coordinates(query_location)
                if query_coords and isinstance(post.coordinates, dict):
                    distance = self.location_matcher.calculate_distance(query_coords, post.coordinates)
                    if distance <= self.config.PROXIMITY_THRESHOLD_KM:
                        filtered_posts.append(post)
                        continue
            else:
                # No location specified - include all
                filtered_posts.append(post)
        
        return filtered_posts
    
    def _heuristic_enough_info(self, query: str, top_posts: List[Post]) -> Tuple[bool, str]:
        """Conservative heuristic to decide if the available posts are enough to answer.
        returns (enough: bool, reason: str)
        - For explicit dates (this weekend, this saturday, tonight, today) require at least one post matching date/time.
        - For explicit location queries require at least one post in same-location or adjacent locations.
        - Otherwise, if we have >=1 post, accept; else not enough.
        """
        date_list, explicit_date = self.datetime_processor.get_target_dates(query)
        date_ref, time_ref = self.datetime_processor.parse_query_time(query)
        query_location = self.location_matcher.extract_location(query)

        # if no candidate posts at all -> not enough
        if not top_posts:
            return False, "no candidate posts returned"

        # explicit temporal requirements
        if explicit_date:
            # require at least one post with date in date_list
            matched_by_date = []
            for p in top_posts:
                p_date = (p.date or "").split("T")[0]
                if p_date in date_list:
                    # if time_ref present and is evening/tonight, check time as well
                    if time_ref in ("evening", "night", "tonight"):
                        pt = (p.time or "").lower()
                        if any(x in pt for x in ["pm", "night", "evening"]) or re.search(r"\b(1[89]|2[0-3]|0?[6-9])\b", pt):
                            matched_by_date.append(p)
                    else:
                        matched_by_date.append(p)
            if not matched_by_date:
                return False, f"no posts match explicit date(s) {date_list}"
            # if explicit location too, check location match
            if query_location:
                loc_matches = [p for p in matched_by_date if query_location.lower() in (p.location or "").lower()]
                if loc_matches:
                    return True, "found posts matching explicit date and location"
                # allow adjacent if adjacency map says so
                adjacency = self.location_matcher.load_adjacency()
                adjacent_locs = self.location_matcher.get_adjacent(query_location, adjacency)
                adj_matches = [p for p in matched_by_date if any(adj in (p.location or "").lower() for adj in adjacent_locs)]
                if adj_matches:
                    return True, "found posts matching date in adjacent locations"
                # otherwise, still not enough for strict explicit-date+location queries
                return False, "found posts on date but none in the requested location or adjacent neighborhoods"

            return True, "found posts matching explicit date"

        # location-only explicit queries (no explicit date): prefer same-location or adjacent
        if query_location:
            loc_matches = [p for p in top_posts if query_location.lower() in (p.location or "").lower()]
            if loc_matches:
                return True, "found posts in requested location"
            adjacency = self.location_matcher.load_adjacency()
            adjacent_locs = self.location_matcher.get_adjacent(query_location, adjacency)
            adj_matches = [p for p in top_posts if any(adj in (p.location or "").lower() for adj in adjacent_locs)]
            if adj_matches:
                return True, "found posts in adjacent locations"
            # if none, but there are posts within proximity threshold, allow them as 'alternatives' but mark not enough
            return False, "no posts in requested or adjacent locations"

        # default: at least one post is enough
        if len(top_posts) >= 1:
            return True, "sufficient posts available"
        return False, "insufficient posts"

    def _generate_varied_summary(self, query: str, top_posts: List[Post]) -> str:
        """Generate varied, natural summaries and let the LLM decide if info is sufficient.
        The LLM is asked to return JSON: {'action':'answer'|'fallback','summary':'...','reason':'...'}
        If Groq LLM is unavailable or returns non-JSON, fall back to a deterministic heuristic.
        """
        # quick guard
        if top_posts is None:
            top_posts = []

        # compute context pieces
        date_list, explicit_date = self.datetime_processor.get_target_dates(query)
        date_ref, time_ref = self.datetime_processor.parse_query_time(query)
        query_location = self.location_matcher.extract_location(query)
        adjacency_map = self.location_matcher.load_adjacency() if query_location else {}

        # built short context about the top posts
        posts_context = []
        for i, post in enumerate(top_posts[:10], 1):
            posts_context.append({
                "idx": i,
                "title": post.title,
                "location": post.location,
                "date": post.date,
                "time": post.time,
                "tags": post.tags,
                "category": post.category,
                "short_desc": (post.description or "")[:140]
            })
        # flatten to text for llm consumption
        posts_text = "\n".join([f"{p['idx']}. {p['title']} - {p['location']} on {p['date']} at {p['time']} | tags: {', '.join(p['tags'])}\n   {p['short_desc']}" for p in posts_context])

        # project expectations to include in the prompt (5 points)
        project_requirements = (
            "Project acceptance criteria (judge against these):\n"
            "1) Basic local query: when user asks 'this weekend' or similar, return relevant events within the concrete dates for that phrase for the user's current calendar week.\n"
            "2) Nearby match: when user asks 'around X' or 'near X', return events in adjacent neighborhoods only if appropriate, and label them as nearby. Use adjacency map when available.\n"
            "3) No match fallback: for strict time queries like 'tonight' or 'this Saturday', if no fresh matching events exist, respond with a fallback rather than returning stale or unrelated results.\n"
            "4) Topic-specific: when query includes a topic (e.g., 'live music'), match posts by semantic/topic relevance, not just keyword presence.\n"
            "5) Repetition avoidance: produce varied, natural phrasing and avoid repeating the same sentence structure across responses."
        )

        # Decision prompt asks the llm to decide whether available context is enough.
        decision_prompt = f"""
You are a decision-maker assistant which inspects available event candidates and decides if there is enough fresh and local information to directly answer the user's query, or whether you should ask the system to fallback.

User query: "{query}"
Query location: {query_location or 'none'}
Query date targets: {date_list or 'none'} (explicit_date={explicit_date})
Time reference: {time_ref or 'none'}

Top candidate posts (up to 10):
{posts_text or 'No candidate posts.'}

Adjacency map (for the requested location) keys: {list(adjacency_map.keys())[:10]}

{project_requirements}

Task:
1) Decide whether the information above is sufficient to produce a reliable, local answer that satisfies the project acceptance criteria.
2) If sufficient, produce a concise, user-facing summary (1-3 sentences) that mentions specific events and, where relevant, labels nearby alternatives.
3) If not sufficient, produce a short fallback recommendation message explaining why (e.g., no posts on the requested date, no posts in the requested location) and optionally list safe nearby alternatives but mark them clearly as alternatives.

Output requirement (STRICT):
Return only valid JSON with exactly these keys:
{{"action": "<answer|fallback>", "summary": "<user-facing text>", "reason": "<brief reason why you chose this action>"}}

Make the output JSON parseable. Keep "summary" user-friendly and do not include any debugging or trace data in it.
"""

        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a careful local events assistant who must decide whether there is enough context to answer. If unsure, choose fallback."},
                        {"role": "user", "content": decision_prompt}
                    ],
                    model="openai/gpt-oss-120b",
                    temperature=0.2,
                    max_tokens=400
                )
                raw = response.choices[0].message.content.strip()
                # trying to parse JSON
                try:
                    decision = json.loads(raw)
                    action = decision.get("action", "fallback")
                    summary_text = decision.get("summary", "").strip()
                    reason = decision.get("reason", "")
                    if action == "answer" and summary_text:
                        return summary_text
                    else:
                        # LLM's summary or fallback
                        if summary_text:
                            return summary_text
                        return self._generate_fallback_response(query)
                except Exception:
                    # non-JSON response - fall back to heuristic answer response
                    pass
            except Exception as e:
                # LLM call failed, continue to heuristic
                print(f"LLM decision call failed: {e}")

        # No LLM or LLM couldn't be parsed -> heuristic
        enough, reason = self._heuristic_enough_info(query, top_posts)
        if not enough:
            # LLM-like fallback wording but deterministic
            fallback_msg = self._generate_fallback_response(query)
            # brief reason for transparency
            return f"{fallback_msg} (reason: {reason})"

        # for variation
        summary_templates = [
            "Here are top picks for your query: {lead}.",
            "Good finds nearby: {lead}.",
            "You might like: {lead}."
        ]
        lead_parts = [f"{p.title} ({p.location}, {p.date})" for p in top_posts[:5]]
        lead = "; ".join(lead_parts)
        return random.choice(summary_templates).format(lead=lead)

    def _generate_fallback_response(self, query: str) -> str:
        """Fallback responses with better reasoning"""
        location = self.location_matcher.extract_location(query)
        date_ref, time_ref = self.datetime_processor.parse_query_time(query)
        # Contextual fallback responses
        if date_ref and location:
            if date_ref == "tonight" or date_ref == "today":
                return f"I don't see any fresh events for tonight in {location.title()}. You might want to check out nearby areas like the adjacent neighborhoods, or look for pop-up events that might not be in my current database."
            elif "weekend" in date_ref:
                return f"No specific weekend events found for {location.title()} in my current data. Consider checking local community boards or social media for last-minute weekend activities."
        elif location:
            return f"Limited current events in {location.title()}. Try expanding to nearby areas or check back later for updated listings."
        elif date_ref:
            if date_ref == "tonight":
                return "No specific events found for tonight in the searched area. Consider checking local bars, cafes, or community centers for impromptu gatherings."
            else:
                return f"No events found for the specified time. You might want to broaden your search or check alternative dates."
        else:
            return "I couldn't find events matching your specific criteria. Try broadening your search terms or checking different locations and dates."
    
    def query(self, user_query: str, user_location: str = "Anna Nagar") -> QueryResponse:
        """Main query processing pipeline with strict temporal logic for explicit date queries"""
        # determine query location (lowercased)
        query_location = self.location_matcher.extract_location(user_query) or (user_location or "Anna Nagar").lower()
        
        # compute concrete target dates for explicit temporal phrases
        date_list, explicit_date = self.datetime_processor.get_target_dates(user_query)
        # parse time_ref for "tonight"/"evening"
        _, time_ref = self.datetime_processor.parse_query_time(user_query)
        
        # semantic search as before
        semantic_results = self._semantic_search(user_query, self.config.MAX_POSTS_TO_RETRIEVE)
        
        candidates = []
        for idx, semantic_score in semantic_results:
            if idx < 0 or idx >= len(self.posts):
                continue
            post = self.posts[idx]
            composite_score = self._calculate_composite_score(post, user_query, query_location, semantic_score)
            candidates.append((post, composite_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidate_posts = [post for post, _ in candidates]

        # adjacency expansion
        adjacency_map = LocationMatcher.load_adjacency()  # loads file or default
        adjacent_locs = self.location_matcher.get_adjacent(query_location, adjacency_map) if query_location else []
        # build locations-of-interest set for relaxed proximity decisions (not used when explicit_date enforce)
        locations_of_interest = [query_location] + adjacent_locs
        
        # apply filters - note i am passing date_list and explicit_date flag through
        filtered_posts = self._filter_posts(candidate_posts, user_query, query_location, date_list, explicit_date, time_ref)
        
        # strict policy: if user explicitly asked for specific dates and none match, return fallback
        if explicit_date and not filtered_posts:
            # create fallback summary clarifying the dates
            if len(date_list) == 1:
                checked_str = date_list[0]
            elif len(date_list) == 2:
                checked_str = f"{date_list[0]} to {date_list[1]}"
            else:
                checked_str = ", ".join(date_list)
            fallback_summary = f"I don't see any events on {checked_str} in my dataset. You might try widening the date range or checking local listings."
            # metadata to communicate reason
            metadata = {
                "query_location": query_location,
                "total_candidates": len(candidates),
                "filtered_results": 0,
                "reason": "explicit_date_no_results",
                "checked_dates": date_list
            }
            return QueryResponse(summary=fallback_summary, top_posts=[], metadata=metadata)
        
        # if no results and not explicit_date, previous fallback/expansion behavior (keeps original)
        if not filtered_posts and query_location:
            filtered_posts = candidate_posts[:5]
        
        top_posts = filtered_posts[:self.config.TOP_POSTS_TO_RETURN]
        
        # generate summary
        summary = self._generate_varied_summary(user_query, top_posts)
        
        # format top_posts_data and metadata (include distances and freshness)
        top_posts_data = []
        for post in top_posts:
            top_posts_data.append({
                "id": post.id,
                "title": post.title,
                "location": post.location,
                "date": post.date,
                "time": post.time,
                "tags": post.tags,
                "description": post.description,
                "category": post.category
            })
        
        # calculate distances if possible
        distances = []
        for p in top_posts:
            if query_location and self.location_matcher.get_coordinates(query_location) and isinstance(p.coordinates, dict):
                d = round(self.location_matcher.calculate_distance(self.location_matcher.get_coordinates(query_location), p.coordinates), 2)
            else:
                d = None
            distances.append(d)
        
        metadata = {
            "query_location": query_location,
            "total_candidates": len(candidates),
            "filtered_results": len(filtered_posts),
            "search_strategy": "semantic_proximity_recency",
            "distances_km": distances
        }

        return QueryResponse(summary=summary, top_posts=top_posts_data, metadata=metadata)

# lifespan instead of on_event startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_assistant
    rag_assistant = RAGAssistant()
    yield

app = FastAPI(title="RAG-Powered Local Post Assistant", version="1.0.0", lifespan=lifespan)
rag_assistant = None

@app.post("/query", response_model=QueryResponse)
async def query_posts(request: QueryRequest):
    """Main API endpoint for querying local posts"""
    if not rag_assistant:
        raise HTTPException(status_code=500, detail="RAG Assistant not initialized")
    
    try:
        response = rag_assistant.query(request.query, request.user_location or "Anna Nagar")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RAG Assistant is running"}

# Testing function
def test_scenarios():
    """Test the assistant with different scenarios"""
    assistant = RAGAssistant()
    
    test_cases = [
        "What's trending this weekend near Anna Nagar?",
        "Anything happening around T Nagar today?",
        "Any events near Chromepet tonight?",
        "Live music in Besant Nagar tonight",
        "What events this Saturday near Velachery?"
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"Test Case {i}: {query}")
        print("-" * 50)
        result = assistant.query(query)
        print(result.summary)
        print(f"Top Posts ({len(result.top_posts)}):")
        for j, post in enumerate(result.top_posts, 1):
            print(f"  {j}. {post['title']} - {post['location']} ({post['date']})")
        print(f"Metadata: {result.metadata}")
        print("\n")

if __name__ == "__main__":
    test_scenarios()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
