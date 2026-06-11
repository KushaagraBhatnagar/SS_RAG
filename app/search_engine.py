import logging
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("search_engine")

class SearchEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern to avoid reloading heavy models into RAM on every query."""
        if cls._instance is None:
            cls._instance = super(SearchEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Initialize Qdrant Client
        mode = settings.QDRANT_MODE.lower()
        logger.info(f"Connecting Search Engine to Qdrant using mode '{mode}'...")
        if mode == "local":
            self.qdrant_client = QdrantClient(path="qdrant_local_data")
        elif mode == "cloud" or settings.QDRANT_API_KEY:
            url = settings.QDRANT_HOST if settings.QDRANT_HOST.startswith("http") else f"https://{settings.QDRANT_HOST}"
            self.qdrant_client = QdrantClient(url=url, api_key=settings.QDRANT_API_KEY)
        else:
            self.qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

        # Initialize Models
        logger.info(f"Loading Bi-Encoder model: {settings.BI_ENCODER_MODEL}...")
        self.bi_encoder = SentenceTransformer(
            settings.BI_ENCODER_MODEL, 
            cache_folder=settings.MODEL_CACHE_DIR
        )

        logger.info(f"Loading Cross-Encoder model: {settings.CROSS_ENCODER_MODEL}...")
        self.cross_encoder = CrossEncoder(
            settings.CROSS_ENCODER_MODEL, 
            cache_folder=settings.MODEL_CACHE_DIR
        )

        self._initialized = True
        logger.info("Search Engine successfully initialized and models loaded!")

    def stage1_search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Stage 1: Broad Recall via Bi-Encoder vector search against Qdrant ANN index."""
        # 1. Encode query
        query_vector = self.bi_encoder.encode(query, convert_to_numpy=True).tolist()

        # 2. Search Qdrant with custom HNSW search params
        query_response = self.qdrant_client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
            search_params=models.SearchParams(
                hnsw_ef=64  # Custom ef_search value
            )
        )

        # 3. Format result candidates
        candidates = []
        for hit in query_response.points:
            candidates.append({
                "parent_asin": hit.payload.get("parent_asin"),
                "title": hit.payload.get("title"),
                "description": hit.payload.get("description"),
                "reviews": hit.payload.get("reviews", []),
                "stage1_score": float(hit.score)
            })
        
        return candidates

    def stage2_rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        """Stage 2: High Precision Re-ranking using Cross-Encoder."""
        if not candidates:
            return []

        # 1. Prepare evaluation pairs [query, doc_context]
        # Context is built exactly the same as in indexing
        pairs = []
        for cand in candidates:
            doc_context = f"Title: {cand['title']} | Description: {cand['description']}"
            pairs.append((query, doc_context))

        # 2. Predict scores with Cross-Encoder
        scores = self.cross_encoder.predict(pairs, convert_to_numpy=True)

        # 3. Associate scores with candidates
        for idx, score in enumerate(scores):
            candidates[idx]["re_ranked_score"] = float(score)

        # 4. Sort by Cross-Encoder score (highest first) and return top N
        sorted_candidates = sorted(candidates, key=lambda x: x["re_ranked_score"], reverse=True)
        return sorted_candidates[:top_n]

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Orchestrate the 2-Stage Retrieval pipeline."""
        # Retrieve K=100 candidate items (Stage 1)
        stage1_candidates = self.stage1_search(query, top_k=100)
        
        if not stage1_candidates:
            logger.info(f"No results found for query: '{query}' in Stage 1 search.")
            return []

        # Re-rank candidates down to top N (Stage 2)
        reranked_results = self.stage2_rerank(query, stage1_candidates, top_n=top_k)
        
        logger.info(f"Search query '{query}' completed: Stage 1 retrieved {len(stage1_candidates)} items, Stage 2 returned top {len(reranked_results)} items.")
        return reranked_results
