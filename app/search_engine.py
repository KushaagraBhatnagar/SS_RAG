import logging
import os
import requests
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models

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
            self.qdrant_client = QdrantClient(url=url, api_key=settings.QDRANT_API_KEY, prefer_grpc=False)
        else:
            self.qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, prefer_grpc=False)

        # Initialize Models conditionally based on deployment mode (saves RAM on Render Free tier)
        if not settings.USE_CLOUD_API:
            # Heavy ML imports loaded lazily only when executing locally
            import torch
            import gc
            from sentence_transformers import SentenceTransformer, CrossEncoder

            # Limit threads to reduce memory overhead and CPU thrashing
            torch.set_num_threads(1)
            
            logger.info(f"Loading Bi-Encoder model locally: {settings.BI_ENCODER_MODEL}...")
            self.bi_encoder = SentenceTransformer(
                settings.BI_ENCODER_MODEL, 
                cache_folder=settings.MODEL_CACHE_DIR
            )
            logger.info("Applying dynamic quantization to Bi-Encoder...")
            self.bi_encoder = torch.quantization.quantize_dynamic(
                self.bi_encoder, {torch.nn.Linear}, dtype=torch.qint8
            )

            logger.info(f"Loading Cross-Encoder model locally: {settings.CROSS_ENCODER_MODEL}...")
            self.cross_encoder = CrossEncoder(
                settings.CROSS_ENCODER_MODEL, 
                cache_folder=settings.MODEL_CACHE_DIR
            )
            logger.info("Applying dynamic quantization to Cross-Encoder...")
            self.cross_encoder.model = torch.quantization.quantize_dynamic(
                self.cross_encoder.model, {torch.nn.Linear}, dtype=torch.qint8
            )

            # Force garbage collection to release temporary memory buffers
            gc.collect()
            logger.info("Search Engine successfully initialized and local models loaded!")
        else:
            logger.info("USE_CLOUD_API is True. Skipping loading of local models to conserve memory.")
            self.bi_encoder = None
            self.cross_encoder = None
            logger.info("Search Engine successfully initialized in Cloud API mode!")

        self._initialized = True

    def _get_embedding_cloud(self, query: str) -> List[float]:
        """Fetch 384-dimensional query embedding from Hugging Face Inference API."""
        api_url = f"https://api-inference.huggingface.co/models/{settings.BI_ENCODER_MODEL}"
        
        # Determine API key/token
        headers = {}
        hf_token = os.getenv("HF_TOKEN") or settings.LLM_API_KEY
        if hf_token and (hf_token.startswith("hf_") or len(hf_token) > 20):
            headers["Authorization"] = f"Bearer {hf_token}"
            
        try:
            logger.info(f"Querying cloud embedding API: {api_url}...")
            response = requests.post(api_url, headers=headers, json={"inputs": query}, timeout=10)
            
            # Handle model loading state (Hugging Face serverless APIs return 503 while loading a model)
            if response.status_code == 503:
                logger.warning("Hugging Face model is loading. Retrying in 3 seconds...")
                import time
                time.sleep(3)
                response = requests.post(api_url, headers=headers, json={"inputs": query}, timeout=10)
                
            response.raise_for_status()
            result = response.json()
            
            # The response can be a list of floats (for a single input) or a list of list of floats
            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], list):
                    return result[0]
                return result
            raise ValueError(f"Unexpected API response type: {type(result)}")
        except Exception as e:
            logger.error(f"Error generating cloud embedding via Hugging Face: {e}")
            # Return a zero vector of correct dimensions (384) to avoid crashing the pipeline
            return [0.0] * 384

    def _get_rerank_scores_cloud(self, query: str, pairs: List[Tuple[str, str]]) -> Optional[List[float]]:
        """Fetch rerank scores from Hugging Face Inference API for Cross-Encoder."""
        api_url = f"https://api-inference.huggingface.co/models/{settings.CROSS_ENCODER_MODEL}"
        
        headers = {}
        hf_token = os.getenv("HF_TOKEN") or settings.LLM_API_KEY
        if hf_token and (hf_token.startswith("hf_") or len(hf_token) > 20):
            headers["Authorization"] = f"Bearer {hf_token}"

        # Build payload as list of text/text_pair dictionaries
        payload = {
            "inputs": [
                {"text": pair[0], "text_pair": pair[1]}
                for pair in pairs
            ]
        }

        try:
            logger.info(f"Querying cloud rerank API for {len(pairs)} pairs: {api_url}...")
            response = requests.post(api_url, headers=headers, json=payload, timeout=12)
            
            if response.status_code == 503:
                logger.warning("Hugging Face reranker model is loading. Retrying in 4 seconds...")
                import time
                time.sleep(4)
                response = requests.post(api_url, headers=headers, json=payload, timeout=12)
                
            response.raise_for_status()
            result = response.json()
            
            # Parse Hugging Face output format
            # Hugging Face usually returns a list of list of dicts or list of dicts.
            # Example text-classification output:
            # [[{"label": "LABEL_0", "score": 0.85}], [{"label": "LABEL_0", "score": 0.12}]]
            # Or if it's a regression model, it can be a list of dicts or list of floats.
            scores = []
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
                        scores.append(float(item[0].get("score", 0.0)))
                    elif isinstance(item, dict):
                        scores.append(float(item.get("score", 0.0)))
                    elif isinstance(item, (int, float)):
                        scores.append(float(item))
                    else:
                        raise ValueError(f"Unexpected item type: {type(item)}")
                return scores
            raise ValueError(f"Unexpected API response type: {type(result)}")
        except Exception as e:
            logger.error(f"Error querying cloud reranker via Hugging Face: {e}")
            return None

    def stage1_search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Stage 1: Broad Recall via vector search against Qdrant ANN index."""
        # 1. Encode query
        if not settings.USE_CLOUD_API:
            query_vector = self.bi_encoder.encode(query, convert_to_numpy=True).tolist()
        else:
            query_vector = self._get_embedding_cloud(query)

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
        pairs = []
        for cand in candidates:
            doc_context = f"Title: {cand['title']} | Description: {cand['description']}"
            pairs.append((query, doc_context))

        # 2. Predict scores
        scores = None
        if not settings.USE_CLOUD_API:
            scores = self.cross_encoder.predict(pairs, convert_to_numpy=True)
        else:
            scores = self._get_rerank_scores_cloud(query, pairs)

        # 3. Associate scores with candidates
        if scores is not None and len(scores) == len(candidates):
            for idx, score in enumerate(scores):
                candidates[idx]["re_ranked_score"] = float(score)
        else:
            logger.warning("Reranking scores unavailable. Falling back to Stage 1 search scores.")
            for cand in candidates:
                cand["re_ranked_score"] = cand["stage1_score"]

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

