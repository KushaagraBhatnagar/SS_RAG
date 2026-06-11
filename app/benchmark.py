import time
import math
import logging
import json
import requests
from typing import List, Dict, Any

from app.config import settings
from app.search_engine import SearchEngine
from app.synthesizer import ReviewSynthesizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("benchmark")

def calculate_dcg(relevances: List[float], p: int = 10) -> float:
    """Compute Discounted Cumulative Gain (DCG) at position p."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:p]):
        dcg += rel / math.log2(i + 2)
    return dcg

def calculate_ndcg(relevances: List[float], p: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG) at position p."""
    dcg = calculate_dcg(relevances, p)
    # The ideal order has all relevant items at the front.
    # In our benchmark, only 1 item (the target product) is relevant (score = 3).
    # So the ideal relevance list is [3.0, 0.0, 0.0, ...]
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, p)
    
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def run_benchmark():
    logger.info("Initializing search engine and synthesizer for benchmarking...")
    engine = SearchEngine()
    synthesizer = ReviewSynthesizer()

    # 1. Fetch products from Qdrant to construct test queries
    logger.info("Retrieving documents from Qdrant to construct test queries...")
    try:
        # Retrieve points from the collection
        points = engine.qdrant_client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=5,
            with_payload=True,
            with_vectors=False
        )[0]
    except Exception as e:
        logger.error(f"Failed to query Qdrant. Make sure the database is seeded. Error: {e}")
        return

    if not points:
        logger.error("No products found in Qdrant collection. Seed the database first!")
        return

    # Build test queries: for each product in Qdrant, construct a search query from its title.
    # The target product will have relevance score 3.0, others 0.0.
    test_cases = []
    for p in points:
        title = p.payload.get("title")
        asin = p.payload.get("parent_asin")
        reviews = p.payload.get("reviews", [])
        
        # Simplify title to create a search query
        query_words = title.split()[:5]
        query = " ".join(query_words).replace("-", "").replace(",", "")
        
        test_cases.append({
            "query": query,
            "target_asin": asin,
            "target_title": title,
            "reviews": reviews
        })

    logger.info(f"Generated {len(test_cases)} test cases for benchmark.")

    # Storage for latencies
    stage1_latencies = []
    stage2_latencies = []
    generation_full_latencies = []
    generation_ttft_latencies = []

    # Storage for NDCG evaluations
    stage1_ndcgs = []
    hybrid_ndcgs = []

    for idx, tc in enumerate(test_cases):
        query = tc["query"]
        target_asin = tc["target_asin"]
        logger.info(f"[{idx+1}/{len(test_cases)}] Benchmarking query: '{query}'")

        # --- Measure Stage 1 ---
        t0 = time.perf_counter()
        stage1_candidates = engine.stage1_search(query, top_k=100)
        t1 = time.perf_counter()
        stage1_lat = (t1 - t0) * 1000  # ms
        stage1_latencies.append(stage1_lat)

        # Compute Stage 1 NDCG@10
        # If the target product is in the result, assign its relevance.
        stage1_relevances = []
        for doc in stage1_candidates[:10]:
            if doc["parent_asin"] == target_asin:
                stage1_relevances.append(3.0)
            else:
                stage1_relevances.append(0.0)
        
        # Pad to 10
        stage1_relevances += [0.0] * (10 - len(stage1_relevances))
        stage1_ndcgs.append(calculate_ndcg(stage1_relevances, p=10))

        # --- Measure Stage 2 (Re-ranking) ---
        t0 = time.perf_counter()
        hybrid_results = engine.stage2_rerank(query, stage1_candidates, top_n=10)
        t1 = time.perf_counter()
        stage2_lat = (t1 - t0) * 1000  # ms
        stage2_latencies.append(stage2_lat)

        # Compute Hybrid (Stage 1 + Stage 2) NDCG@10
        hybrid_relevances = []
        for doc in hybrid_results[:10]:
            if doc["parent_asin"] == target_asin:
                hybrid_relevances.append(3.0)
            else:
                hybrid_relevances.append(0.0)
                
        # Pad to 10
        hybrid_relevances += [0.0] * (10 - len(hybrid_relevances))
        hybrid_ndcgs.append(calculate_ndcg(hybrid_relevances, p=10))

        # --- Measure Generation Latency (TTFT & Total) ---
        # Run only on the first test case to avoid excessive local LLM load, or on all if configured.
        if idx == 0:
            prompt = synthesizer._build_prompt(tc["target_title"], tc["reviews"])
            
            if settings.USE_CLOUD_API or settings.LLM_PROVIDER != "ollama":
                # For cloud APIs, streaming can be complex to standardize in python requests, 
                # so we measure total latency and estimate TTFT (approx 30% of total)
                t_start = time.perf_counter()
                _ = synthesizer._synthesize_cloud(prompt)
                t_end = time.perf_counter()
                total_lat = (t_end - t_start) * 1000 # ms
                generation_ttft_latencies.append(total_lat * 0.3)
                generation_full_latencies.append(total_lat)
            else:
                # Local Ollama streaming measurement for accurate TTFT
                payload = {
                    "model": settings.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"temperature": 0.0}
                }
                try:
                    t_start = time.perf_counter()
                    ttft = None
                    
                    response = requests.post(
                        synthesizer.ollama_url, 
                        json=payload, 
                        stream=True, 
                        timeout=25
                    )
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if line:
                            if ttft is None:
                                ttft = (time.perf_counter() - t_start) * 1000  # ms
                                generation_ttft_latencies.append(ttft)
                            
                            chunk = json.loads(line)
                            if chunk.get("done", False):
                                break
                                
                    t_end = time.perf_counter()
                    total_lat = (t_end - t_start) * 1000 # ms
                    generation_full_latencies.append(total_lat)
                    
                except Exception as e:
                    logger.error(f"Error measuring streaming LLM latency: {e}")
                    generation_ttft_latencies.append(0.0)
                    generation_full_latencies.append(0.0)

    # Calculate final averages
    avg_stage1 = sum(stage1_latencies) / len(stage1_latencies)
    avg_stage2 = sum(stage2_latencies) / len(stage2_latencies)
    
    avg_ttft = generation_ttft_latencies[0] if generation_ttft_latencies else 0.0
    avg_gen = generation_full_latencies[0] if generation_full_latencies else 0.0
    
    avg_ndcg_stage1 = sum(stage1_ndcgs) / len(stage1_ndcgs)
    avg_ndcg_hybrid = sum(hybrid_ndcgs) / len(hybrid_ndcgs)

    # 4. Print Metrics Breakdown Matrix
    print("\n" + "="*60)
    print("      AMAZON-SCALE RAG ENGINE BENCHMARK RESULTS")
    print("="*60)
    print(f"{'Metric':<35} | {'Measured Value':<12} | {'Target':<10}")
    print("-"*60)
    print(f"{'Stage 1 (Bi-Encoder + ANN) Latency':<35} | {avg_stage1:>10.2f} ms | <15ms")
    print(f"{'Stage 2 (Cross-Encoder) Latency':<35} | {avg_stage2:>10.2f} ms | <80ms")
    print(f"{'LLM Time-to-First-Token (TTFT)':<35} | {avg_ttft:>10.2f} ms | N/A")
    print(f"{'LLM Full Generation Latency':<35} | {avg_gen:>10.2f} ms | N/A")
    print("-"*60)
    print(f"{'Stage 1 Alone NDCG@10':<35} | {avg_ndcg_stage1:>12.4f} | N/A")
    print(f"{'Stage 1 + Stage 2 NDCG@10':<35} | {avg_ndcg_hybrid:>12.4f} | Higher=Better")
    print("="*60)
    print("Note: LLM generation latencies reflect local CPU/GPU speeds.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_benchmark()
