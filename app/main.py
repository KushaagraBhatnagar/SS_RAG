import logging
import requests
import asyncio # Needed for startup / thread calls
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.search_engine import SearchEngine
from app.synthesizer import ReviewSynthesizer


# Configure logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api_gateway")

# Initialize FastAPI app
app = FastAPI(
    title="Amazon-Scale Semantic Search & RAG Engine",
    description="High-throughput search engine using a two-stage retriever (Bi-Encoder + Cross-Encoder) and LLM-synthesized reviews.",
    version="1.0.0"
)

# In-memory singletons, loaded lazily at startup
search_engine: Optional[SearchEngine] = None
synthesizer: Optional[ReviewSynthesizer] = None

@app.on_event("startup")
def startup_event():
    """Load the models and database connection on startup to avoid request-time latency."""
    global search_engine, synthesizer
    logger.info("Starting up FastAPI application and loading ML models...")
    try:
        search_engine = SearchEngine()
        synthesizer = ReviewSynthesizer()
        logger.info("Startup complete. All models and services successfully loaded.")
    except Exception as e:
        logger.critical(f"Failed to initialize services during startup: {e}", exc_info=True)

@app.get("/", response_class=HTMLResponse, summary="Serve Search UI Page")
async def serve_ui():
    """Serves the semantic search client interface."""
    try:
        import os
        template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
        with open(template_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error loading HTML template: {e}")
        return HTMLResponse(content="<h1>Error loading index.html template</h1>", status_code=500)

class SearchRequest(BaseModel):
    query: str = Field(..., description="The semantic search query string.", min_length=1)
    top_k: int = Field(10, description="Number of re-ranked products to return.", ge=1, le=100)

class SearchResult(BaseModel):
    product_id: str
    title: str
    re_ranked_score: float
    review_summary: Optional[str] = None

@app.post(
    "/search", 
    response_model=List[SearchResult], 
    status_code=status.HTTP_200_OK,
    summary="Execute two-stage semantic search and synthesize reviews for the top product"
)
async def search_products(request: SearchRequest):
    """
    Search endpoint that performs a two-stage retrieval (Bi-Encoder + Cross-Encoder re-ranking),
    then triggers LLM review synthesis on the single top result.
    """
    if search_engine is None or synthesizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="ML models or Database components are still loading."
        )

    try:
        logger.info(f"Received search request: query='{request.query}', top_k={request.top_k}")
        
        # 1. Execute Two-Stage Search (Recall + Re-ranking)
        # run_in_executor/to_thread if search uses CPU-heavy models
        results = await asyncio.to_thread(search_engine.search, request.query, request.top_k)
        
        if not results:
            return []

        # 2. Format search results and synthesize reviews for the TOP (first) result only
        formatted_results: List[SearchResult] = []
        
        for idx, item in enumerate(results):
            summary = None
            if idx == 0:
                # Top product gets review synthesis
                logger.info(f"Synthesizing reviews for top product: {item['parent_asin']}")
                summary = await synthesizer.synthesize_reviews(item["title"], item["reviews"])
            
            formatted_results.append(SearchResult(
                product_id=item["parent_asin"],
                title=item["title"],
                re_ranked_score=item["re_ranked_score"],
                review_summary=summary
            ))

        return formatted_results

    except Exception as e:
        logger.error(f"Error handling search request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the semantic search."
        )

@app.get(
    "/health", 
    status_code=status.HTTP_200_OK,
    summary="Check database and service connectivity"
)
async def health_check():
    """
    Health check endpoint verifying:
    1. Connection status to the Qdrant vector database.
    2. Connection to the LLM (Ollama or Cloud provider).
    """
    health_status: Dict[str, Any] = {
        "status": "healthy",
        "qdrant_connected": False,
        "llm_connected": False
    }
    
    if search_engine is None:
        health_status["status"] = "degraded"
        health_status["message"] = "Search engine is not initialized yet."
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=health_status)

    # 1. Check Qdrant Connection
    try:
        search_engine.qdrant_client.get_collections()
        health_status["qdrant_connected"] = True
    except Exception as e:
        logger.error(f"Healthcheck failed Qdrant connection check: {e}")
        health_status["status"] = "unhealthy"

    # 2. Check LLM Connection
    try:
        if settings.USE_CLOUD_API or settings.LLM_PROVIDER != "ollama":
            # Simple ping to cloud service, or assume true if configs exist
            if settings.LLM_API_KEY:
                health_status["llm_connected"] = True
        else:
            # Check if local Ollama responds
            resp = await asyncio.to_thread(requests.get, settings.OLLAMA_BASE_URL, timeout=2)
            if resp.status_code == 200:
                health_status["llm_connected"] = True
    except Exception as e:
        logger.error(f"Healthcheck failed LLM connection check: {e}")
        if health_status["status"] != "unhealthy":
            health_status["status"] = "degraded"

    if health_status["status"] == "unhealthy":
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=health_status)
    
    return health_status
