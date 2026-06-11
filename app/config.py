import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Vector Database (Qdrant) Configuration
    # Modes: "local" (in-process on-disk), "server" (local docker/port), "cloud" (qdrant cloud)
    QDRANT_MODE: str = "local"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "amazon_products"


    # Local ML Models (Sentence-Transformers)
    BI_ENCODER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Ollama Local LLM Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "phi3:mini"  # Light-weight model (3.8B), excellent for consumer hardware

    # Ingestion Configuration
    # We use McAuley-Lab/Amazon-Reviews-2023 dataset on HuggingFace
    DATASET_CATEGORY: str = "Subscription_Boxes" # Smallest category, ideal for light load
    MAX_PRODUCTS_TO_INGEST: int = 200            # Limit to keep indexing memory low
    MAX_REVIEWS_PER_PRODUCT: int = 5            # Limit reviews per product to keep RAG prompt compact

    # Hybrid Cloud/Deployment Configuration
    USE_CLOUD_API: bool = False                  # Set to True when deploying to Vercel/Render without local LLM/Qdrant
    LLM_PROVIDER: str = "ollama"                 # "ollama", "openai", "groq"
    LLM_API_KEY: Optional[str] = None
    LLM_API_URL: Optional[str] = None            # Override URL for external LLM API if needed

    # Cache Configuration
    MODEL_CACHE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
