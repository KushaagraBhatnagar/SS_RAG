import logging
import requests
from typing import List, Dict, Any
import asyncio

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("synthesizer")

class ReviewSynthesizer:
    def __init__(self):
        self.ollama_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        logger.info(f"ReviewSynthesizer initialized with mode: {'Cloud' if settings.USE_CLOUD_API else 'Local Ollama'}")

    def _build_prompt(self, title: str, reviews: List[str]) -> str:
        """Construct a strict context-anchored prompt using explicit boundaries."""
        if not reviews:
            reviews_text = "- No customer reviews available."
        else:
            reviews_text = "\n".join([f"- {rev}" for rev in reviews])

        prompt = f"""<|system|>
You are a strict and objective product review synthesizer. Your task is to summarize the customer reviews provided below.
You must adhere to these rules:
1. Output exactly 2 bullet points: one starting with "- Pro:" and one starting with "- Con:".
2. Base your output entirely on the provided Context.
3. Do not include introductory text, explanations, or conclusions. Only output the 2 bullet points.
4. If no positive/negative aspect is mentioned, summarize what is available. If no reviews exist, state "No reviews available."
5. Zero hallucination. Do not add external knowledge, product specs, or opinions not found in the Context.
<|end|>

<|context|>
Product: {title}
Reviews:
{reviews_text}
<|end|>

<|output|>
Provide exactly 2 bullet points (one pro and one con):
"""
        return prompt

    def _synthesize_local(self, prompt: str) -> str:
        """Call local Ollama service for review synthesis."""
        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0, # Greedy decoding to minimize hallucinations
                "num_predict": 150  # Cap response length
            }
        }
        try:
            logger.info(f"Sending request to Ollama ({settings.OLLAMA_MODEL}) at {self.ollama_url}...")
            response = requests.post(self.ollama_url, json=payload, timeout=25)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            logger.error(f"Error calling local Ollama service: {e}")
            return (
                "- Pro: Could not connect to LLM container.\n"
                "- Con: Please verify Ollama is running and model is pulled."
            )

    def _synthesize_cloud(self, prompt: str) -> str:
        """Call external LLM Provider (OpenAI/Groq/HuggingFace compatible endpoint)."""
        provider = settings.LLM_PROVIDER.lower()
        api_key = settings.LLM_API_KEY
        
        if not api_key:
            logger.warning("Cloud API key is missing. Falling back to local/default response.")
            return "- Pro: Cloud API Key missing.\n- Con: Please configure LLM_API_KEY in environment."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Determine URL and Model based on Provider
        url = settings.LLM_API_URL
        if not url:
            if provider == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                model = "gpt-3.5-turbo"
            elif provider == "groq":
                url = "https://api.groq.com/openai/v1/chat/completions"
                model = "llama-3.1-8b-instant"
            else:
                url = "https://api.openai.com/v1/chat/completions" # Default fallback
                model = "gpt-3.5-turbo"
        else:
            model = settings.OLLAMA_MODEL  # Use configured model for custom endpoints

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a review synthesizer. Output exactly 2 bullet points: '- Pro: ...' and '- Con: ...' with no extra text."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 150
        }

        try:
            logger.info(f"Sending request to cloud LLM provider ({provider}) at {url}...")
            response = requests.post(url, json=payload, headers=headers, timeout=20)
            if response.status_code != 200:
                logger.error(f"Cloud LLM provider error response body: {response.text}")
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error calling cloud LLM provider {provider}: {e}")
            return (
                "- Pro: Cloud LLM API call failed.\n"
                f"- Con: Details: {str(e)[:60]}..."
            )

    async def synthesize_reviews(self, title: str, reviews: List[str]) -> str:
        """Asynchronously run LLM inference in a thread pool to avoid blocking the event loop."""
        prompt = self._build_prompt(title, reviews)
        
        if settings.USE_CLOUD_API or settings.LLM_PROVIDER != "ollama":
            # Run cloud synthesis asynchronously
            summary = await asyncio.to_thread(self._synthesize_cloud, prompt)
        else:
            # Run local synthesis asynchronously
            summary = await asyncio.to_thread(self._synthesize_local, prompt)
            
        return summary
