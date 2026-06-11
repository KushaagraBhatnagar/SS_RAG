import logging
import uuid
from typing import List, Dict, Any
import numpy as np
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from app.config import settings

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingestor")

# Real-world expanded fallback dataset with multiple categories (Electronics, Office/Pens, Kitchen, Books)
FALLBACK_METADATA = [
    # --- Electronics ---
    {
        "parent_asin": "B07HZ2MJA6",
        "title": "Echo Dot (3rd Gen) - Smart speaker with Alexa - Charcoal",
        "description": "Meet Echo Dot - Our most popular voice-controlled speaker, now with improved sound and a new design. Ask Alexa to play music, answer questions, read the news, check the weather, set alarms, control compatible smart home devices, and more."
    },
    {
        "parent_asin": "B08XM44C7A",
        "title": "Kindle Paperwhite (16 GB) - Now with a 6.8 inch display and adjustable warm light",
        "description": "Purpose-built for reading - With a flush-front design and 300 ppi glare-free display that reads like real paper, even in bright sunlight. Adjustable warm light to shift screen shade from white to amber."
    },
    {
        "parent_asin": "B09B8WLYRD",
        "title": "Fire TV Stick 4K Max streaming device, Wi-Fi 6, Alexa Voice Remote",
        "description": "Our most powerful streaming stick - 40% more powerful than Fire TV Stick 4K, with faster app starts and more fluid navigation. Support for next-gen Wi-Fi 6 for smoother 4K streaming."
    },
    {
        "parent_asin": "B01D78N8CA",
        "title": "Sony WH-1000XM4 Wireless Premium Noise Canceling Overhead Headphones",
        "description": "Industry leading noise canceling with Dual Noise Sensor technology. Next-level music with Edge-AI, co-developed with Sony Music Studios Tokyo. Up to 30-hour battery life with quick charging."
    },
    {
        "parent_asin": "B08L5TNJHG",
        "title": "Apple iPhone 12 Pro, 128GB, Pacific Blue - Fully Unlocked (Renewed)",
        "description": "This device is unlocked and compatible with any carrier of choice on GSM and CDMA networks. Super Retina XDR display, Ceramic Shield, 5G speed, A14 Bionic chip, and Pro camera system."
    },
    # --- Office / Pens / Stationery ---
    {
        "parent_asin": "B001GAOT7C",
        "title": "Pilot G2 Premium Gel Ink Rolling Ball Pens, Fine Point 0.7mm, Black, 12-Pack",
        "description": "Super smooth writing black gel ink pens. Long-lasting gel ink with a comfortable contoured rubber grip for writing control. Excellent writing instrument for offices, schools, and workspaces."
    },
    {
        "parent_asin": "B07H8N3F5V",
        "title": "BIC Round Stic Xtra Life Ballpoint Pen, Medium Point 1.0mm, Blue, 60-Pack",
        "description": "Reliable blue ballpoint pens featuring a flexible round barrel for writing comfort. Long-lasting ink writes 90% longer on average than other comparable pens. Perfect for daily writing tasks."
    },
    {
        "parent_asin": "B0028R3R52",
        "title": "Five Star Spiral Notebooks, 1-Subject College Ruled, 100 Sheets, 3-Pack",
        "description": "Durably bound 1-subject notebook with water-resistant cover. Contains 100 college ruled double-sided sheets. Great for notes, journaling, sketch mapping, and organization."
    },
    # --- Kitchen & Dining ---
    {
        "parent_asin": "B07HMGD9R6",
        "title": "Keurig K-Mini Single Serve Coffee Maker, Compact 5-Inch Wide, Dusty Rose",
        "description": "Ultra-compact single serve K-Cup coffee brewer. Fits anywhere with a width under 5 inches. Brews any cup size between 6-12oz with Keurig K-Cup pods. Cord storage helps keep countertops neat."
    },
    {
        "parent_asin": "B00NGV4506",
        "title": "Ninja Professional 72oz Countertop Blender with 1000-Watt Base",
        "description": "Professional high-powered kitchen blender. 1000 watts of power crushes ice, frozen fruits, and greens in seconds. Perfect for making smoothies, shakes, purees, and frozen drinks."
    },
    # --- Books / Growth ---
    {
        "parent_asin": "B07D234S4A",
        "title": "Atomic Habits: An Easy & Proven Way to Build Good Habits & Break Bad Ones",
        "description": "Hardcover edition of the best-selling book by James Clear. Formulates practical strategies to design a system where good habits are easy and bad habits are broken through tiny daily changes."
    }
]

FALLBACK_REVIEWS = [
    # Electronics reviews
    {"parent_asin": "B07HZ2MJA6", "text": "Sound quality is surprisingly good for such a small speaker. Alexa responds quickly and controls all my smart lights perfectly."},
    {"parent_asin": "B07HZ2MJA6", "text": "Extremely useful in the kitchen for timers and music. However, it sometimes disconnects from Wi-Fi for no apparent reason."},
    {"parent_asin": "B08XM44C7A", "text": "The warm light is a game changer for reading in bed at night. Battery lasts for weeks, and the screen size is perfect."},
    {"parent_asin": "B08XM44C7A", "text": "Great screen quality but a bit sluggish when browsing the store or switching books compared to a phone."},
    {"parent_asin": "B09B8WLYRD", "text": "Very fast, handles 4K streaming without any buffering or lag. The voice control works smoothly."},
    {"parent_asin": "B09B8WLYRD", "text": "Good performance, but the remote feels a bit cheap and has too many sponsored buttons that cannot be remapped."},
    {"parent_asin": "B01D78N8CA", "text": "The noise cancellation is absolutely magical. I can block out airline engine noise completely. Sound is balanced and rich."},
    {"parent_asin": "B01D78N8CA", "text": "Great audio, but the touch controls on the ear cup can be sensitive and trigger accidentally when adjusting them."},
    {"parent_asin": "B08L5TNJHG", "text": "Phone arrived in perfect condition with 90% battery health. Runs fast and the cameras are amazing in low light."},
    {"parent_asin": "B08L5TNJHG", "text": "Excellent value for money being renewed. The battery drains a bit fast on 5G but lasts the whole day on Wi-Fi."},
    
    # Pens / Stationery reviews
    {"parent_asin": "B001GAOT7C", "text": "These are the best gel pens I have ever used. They write super smoothly and do not smudge at all. Perfect for taking notes."},
    {"parent_asin": "B001GAOT7C", "text": "Great ink quality, but sometimes they leak if left uncapped in a backpack. Still my go-to pen for daily work."},
    {"parent_asin": "B07H8N3F5V", "text": "Simple, cheap, and reliable blue pens. You get 60 of them, so it doesn't matter if you lose some. Ink flows well."},
    {"parent_asin": "B07H8N3F5V", "text": "Basic plastic barrel, feels a bit light in the hand, but they write fine without skipping. Very economical."},
    {"parent_asin": "B0028R3R52", "text": "Nice sturdy notebooks. The covers are durable, and the pocket dividers inside are handy for sorting loose sheets."},
    {"parent_asin": "B0028R3R52", "text": "Paper is slightly thin, so heavy ink might bleed through. Works great for normal pencil and ballpoint pens."},
    
    # Kitchen reviews
    {"parent_asin": "B07HMGD9R6", "text": "Perfect for my small studio apartment countertop. Brews a quick single cup in the morning with no clean-up hassle."},
    {"parent_asin": "B07HMGD9R6", "text": "Cute color and compact size. However, you have to add fresh water every single time since there is no storage reservoir."},
    {"parent_asin": "B00NGV4506", "text": "Crushes ice into snow in under 5 seconds! Ideal for daily protein shakes and fruit smoothies. Very sturdy base."},
    {"parent_asin": "B00NGV4506", "text": "Very powerful, but it is extremely loud while running. Sounds like a lawnmower in the kitchen, but does the job perfectly."},
    
    # Books reviews
    {"parent_asin": "B07D234S4A", "text": "One of the most practical self-help books ever written. The 1% better every day rule actually works. Highly recommend."},
    {"parent_asin": "B07D234S4A", "text": "Interesting concepts, though some ideas are repeated. The habit loop explanations are clear and easy to implement."}
]


def load_real_data() -> List[Dict[str, Any]]:
    """Loads and joins Amazon Reviews and Metadata from GitHub CSV, falling back to static real data."""
    csv_url = "https://raw.githubusercontent.com/ArfaNada/Intelligent-Report-Generator/main/merged_electronics_dataset.csv"
    try:
        logger.info(f"Attempting to load real Amazon Electronics dataset from GitHub: {csv_url}...")
        import pandas as pd
        df = pd.read_csv(csv_url)
        
        # Clean null values in name and review_text
        df = df.dropna(subset=["name", "review_text"])
        
        # Group by product name to aggregate reviews
        grouped = df.groupby("name")
        products: List[Dict[str, Any]] = []
        
        count = 0
        for name, group in grouped:
            if count >= settings.MAX_PRODUCTS_TO_INGEST:
                break
                
            first_row = group.iloc[0]
            # Generate a stable 10-char ASIN from the product name
            stable_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, name).hex
            parent_asin = f"ASIN_{stable_uuid[:8].upper()}"
            
            sub_cat = first_row.get("sub_category", "Electronics")
            actual_price = first_row.get("actual_price", "N/A")
            discount_price = first_row.get("discount_price", "N/A")
            
            # Form description
            description_text = f"Category: {sub_cat} | Original Price: {actual_price} | Offer Price: {discount_price}"
            
            # Aggregate reviews
            reviews = group["review_text"].head(settings.MAX_REVIEWS_PER_PRODUCT).tolist()
            reviews = [str(r) for r in reviews if str(r).strip()]
            
            products.append({
                "parent_asin": parent_asin,
                "title": name,
                "description": description_text,
                "reviews": reviews
            })
            count += 1
            
        logger.info(f"Successfully loaded and grouped {len(products)} unique products from online CSV.")
        return products

    except Exception as e:
        logger.warning(f"Could not load online CSV from GitHub: {e}. Falling back to static fallback data.")
        
        # Process fallback data
        products = {item["parent_asin"]: {**item, "reviews": []} for item in FALLBACK_METADATA}
        for rev in FALLBACK_REVIEWS:
            asin = rev["parent_asin"]
            if asin in products:
                products[asin]["reviews"].append(rev["text"])
                
        return list(products.values())


def seed_database(products: List[Dict[str, Any]]) -> None:
    """Seeds the processed Amazon products into the Qdrant Vector database."""
    # 1. Connect to Qdrant Client
    mode = settings.QDRANT_MODE.lower()
    logger.info(f"Connecting to Qdrant using mode '{mode}'...")
    if mode == "local":
        client = QdrantClient(path="qdrant_local_data")
    elif mode == "cloud" or settings.QDRANT_API_KEY:
        url = settings.QDRANT_HOST if settings.QDRANT_HOST.startswith("http") else f"https://{settings.QDRANT_HOST}"
        client = QdrantClient(url=url, api_key=settings.QDRANT_API_KEY)
    else:
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    # 2. Instantiate Bi-Encoder
    logger.info(f"Initializing Bi-Encoder: {settings.BI_ENCODER_MODEL}...")
    bi_encoder = SentenceTransformer(settings.BI_ENCODER_MODEL, cache_folder=settings.MODEL_CACHE_DIR)

    # 3. Recreate collection with optimized HNSW index parameters
    logger.info(f"Creating/Recreating Qdrant collection: {settings.QDRANT_COLLECTION}...")
    client.recreate_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=384,  # output dimension of sentence-transformers/all-MiniLM-L6-v2
            distance=models.Distance.COSINE
        ),
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=200
        )
    )

    # 4. Generate contexts, embed them, and prepare upload points
    logger.info("Encoding product context documents...")
    documents = []
    points = []

    for item in products:
        context = f"Title: {item['title']} | Description: {item['description']}"
        documents.append(context)

    # Generate embeddings in batch
    embeddings = bi_encoder.encode(documents, show_progress_bar=True, convert_to_numpy=True)

    for i, item in enumerate(products):
        # Generate stable UUID from parent_asin
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item["parent_asin"]))
        
        points.append(
            models.PointStruct(
                id=point_id,
                vector=embeddings[i].tolist(),
                payload={
                    "parent_asin": item["parent_asin"],
                    "title": item["title"],
                    "description": item["description"],
                    "reviews": item["reviews"]
                }
            )
        )

    # 5. Insert points in batches
    logger.info(f"Uploading {len(points)} points to Qdrant...")
    batch_size = 100
    for idx in range(0, len(points), batch_size):
        batch = points[idx : idx + batch_size]
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=batch
        )
        
    logger.info("Database seeding successfully completed!")


if __name__ == "__main__":
    data = load_real_data()
    seed_database(data)
