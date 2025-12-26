import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_IMAGES_DIR = os.path.join(DATA_DIR, "raw")
INDEX_DIR = os.path.join(DATA_DIR, "index")
MANUAL_TESTS_DIR = os.path.join(DATA_DIR, "manual_tests")

# Ensure directories exist
os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(MANUAL_TESTS_DIR, exist_ok=True)

# Vector DB Settings
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Fast and effective
VECTOR_DIMENSION = 384 # Dimension for MiniLM-L6-v2