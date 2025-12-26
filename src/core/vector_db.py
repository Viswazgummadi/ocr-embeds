import faiss
import numpy as np
import json
import os
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VectorDB")

class VectorStore:
    def __init__(self, index_path: str, metadata_path: str, dimension: int):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = dimension
        
        self.index = None
        self.metadata = [] # List of dicts: [{"id": 0, "filename": "doc1.png", "text": "..."}]

        self.load_index()

    def create_index(self):
        """Initializes a new FAISS index."""
        # IndexFlatL2 is standard Euclidean distance search (good for general use)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        logger.info("Created new FAISS index.")

    def add_item(self, vector: np.array, meta: Dict):
        """Adds a single vector and its metadata."""
        if self.index is None:
            self.create_index()

        # FAISS expects float32
        vector = np.array([vector]).astype('float32')
        self.index.add(vector)
        
        # Sync metadata ID with FAISS ID (which is sequential 0, 1, 2...)
        meta["id"] = self.index.ntotal - 1 
        self.metadata.append(meta)

    def search(self, query_vector: np.array, top_k: int = 5) -> List[Dict]:
        """Searches the index for similar vectors."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty.")
            return []

        query_vector = np.array([query_vector]).astype('float32')
        
        # distances, indices
        D, I = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(I[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx]
                results.append({
                    "filename": item["filename"],
                    "preview": item["text"][:200] + "...", # Show snippet
                    "score": float(D[0][i]) # Distance score (lower is better for L2)
                })
        
        return results

    def save_index(self):
        """Saves the FAISS index and metadata to disk."""
        if self.index:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            logger.info(f"Index saved to {self.index_path}")

    def load_index(self):
        """Loads index from disk if it exists."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info("Existing index loaded.")
        else:
            self.create_index()