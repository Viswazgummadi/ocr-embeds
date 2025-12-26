import os
import shutil
import unittest
import numpy as np
from src.utils.text_processor import chunk_text
from src.core.vector_db import VectorStore
import cv2 

# Configuration for Test
TEST_DIR = "data/stress_test"
TEST_INDEX_DIR = "data/stress_test_index"
TEST_INDEX_FILE = os.path.join(TEST_INDEX_DIR, "index.bin")
TEST_METADATA_FILE = os.path.join(TEST_INDEX_DIR, "metadata.json")

class StressTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Clean up previous tests
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
        if os.path.exists(TEST_INDEX_DIR):
            shutil.rmtree(TEST_INDEX_DIR)
            
        os.makedirs(TEST_DIR)
        os.makedirs(TEST_INDEX_DIR)
        
        # Create Dummy Images
        # 1. Good Image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(img, "Hello World", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(TEST_DIR, "good.png"), img)
        
        # 2. Corrupt Image (0 bytes)
        with open(os.path.join(TEST_DIR, "corrupt.png"), "w") as f:
            f.write("")
            
        # 3. Text file masked as image
        with open(os.path.join(TEST_DIR, "fake.png"), "w") as f:
            f.write("This is not an image")

    def test_chunking_logic(self):
        print("\n[TEST] Chunking Logic")
        text = "A" * 1200 # 1200 chars
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        # 0-500, 400-900, 800-1300 -- Should be 3 chunks
        self.assertEqual(len(chunks), 3)
        print("✅ Chunking works correctly.")

    def test_vector_db_sorting(self):
        print("\n[TEST] Vector DB Sorting (Cosine Similarity)")
        db = VectorStore(TEST_INDEX_FILE, TEST_METADATA_FILE, 384)
        
        # Vector A (Target)
        vec_a = np.ones(384, dtype='float32') 
        # Vector B (Close to A)
        vec_b = np.copy(vec_a)
        # Vector C (Opposite/Far)
        vec_c = np.zeros(384, dtype='float32') # Orthogonal
        
        db.add_item(vec_a, {"filename": "match.png", "text": "Match"})
        db.add_item(vec_c, {"filename": "garbage.png", "text": "Garbage"})
        
        # Search for A. Match should be first.
        results = db.search(vec_a, top_k=2)
        
        self.assertEqual(results[0]['filename'], "match.png")
        # Score for identical vectors in Inner Product (normalized) should be ~1.0? 
        # Or at least higher than 0.
        print(f"✅ Top match score: {results[0]['score']}")
        self.assertTrue(results[0]['score'] > results[1]['score'])
        print("✅ Sorting is correct (High Score First).")

    def test_mass_indexing_simulation(self):
        print("\n[TEST] Mass Indexing Simulation (1000 items)")
        db = VectorStore(TEST_INDEX_FILE, TEST_METADATA_FILE, 384)
        
        # Add 1000 items
        for i in range(1000):
            vec = np.random.rand(384).astype('float32')
            db.add_item(vec, {"filename": f"doc_{i}.png", "text": f"text {i}"})
            
        db.save_index()
        
        # Reload
        db2 = VectorStore(TEST_INDEX_FILE, TEST_METADATA_FILE, 384)
        self.assertEqual(db2.index.ntotal, 1000) # Should be exactly 1000
        print("✅ Successfully indexed and reloaded 1000 items.")

if __name__ == "__main__":
    unittest.main()
