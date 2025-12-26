from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Embedder")

class TextEmbedder:
    def __init__(self, model_name: str):
        logger.info(f"Loading Embedding Model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        logger.info("Model loaded.")

    def embed_text(self, text: str):
        """
        Converts a string of text into a vector embedding.
        """
        try:
            # Generate embedding
            embedding = self.model.encode(text)
            return embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None