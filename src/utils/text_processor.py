import logging

logger = logging.getLogger("TextProcessor")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Splits text into overlapping chunks.
    
    Args:
        text: The full string from OCR.
        chunk_size: Target characters per chunk.
        overlap: How many characters to slide back (context preservation).
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        
        # Adjust 'end' to not cut a word in half
        if end < text_len:
            # Look for the last space within the chunk to break safely
            last_space = text.rfind(' ', start, end)
            if last_space != -1:
                end = last_space
        
        # Extract the chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move the start forward, minus the overlap
        start = end - overlap
        
        # Breaking the loop if we aren't moving forward (prevents infinite loops on huge words)
        if start >= end:
            start = end

    logger.info(f"Split text into {len(chunks)} chunks.")
    return chunks