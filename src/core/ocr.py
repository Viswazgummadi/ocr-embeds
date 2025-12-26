import pytesseract
from PIL import Image
import logging
import os
import sys

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OCR")

# WINDOWS CONFIGURATION
# If you didn't add Tesseract to your System PATH, un-comment the line below 
# and make sure the path matches your installation:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCRProcessor:
    def __init__(self):
        # Auto-detect Tesseract on Windows if standard path
        default_path = r'C:\Users\viswa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        if os.path.exists(default_path):
            pytesseract.pytesseract.tesseract_cmd = default_path
        else:
            logger.warning(f"Could not find Tesseract at {default_path}")
    def extract_text(self, image_path: str) -> str:
        """
        Loads an image and returns the text found within it.
        """
        try:
            logger.info(f"Processing image: {image_path}")
            img = Image.open(image_path)
            img = img.convert('RGB')
            text = pytesseract.image_to_string(img)
            clean_text = text.strip()
            
            if not clean_text:
                logger.warning(f"No text found in {image_path}")
                return ""
                
            return clean_text

        except FileNotFoundError:
             logger.error("Tesseract not found! Is it installed?")
             print("ERROR: Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
             return ""
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {str(e)}")
            return ""