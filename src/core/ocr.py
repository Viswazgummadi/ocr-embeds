import pytesseract
from PIL import Image
import logging
import os
import cv2 # OpenCV
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OCR")

class OCRProcessor:
    def __init__(self):
        # 1. PATH CONFIGURATION
        default_path = r'C:\Users\viswa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        if os.path.exists(default_path):
            pytesseract.pytesseract.tesseract_cmd = default_path
        else:
            logger.warning(f"Could not find Tesseract at {default_path}")

    def preprocess_image(self, image_path: str):
        """
        Applies computer vision tricks to make text pop out.
        """
        # Load image using OpenCV
        img = cv2.imread(image_path)
        
        # 1. Resize (Upscale) - Making it 2x or 3x bigger helps read small text
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 2. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Apply Thresholding (Binarization)
        # This turns the image into purely Black and White.
        # OTSU's method automatically finds the best cutoff point to separate text from background.
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. (Optional) Denoising - remove salt-and-pepper noise
        processed_img = cv2.medianBlur(thresh, 3)
        
        return processed_img

    def extract_text(self, image_path: str) -> str:
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Use the new pre-processor
            processed_img_cv = self.preprocess_image(image_path)
            
            # Convert back to PIL image because pytesseract expects PIL or file path
            img_pil = Image.fromarray(processed_img_cv)
            
            # CONFIGURATION TWEAKS:
            # --psm 11: Sparse text. Good for text scattered around (not a paragraph).
            # --psm 6: Assume a single uniform block of text.
            custom_config = r'--oem 3 --psm 11' 
            
            text = pytesseract.image_to_string(img_pil, config=custom_config)
            
            clean_text = text.strip()
            
            if not clean_text:
                logger.warning(f"No text found in {image_path}")
                return ""
            
            # Debug: Print what it found to console to verify improvements
            logger.info(f"OCR Result: {clean_text}")
            
            return clean_text

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {str(e)}")
            return ""