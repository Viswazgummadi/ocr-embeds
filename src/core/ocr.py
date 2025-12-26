import pytesseract
from PIL import Image
import logging
import os
import cv2 # OpenCV
import numpy as np
from pytesseract import Output

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
        
        # 1. Resize (Upscale) - 3x is better for stylized/small text
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # 2. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Apply Thresholding (Binarization)
        # Binary Inverse is often better for posters. It turns letters WHITE and background BLACK.
        # This helps isolate text from the wood grain.
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 4. Dilation (The "Thickening" Trick)
        # This fills in the hollow "Outline Fonts" (like NO PAIN) so Tesseract sees solid letters.
        kernel = np.ones((2,2), np.uint8)
        processed_img = cv2.dilate(thresh, kernel, iterations=1)
        
        # 5. (Optional) Denoising - remove salt-and-pepper noise
        processed_img = cv2.medianBlur(processed_img, 3)
        
        return processed_img

    def extract_text(self, image_path: str) -> str:
        try:
            logger.info(f"Processing image: {image_path}")
            
            # 1. Pre-process (CRITICAL FIX: Don't reload original image after this!)
            processed_img = self.preprocess_image(image_path)
            img = Image.fromarray(processed_img) 
            
            # --- TESSERACT CONFIGURATION ---
            # --psm 11: Sparse Text (Best for posters/scattered words)
            # --oem 3: Default Engine
            custom_config = r'--oem 3 --psm 11'
            
            # 2. Get Data + Confidence Scores
            data = pytesseract.image_to_data(img, output_type=Output.DICT, config=custom_config)
            
            valid_words = []
            
            # 3. Iterate through every word found
            for i in range(len(data['text'])):
                word = data['text'][i].strip()
                
                try:
                    conf = int(data['conf'][i])
                except:
                    conf = 0

                # --- FILTERING LOGIC ---
                
                # Rule A: Minimum Confidence
                # Stylized fonts have LOW confidence. We must lower the bar.
                # Was 60 -> Now 30
                if conf < 30:
                    continue
                
                # Rule B: Empty
                if not word:
                    continue

                # Rule C: Symbol Garbage
                # Allow basic punctuation but filter pure garbage
                if len(word) > 1 and not any(char.isalnum() for char in word):
                    continue

                # Rule D: Single Letter Noise
                if len(word) == 1 and word.lower() not in ['a', 'i']:
                    continue

                valid_words.append(word)

            final_text = " ".join(valid_words)
            
            if not final_text:
                logger.warning(f"No valid text found in {image_path}")
                return ""
            
            logger.info(f"Cleaned OCR Result: {final_text}")
            return final_text

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {str(e)}")
            return ""