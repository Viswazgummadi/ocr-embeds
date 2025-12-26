from src.core.ocr import OCRProcessor
import os
from src import config

def test_ocr(image_name):
    ocr = OCRProcessor()
    image_path = os.path.join(config.RAW_IMAGES_DIR, image_name)
    
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    print(f"--- Processing {image_name} ---")
    text = ocr.extract_text(image_path)
    print("\n--- Extracted Text ---")
    print(text)
    print("\n----------------------")

if __name__ == "__main__":
    # You can change this to any image file in data/raw
    test_ocr("test_image.png")
