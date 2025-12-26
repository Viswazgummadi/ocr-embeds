# OCR Search Engine (Local)

A powerful, privacy-focused tool that extracts text from your local images and allows you to search through them using natural language. No cloud, no API keys, runs 100% on your machine.

## 🏗 Architecture

The system operates in a multi-stage pipeline designed for high accuracy:

1.  **Preprocessing & OCR**:
    *   **Tools**: OpenCV & Tesseract
    *   **Process**: Images are upscaled, binarized (black & white), and denoised using OpenCV to maximize text clarity before passing them to Tesseract OCR.
    *   **Code**: `src/core/ocr.py`

2.  **Text Chunking**:
    *   **Process**: Long documents are split into overlapping chunks (500 chars). This ensures that specific details don't get lost in a large wall of text.
    *   **Code**: `src/utils/text_processor.py`

3.  **Vector Embedding**:
    *   **Model**: `all-MiniLM-L6-v2`
    *   **Process**: Converts text chunks into normalized 384-dimensional vectors.
    *   **Code**: `src/core/embedder.py`

4.  **Vector Database (Cosine Similarity)**:
    *   **Tool**: FAISS (IndexFlatIP)
    *   **Process**: Stores vectors and retrieves them using **Cosine Similarity**. Search results are grouped by filename, so you always get the best matching snippet for each document.
    *   **Code**: `src/core/vector_db.py`

---

## 🚀 Installation

### 1. Prerequisites
- **Python 3.9+**
- **Tesseract OCR Engine**:
    - **Windows**: [Download Installer](https://github.com/UB-Mannheim/tesseract/wiki)
    - Add Tesseract to your System PATH.

### 2. Setup
Clone the repository and install dependencies:

```bash
git clone git@github.com:Viswazgummadi/ocr-embeds.git
cd ocr-embeds
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

---

## 📖 Usage

### 1. Add Data
Place your images (PNG, JPG, TIFF) into the `data/raw/` folder.

### 2. Manual Verification (Optional)
Check how the new OpenCV pre-processing handles your images:

```bash
python main.py test-ocr image_name.png
```
*   **Output**: Prints the extracted text to the console and saves it to `data/manual_tests/`.

### 3. Build Index
Scan images, process text, chunk it, and build the search index.
**Note**: If you changed code or added new images, you can force a rebuild:

```bash
python main.py index --force
```

### 4. Search
Search your documents. The results will show the **Score** (Higher is Better) and the specific snippet that matched.

```bash
# Return top 3 documents
python main.py search "invoice from amazon" -k 3
```

### 5. Check Status
View database stats (Total chunks indexed):

```bash
python main.py info
```

### 6. Run Stress Tests
Verify the system's integrity (Sort order, Chunking logic):

```bash
python -m tests.stress_test
```

---

## 📂 Project Structure

```text
ocr-search-engine/
├── data/
│   ├── raw/             # DROP YOUR IMAGES HERE
│   ├── index/           # FAISS Index & Metadata
│   └── manual_tests/    # Debug output from 'test-ocr'
├── src/
│   ├── core/
│   │   ├── ocr.py       # OpenCV + Tesseract Logic
│   │   ├── embedder.py  # Embedding Logic
│   │   └── vector_db.py # FAISS Vector Store
│   ├── utils/
│   │   └── text_processor.py # Chunking Logic
│   └── config.py
├── main.py              # CLI Application
└── requirements.txt
```
