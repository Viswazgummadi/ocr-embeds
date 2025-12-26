# OCR Search Engine (Local)

A powerful, privacy-focused tool that extracts text from your local images and allows you to search through them using natural language. No cloud, no API keys, runs 100% on your machine.

## 🏗 Architecture

The system operates in a three-step pipeline:

1.  **Optical Character Recognition (OCR)**:
    *   **Tool**: Tesseract OCR
    *   **Process**: Scans images in `data/raw` and extracts raw text string.
    *   **Code**: `src/core/ocr.py`

2.  **Vector Embedding**:
    *   **Model**: `all-MiniLM-L6-v2` (via `sentence-transformers`)
    *   **Process**: Converts the extracted text into a 384-dimensional vector representation. This allows for semantic search (finding meaning, not just keyword matching).
    *   **Code**: `src/core/embedder.py`

3.  **Vector Database**:
    *   **Tool**: FAISS (Facebook AI Similarity Search)
    *   **Process**: Stores the vectors and metadata (filenames, original text) for efficient similarity retrieval.
    *   **Code**: `src/core/vector_db.py`

---

## 🚀 Installation

### 1. Prerequisites
- **Python 3.9+**
- **Tesseract OCR Engine**:
    - **Windows**: [Download Installer](https://github.com/UB-Mannheim/tesseract/wiki)
    - Add Tesseract to your System PATH or update `src/core/ocr.py` if needed.

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
If you want to check if the OCR is working correctly on a specific image:

```bash
python main.py test-ocr image_name.png
```
*   **Output**: Prints text to console and saves it to `data/manual_tests/image_name.txt`.

### 3. Build Index
Scan all images and build the searchable database:

```bash
python main.py index
```

### 4. Search
Search your documents using natural language:

```bash
python main.py search "invoice from amazon"
python main.py search "meeting notes about project alpha"
```

### 5. Check Status
View database stats:

```bash
python main.py info
```

---

## 📂 Project Structure

```text
ocr-search-engine/
├── data/
│   ├── raw/             # DROP YOUR IMAGES HERE
│   ├── index/           # FAISS Index & Metadata (Generated)
│   └── manual_tests/    # Text outputs from 'test-ocr'
├── src/
│   ├── core/
│   │   ├── ocr.py       # Tesseract Wrapper
│   │   ├── embedder.py  # SentenceTransformer Wrapper
│   │   └── vector_db.py # FAISS Wrapper
│   └── config.py        # Path settings
├── main.py              # CLI Entry Point
└── requirements.txt     # Python Dependencies
```
