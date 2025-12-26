import typer
import os
import time
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from typing import Optional

# Import our custom modules
from src import config
from src.core.ocr import OCRProcessor
from src.core.embedder import TextEmbedder
from src.core.vector_db import VectorStore
from src.utils.text_processor import chunk_text
# Initialize App and UI
app = typer.Typer(help="OCR Search Engine CLI")
console = Console()

@app.command()
def index(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-indexing of all files")
):
    """
    Scans the 'data/raw' folder, runs OCR, and builds the search index.
    """
    console.print(Panel("[bold green]Starting Indexing Pipeline[/bold green]"))

    # 1. Check for images
    image_files = [f for f in os.listdir(config.RAW_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    
    if not image_files:
        console.print("[bold red]Error:[/bold red] No images found in [yellow]data/raw[/yellow].")
        console.print("Please add some images and try again.")
        return

    # 2. Initialize Components (Lazy loading)
    with console.status("[bold green]Loading AI Models...[/bold green] (This happens once)", spinner="dots"):
        ocr = OCRProcessor()
        embedder = TextEmbedder(config.EMBEDDING_MODEL_NAME)
        vector_db = VectorStore(config.INDEX_FILE, config.METADATA_FILE, config.VECTOR_DIMENSION)

    # 3. Process Files
    console.print(f"Found [bold]{len(image_files)}[/bold] images to process.")
    
    new_items_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Processing...", total=len(image_files))

        for img_file in image_files:
            img_path = os.path.join(config.RAW_IMAGES_DIR, img_file)
            progress.update(task, description=f"Reading [bold]{img_file}[/bold]...")
            
            # Step A: OCR
            full_text = ocr.extract_text(img_path)
            if not full_text:
                continue

            # Step B: Chunking (The New Part)
            # We use 500 chars per chunk with 100 char overlap
            text_chunks = chunk_text(full_text, chunk_size=500, overlap=100)

            # Step C: Embed & Store EACH chunk
            for i, chunk in enumerate(text_chunks):
                vector = embedder.embed_text(chunk)
                
                if vector is not None:
                    # We store the same filename, but different text segments
                    meta = {
                        "filename": img_file, 
                        "text": chunk, 
                        "chunk_id": i,
                        "total_chunks": len(text_chunks)
                    }
                    vector_db.add_item(vector, meta)
                    new_items_count += 1
            
            progress.advance(task)

    # 4. Save
    vector_db.save_index()
    
    console.print(Panel(f"[bold green]Indexing Complete![/bold green]\n\nIndexed Documents: {new_items_count}\nDatabase stored at: {config.INDEX_DIR}"))

@app.command()
def search(
    query: str = typer.Argument(..., help="The text you want to find"),
    k: int = typer.Option(3, "--top", "-k", help="Number of unique documents to return")
):
    """
    Search indexed documents with Document-Level Grouping.
    """
    if not os.path.exists(config.INDEX_FILE):
        console.print("[red]Index not found. Run 'python main.py index' first.[/red]")
        return

    # 1. Init
    embedder = TextEmbedder(config.EMBEDDING_MODEL_NAME)
    vector_db = VectorStore(config.INDEX_FILE, config.METADATA_FILE, config.VECTOR_DIMENSION)

    # 2. Embed
    query_vector = embedder.embed_text(query)

    # 3. Search (OVERSAMPLING)
    # We ask for k * 10 results to ensure we have enough unique files after grouping.
    # If the user wants Top 3 docs, we fetch Top 30 chunks.
    raw_results = vector_db.search(query_vector, top_k=k * 10)

    # 4. Grouping & Aggregation (The "Re-ranking" Step)
    unique_docs = {}

    for res in raw_results:
        filename = res['filename']
        score = res['score'] # Remember: Lower is better for L2 Distance
        
        # If we haven't seen this file yet, add it.
        # OR if we found a better chunk for an existing file, update the score/preview.
        if filename not in unique_docs:
            unique_docs[filename] = {
                "score": score,
                "preview": res['preview'],
                "chunk_match": res['preview'] # The specific text that matched
            }
        else:
            # For Inner Product (Cosine Similarity since normalized), HIGHER is Better.
            # If we found a chunk with a HIGHER score, keep that one.
            if score > unique_docs[filename]['score']:
                unique_docs[filename]['score'] = score
                unique_docs[filename]['preview'] = res['preview']

    # 5. Convert Dict back to List and Sort
    # Sort by score DESCENDING (Highest score = Best match for Cosine/Inner Product)
    final_results = sorted(unique_docs.items(), key=lambda x: x[1]['score'], reverse=True)

    # 6. Slice to the user's requested 'k'
    top_docs = final_results[:k]

    # 7. Display
    if not top_docs:
        console.print("[yellow]No matches found.[/yellow]")
        return

    table = Table(title=f"Top {k} Documents for: '{query}'")
    table.add_column("Rank", style="dim")
    table.add_column("Score", justify="right", style="cyan")
    table.add_column("Filename", style="magenta")
    table.add_column("Best Matching Snippet", style="green")

    for i, (fname, data) in enumerate(top_docs, 1):
        score_display = f"{data['score']:.4f}"
        snippet = data['preview'].replace('\n', ' ')[:100] + "..." # Truncate for clean table
        
        table.add_row(str(i), score_display, fname, snippet)

    console.print(table)

@app.command()
def info():
    """Check the status of the Vector Database."""
    if os.path.exists(config.METADATA_FILE):
        import json
        with open(config.METADATA_FILE, 'r') as f:
            data = json.load(f)
        console.print(f"[bold]Total Documents Indexed:[/bold] {len(data)}")
        console.print(f"[bold]Index Location:[/bold] {config.INDEX_DIR}")
    else:
        console.print("[yellow]No index found.[/yellow]")

@app.command("test-ocr")
def test_ocr(
    image_name: str = typer.Argument(..., help="Name of the image in data/raw to test"),
    save: bool = typer.Option(True, help="Save the extracted text to data/manual_tests")
):
    """
    Run OCR on a single image and display/save the results.
    """
    image_path = os.path.join(config.RAW_IMAGES_DIR, image_name)
    
    if not os.path.exists(image_path):
        console.print(f"[bold red]Error:[/bold red] File not found: {image_path}")
        return

    # Initialize OCR
    with console.status("[bold green]Running OCR...[/bold green]", spinner="dots"):
        ocr = OCRProcessor()
        text = ocr.extract_text(image_path)

    # Display Results
    console.print(Panel(f"[bold]Extracted Text from {image_name}:[/bold]\n\n{text}", border_style="green"))

    # Save to file
    if save:
        output_filename = f"{os.path.splitext(image_name)[0]}.txt"
        output_path = os.path.join(config.MANUAL_TESTS_DIR, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        console.print(f"[dim]Saved output to: {output_path}[/dim]")

if __name__ == "__main__":
    app()