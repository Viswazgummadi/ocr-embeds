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
            
            # (Optional Optimization) Check if already in metadata to skip
            # For now, we process everything to keep it simple or if forced
            
            progress.update(task, description=f"Reading [bold]{img_file}[/bold]...")
            
            # Step A: OCR
            text = ocr.extract_text(img_path)
            if not text:
                console.print(f"[yellow]Skipped {img_file} (No text found)[/yellow]")
                progress.advance(task)
                continue

            # Step B: Embed
            vector = embedder.embed_text(text)
            
            # Step C: Store
            if vector is not None:
                meta = {"filename": img_file, "text": text}
                vector_db.add_item(vector, meta)
                new_items_count += 1
            
            progress.advance(task)

    # 4. Save
    vector_db.save_index()
    
    console.print(Panel(f"[bold green]Indexing Complete![/bold green]\n\nIndexed Documents: {new_items_count}\nDatabase stored at: {config.INDEX_DIR}"))


@app.command()
def search(
    query: str = typer.Argument(..., help="The text you want to find inside your images"),
    k: int = typer.Option(3, "--top", "-k", help="Number of results to return")
):
    """
    Search your indexed documents using a natural language query.
    """
    # 1. Load Components (Fast load, models cached)
    if not os.path.exists(config.INDEX_FILE):
        console.print("[bold red]Error:[/bold red] Index not found.")
        console.print("Run [yellow]python main.py index[/yellow] first.")
        return

    embedder = TextEmbedder(config.EMBEDDING_MODEL_NAME)
    vector_db = VectorStore(config.INDEX_FILE, config.METADATA_FILE, config.VECTOR_DIMENSION)

    # 2. Embed Query
    query_vector = embedder.embed_text(query)

    # 3. Search
    results = vector_db.search(query_vector, top_k=k)

    # 4. Display Results
    if not results:
        console.print("[yellow]No matches found.[/yellow]")
        return

    table = Table(title=f"Search Results for: '{query}'")
    table.add_column("Score", justify="right", style="cyan", no_wrap=True)
    table.add_column("Filename", style="magenta")
    table.add_column("Text Preview", style="green")

    for res in results:
        # Format score (lower L2 distance = better match, but let's just show raw for now)
        score_display = f"{res['score']:.4f}"
        
        # specific formatting
        preview = res['preview'].replace('\n', ' ')
        
        table.add_row(score_display, res['filename'], preview)

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