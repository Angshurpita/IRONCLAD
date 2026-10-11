"""CLI entry point for IRONCLAD with Rich terminal formatting."""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ironclad import __version__
from ironclad.config import get_settings
from ironclad.errors import (
    KnowledgeBaseNotFoundError,
)
from ironclad.logging import setup_logging
from ironclad.services.status import get_status, run_diagnostics

app = typer.Typer(
    name="ironclad",
    help="IRONCLAD: A minimal, reliable, production-minded Retrieval-Augmented Generation CLI.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


def output_json(data: object) -> None:
    """Output machine-readable JSON without Rich terminal wrapping."""
    if isinstance(data, str):
        sys.stdout.write(data + "\n")
    else:
        sys.stdout.write(json.dumps(data, indent=2) + "\n")
    sys.stdout.flush()


def version_callback(value: bool):
    if value:
        console.print(f"[bold cyan]IRONCLAD[/bold cyan] version [bold green]{__version__}[/bold green]")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show the application version and exit.", callback=version_callback
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose debug logging."),
):
    """Production-minded RAG developer CLI."""
    setup_logging(verbose=verbose)


@app.command()
def init(
    path: Path = typer.Argument(
        Path("."),
        help="Path to project directory to initialize knowledge base for.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
):
    """Create or initialize an Ironclad knowledge base in the target project."""
    try:
        settings = get_settings(path)
        kb_dir = settings.ironclad_path
        kb_dir.mkdir(parents=True, exist_ok=True)
        (kb_dir / "vectors").mkdir(parents=True, exist_ok=True)

        # Initialize SQLite metadata database
        db_path = settings.metadata_db_path
        created = not db_path.exists()

        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime REAL NOT NULL,
                    doc_type TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    heading TEXT,
                    file_hash TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    FOREIGN KEY (file_path) REFERENCES documents(file_path) ON DELETE CASCADE
                );
                """
            )
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT OR IGNORE INTO metadata_info (key, value) VALUES ('schema_version', '1');"
            )
            conn.execute(
                "INSERT OR IGNORE INTO metadata_info (key, value) VALUES ('created_at', ?);",
                (now,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata_info (key, value) VALUES ('embedding_model', ?);",
                (settings.embedding_model,),
            )
            conn.commit()

        if json_output:
            output_json(
                {
                    "status": "success",
                    "initialized": True,
                    "created": created,
                    "project_root": str(settings.project_root),
                    "knowledge_base_path": str(kb_dir),
                }
            )
        else:
            action = "Initialized new" if created else "Re-verified existing"
            console.print(
                Panel.fit(
                    f"[bold green]✔[/bold green] {action} Ironclad knowledge base.\n\n"
                    f"[dim]Project Root:[/dim] [cyan]{settings.project_root}[/cyan]\n"
                    f"[dim]Knowledge Base:[/dim] [cyan]{kb_dir}[/cyan]\n"
                    f"[dim]Metadata DB:[/dim] [cyan]{db_path}[/cyan]\n"
                    f"[dim]Embedding Model:[/dim] [cyan]{settings.embedding_model}[/cyan]\n\n"
                    f"Next step: Run [bold white]ironclad ingest {path}[/bold white] to index documents.",
                    title="[bold cyan]IRONCLAD INIT[/bold cyan]",
                    border_style="green",
                )
            )
    except Exception as e:
        err_console.print(f"[bold red]Initialization failed:[/bold red] {e}")
        raise typer.Exit(code=getattr(e, "exit_code", 1)) from e


@app.command()
def ingest(
    path: Path = typer.Argument(
        Path("."),
        help="Path to project directory containing documents and code.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    update: bool = typer.Option(
        False, "--update", help="Incrementally update changed files and remove stale entries."
    ),
    reindex: bool = typer.Option(
        False, "--reindex", help="Wipe existing index and re-ingest all documents from scratch."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
):
    """Ingest documents and build or update the knowledge base index."""
    settings = get_settings(path)
    if not settings.is_initialized():
        err_console.print(
            f"[bold red]Error:[/bold red] Knowledge base not initialized at '{settings.project_root}'.\n"
            f"[yellow]Run `ironclad init {path}` first.[/yellow]"
        )
        raise typer.Exit(code=KnowledgeBaseNotFoundError.exit_code)

    from ironclad.services.ingest import run_ingestion

    try:
        summary = run_ingestion(settings=settings, target_path=path, update=update, reindex=reindex)
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Ingestion interrupted. Knowledge base remains intact.[/yellow]")
        raise typer.Exit(code=130) from None
    except Exception as e:
        err_console.print(f"[bold red]Ingestion failed:[/bold red] {e}")
        raise typer.Exit(code=getattr(e, "exit_code", 1)) from e

    if json_output:
        output_json(summary.to_dict())
        return

    table = Table(
        title="IRONCLAD Ingestion Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="dim", width=25)
    table.add_column("Value")

    table.add_row("Mode", "[bold cyan]Incremental Update[/bold cyan]" if update else "[bold green]Full Ingestion[/bold green]")
    table.add_row("Target Directory", summary.target_path)
    table.add_row("Files Added", str(summary.files_added))
    table.add_row("Files Updated", str(summary.files_updated))
    table.add_row("Files Deleted", str(summary.files_deleted))
    table.add_row("Files Skipped (Unchanged)", str(summary.files_skipped))
    table.add_row("Total Indexed Documents", f"[bold green]{summary.total_files}[/bold green]")
    table.add_row("Total Chunks Created", f"[bold green]{summary.total_chunks}[/bold green]")
    table.add_row("Duration", f"{summary.duration_seconds}s")

    console.print(table)

    if summary.errors:
        console.print("\n[bold red]Ingestion Warnings / Errors:[/bold red]")
        for err in summary.errors:
            console.print(f"  • [yellow]{err}[/yellow]")
    else:
        console.print(
            "\n[bold green]✔ Ingestion complete.[/bold green] Knowledge base ready for queries."
        )


@app.command()
def ask(
    query: str = typer.Argument(..., help="Question to answer using grounded retrieved context."),
    limit: int = typer.Option(5, "--limit", "-n", help="Maximum evidence passages to retrieve."),
    path: Optional[Path] = typer.Option(
        None, "--path", "-p", help="Target project path (defaults to current working directory)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
):
    """Retrieve evidence and generate a grounded answer using Gemma."""
    settings = get_settings(path)
    if not settings.is_initialized():
        err_console.print(
            "[bold red]Error:[/bold red] Knowledge base not found.\n"
            "[yellow]Run `ironclad init` and `ironclad ingest` first.[/yellow]"
        )
        raise typer.Exit(code=KnowledgeBaseNotFoundError.exit_code)

    from ironclad.generation.gemma import generate_grounded_answer
    from ironclad.generation.provider import get_inference_provider
    from ironclad.services.query import execute_hybrid_search

    # 1. Retrieve hybrid evidence
    try:
        selected_evidence, all_fused, total_tokens = execute_hybrid_search(
            settings=settings,
            query=query,
            top_k=limit,
        )
    except Exception as e:
        err_console.print(f"[bold red]Retrieval failed:[/bold red] {e}")
        raise typer.Exit(code=getattr(e, "exit_code", 1)) from e

    # 2. Privacy notification for remote hosted inference
    if not json_output and settings.inference_provider == "openrouter" and selected_evidence:
        err_console.print(
            f"[dim]Notice: Transmitting {len(selected_evidence)} retrieved evidence blocks "
            f"(~{total_tokens} tokens) to OpenRouter ({settings.openrouter_model}).[/dim]"
        )

    # 3. Call generation pipeline
    try:
        provider = get_inference_provider(settings)
        rag_response = generate_grounded_answer(
            query=query,
            evidence=selected_evidence,
            provider=provider,
        )
    except Exception as e:
        err_console.print(f"[bold red]Generation failed:[/bold red] {e}")
        raise typer.Exit(code=getattr(e, "exit_code", 1)) from e

    if json_output:
        output_json(rag_response.model_dump(mode="json"))
        return

    # 4. Display formatted CLI output
    meta_info = (
        f"[dim]Model: {rag_response.model_used} | "
        f"Latency: {rag_response.latency_ms}ms | "
        f"Evidence Chunks: {rag_response.retrieved_chunks_count}[/dim]"
    )

    if rag_response.insufficient_evidence:
        console.print(
            Panel(
                f"{rag_response.answer}\n\n{meta_info}",
                title="[bold yellow]⚠ Insufficient Evidence[/bold yellow]",
                border_style="yellow",
            )
        )
        return

    console.print(
        Panel(
            f"{rag_response.answer}\n\n{meta_info}",
            title="[bold green]IRONCLAD Answer[/bold green]",
            border_style="green",
        )
    )

    if rag_response.citations:
        cit_table = Table(
            title="Verified Ground-Truth Citations",
            show_header=True,
            header_style="bold cyan",
        )
        cit_table.add_column("#", style="bold", width=4)
        cit_table.add_column("Source File", style="cyan", width=30)
        cit_table.add_column("Lines", style="green", width=12)
        cit_table.add_column("Excerpt", style="dim")

        for cit in rag_response.citations:
            cit_table.add_row(
                f"[{cit.citation_index}]",
                cit.file_path,
                f"L{cit.start_line}-L{cit.end_line}",
                cit.excerpt.replace("\n", " ").strip()[:100],
            )

        console.print(cit_table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search term or phrase to retrieve matching excerpts."),
    limit: int = typer.Option(5, "--limit", "-n", help="Maximum number of search results to return."),
    path: Optional[Path] = typer.Option(
        None, "--path", "-p", help="Target project path (defaults to current working directory)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
):
    """Display matching excerpts, relevance information, and source locations without invoking LLM."""
    settings = get_settings(path)
    if not settings.is_initialized():
        err_console.print(
            "[bold red]Error:[/bold red] Knowledge base not found.\n"
            "[yellow]Run `ironclad init` and `ironclad ingest` first.[/yellow]"
        )
        raise typer.Exit(code=KnowledgeBaseNotFoundError.exit_code)

    from ironclad.services.query import execute_hybrid_search

    try:
        selected_context, all_fused, total_tokens = execute_hybrid_search(
            settings=settings,
            query=query,
            top_k=limit,
        )
    except Exception as e:
        err_console.print(f"[bold red]Search failed:[/bold red] {e}")
        raise typer.Exit(code=getattr(e, "exit_code", 1)) from e

    if json_output:
        output_json(
            {
                "query": query,
                "total_results": len(selected_context),
                "total_tokens": total_tokens,
                "results": [r.model_dump(mode="json") for r in selected_context],
            }
        )
        return

    if not selected_context:
        console.print(f"\n[yellow]No matching results found for:[/yellow] [bold white]{query}[/bold white]\n")
        return

    console.print(
        f"\n[bold cyan]Found {len(selected_context)} matching passages[/bold cyan] "
        f"[dim](Context: ~{total_tokens} tokens)[/dim] for [bold white]\"{query}\"[/bold white]:\n"
    )

    for idx, res in enumerate(selected_context, start=1):
        dense_str = f"#{res.vector_rank}" if res.vector_rank else "—"
        sparse_str = f"#{res.lexical_rank}" if res.lexical_rank else "—"

        panel_content = (
            f"[dim]Source:[/dim] [bold cyan]{res.file_path}:{res.start_line}-{res.end_line}[/bold cyan]  "
            f"[dim]| Type:[/dim] [yellow]{res.doc_type}[/yellow]  "
            f"[dim]| RRF Score:[/dim] [green]{res.score:.4f}[/green]  "
            f"[dim]| Dense Rank:[/dim] {dense_str}  "
            f"[dim]| Sparse Rank:[/dim] {sparse_str}\n\n"
            f"{res.content.strip()}"
        )
        console.print(
            Panel(
                panel_content,
                title=f"[bold green]Result [{idx}][/bold green]",
                border_style="cyan" if idx == 1 else "dim",
                expand=False,
            )
        )


@app.command()
def status(
    path: Optional[Path] = typer.Option(
        None, "--path", "-p", help="Target project path (defaults to current working directory)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
):
    """Display index statistics, configuration status, embedding model, and provider status."""
    settings = get_settings(path)
    stat = get_status(settings)

    if json_output:
        output_json(stat.model_dump(mode="json"))
        return

    table = Table(title="IRONCLAD Knowledge Base Status", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="dim", width=25)
    table.add_column("Value")

    table.add_row(
        "Status",
        "[bold green]Initialized[/bold green]" if stat.initialized else "[bold red]Not Initialized[/bold red]",
    )
    table.add_row("Knowledge Base Path", str(stat.knowledge_base_path))
    table.add_row("Total Documents", str(stat.total_files))
    table.add_row("Total Chunks", str(stat.total_chunks))
    table.add_row("Schema Version", str(stat.index_version))
    table.add_row("Last Ingested", stat.last_ingested_at or "(never)")
    table.add_row("Embedding Model", stat.embedding_model or "(not set)")
    table.add_row("Inference Provider", stat.inference_provider)
    table.add_row("Inference Model", stat.inference_model)
    table.add_row(
        "Integrity",
        "[bold green]Consistent[/bold green]" if stat.is_consistent else "[bold red]Inconsistencies Detected[/bold red]",
    )

    console.print(table)

    if stat.inconsistencies:
        console.print("\n[bold red]Inconsistencies / Warnings:[/bold red]")
        for inc in stat.inconsistencies:
            console.print(f"  • [yellow]{inc}[/yellow]")


@app.command()
def doctor(
    path: Optional[Path] = typer.Option(
        None, "--path", "-p", help="Target project path (defaults to current working directory)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
):
    """Diagnose configuration, permissions, storage, embedding-runtime, and inference connectivity."""
    settings = get_settings(path)
    report = run_diagnostics(settings)

    if json_output:
        output_json(report.model_dump(mode="json"))
        if not report.all_passed:
            raise typer.Exit(code=1)
        return

    table = Table(title="IRONCLAD System Doctor Diagnostics", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="dim", width=15)
    table.add_column("Check", style="bold", width=30)
    table.add_column("Status", width=10)
    table.add_column("Details")

    for check in report.checks:
        status_text = "[bold green]PASS[/bold green]" if check.passed else "[bold red]FAIL[/bold red]"
        msg = check.message
        if check.hint and not check.passed:
            msg += f"\n[dim yellow]Hint: {check.hint}[/dim yellow]"
        table.add_row(check.category, check.name, status_text, msg)

    console.print(table)

    if report.all_passed:
        console.print("\n[bold green]✔ All diagnostic checks passed![/bold green]")
    else:
        console.print("\n[bold red]✖ Some diagnostic checks failed. Review hints above.[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def config(
    path: Optional[Path] = typer.Option(
        None, "--path", "-p", help="Target project path (defaults to current working directory)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
):
    """Show configuration guidance and safe provider configuration without exposing secrets."""
    settings = get_settings(path)
    cfg_data = settings.safe_dict()

    if json_output:
        output_json(cfg_data)
        return

    table = Table(title="IRONCLAD Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="dim", width=25)
    table.add_column("Value")

    for k, v in cfg_data.items():
        table.add_row(k, str(v))

    console.print(table)
    console.print(
        "\n[dim]To modify configuration, edit .env or set environment variables prefixed with IRONCLAD_.[/dim]"
    )


if __name__ == "__main__":
    app()
