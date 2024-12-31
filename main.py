"""
Copyright © 2024 Erick Aleman
Contact: Erick@EACognitive.com

This file is part of the DMSRS implementation.
Contains integration and utility functions.
See LICENSE.md for terms of use.
"""

# main.py

import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.traceback import install
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Install rich traceback handler
install(show_locals=True)

# Initialize Rich console
console = Console()

# Configure logging with Rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger("main_script")

def execute_with_timeout(func, timeout, *args, **kwargs):
    """Run a function with a timeout to prevent hangs."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            logger.error("[red]Timeout occurred during execution.[/red]")
            raise TimeoutError(f"Execution of {func.__name__} exceeded {timeout} seconds.")

def main():
    # Get the question from the user first
    console.print(Panel.fit("[puple]Divrey Yoel Analysis Pipeline[/puple]"))
    question = console.input("[bold green]Please enter your question: [/bold green]").strip()

    if not question:
        logger.error("[red]No question provided. Exiting...[/red]")
        return

    # Import the main functions from each step
    from step_1 import main as step_1_main
    from step_2 import main as step_2_main
    from step_3 import main as step_3_main
    from step_4 import main as step_4_main

    # Define the steps and their corresponding functions
    steps = [
        {"name": "Step 1: Understanding The Question", "function": lambda: execute_with_timeout(step_1_main, 300, question)},  # 300s timeout
        {"name": "Step 2: Reading", "function": step_2_main},
        {"name": "Step 3: Predicting The Future", "function": step_3_main},
        {"name": "Step 4: Making Magic", "function": step_4_main},
    ]

    # Start the progress tracker
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Running steps...", total=len(steps))

        # Iterate over the steps
        for step in steps:
            progress.update(task, description=f"[bold blue]{step['name']}...[/bold blue]")
            try:
                logger.info(f"[cyan]Starting {step['name']}...[/cyan]")
                step['function']()
                progress.advance(task)
                logger.info(f"[cyan]Executing Step 1 with question: {question}[/cyan]")

            except TimeoutError as e:
                logger.error(f"[red]Timeout in {step['name']}: {e}[/red]")
                console.print(Panel.fit(f"[red]Timeout in {step['name']}: {e}[/red]", title="Error Details", border_style="red"))
                exit(1)

            except Exception as e:
                logger.error(f"[red]An error occurred in {step['name']}: {e}[/red]")
                console.print(Panel.fit(f"[red]Error in {step['name']}: {e}[/red]", title="Error Details", border_style="red"))
                exit(1)

    console.print(Panel.fit(f"[green]All steps completed successfully![/green]", title="Process Complete", border_style="green"))

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        from app import app
        app.run(host='0.0.0.0', port=8080, debug=True)
    else:
        main()
