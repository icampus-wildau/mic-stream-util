"""Memory usage monitoring command module."""

from __future__ import annotations

import time

import click
import psutil
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--interval", "-i", type=float, default=1.0, help="Update interval in seconds (default: 1.0)")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed memory information")
def memory_usage(interval: float, detailed: bool):
    """Monitor memory usage in real-time."""
    console = Console()

    try:
        console.print("[bold blue]Memory Usage Monitor[/bold blue]")
        console.print(f"Update Interval: {interval:.1f}s")
        console.print(f"Detailed Mode: {'Yes' if detailed else 'No'}")
        console.print("Press Ctrl+C to stop")
        console.print("-" * 40)

        def create_memory_display(memory_info: psutil.virtual_memory, swap_info: psutil.swap_memory) -> Panel:
            """Create the memory usage display panel."""
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Metric", style="bold")
            table.add_column("Value", style="bold")
            table.add_column("Bar", style="bold")

            # Main memory usage
            mem_percent = memory_info.percent
            mem_bars = int(mem_percent * 50 / 100)
            mem_bar = "█" * min(mem_bars, 50)
            mem_color = "red" if mem_percent > 80 else "yellow" if mem_percent > 50 else "green"
            table.add_row("RAM", f"{mem_percent:.1f}%", f"[{mem_color}]{mem_bar:<50}[/{mem_color}]")

            # Memory details
            used_gb = memory_info.used / (1024**3)
            total_gb = memory_info.total / (1024**3)
            available_gb = memory_info.available / (1024**3)
            table.add_row("Used", f"{used_gb:.2f} GB", "")
            table.add_row("Total", f"{total_gb:.2f} GB", "")
            table.add_row("Available", f"{available_gb:.2f} GB", "")

            # Swap usage
            swap_percent = swap_info.percent
            swap_bars = int(swap_percent * 50 / 100)
            swap_bar = "█" * min(swap_bars, 50)
            swap_color = "red" if swap_percent > 80 else "yellow" if swap_percent > 50 else "green"
            table.add_row("Swap", f"{swap_percent:.1f}%", f"[{swap_color}]{swap_bar:<50}[/{swap_color}]")

            # Swap details
            swap_used_gb = swap_info.used / (1024**3)
            swap_total_gb = swap_info.total / (1024**3)
            table.add_row("Swap Used", f"{swap_used_gb:.2f} GB", "")
            table.add_row("Swap Total", f"{swap_total_gb:.2f} GB", "")

            if detailed:
                # Additional memory information
                table.add_row("", "", "")  # Empty row for spacing
                table.add_row("Cached", f"{memory_info.cached / (1024**3):.2f} GB", "")
                table.add_row("Buffers", f"{memory_info.buffers / (1024**3):.2f} GB", "")
                table.add_row("Shared", f"{memory_info.shared / (1024**3):.2f} GB", "")
                table.add_row("Slab", f"{memory_info.slab / (1024**3):.2f} GB", "")

            return Panel(table, title="[bold]Memory Usage[/bold]", border_style="blue")

        with Live(create_memory_display(psutil.virtual_memory(), psutil.swap_memory()), refresh_per_second=2) as live:
            while True:
                try:
                    # Get memory information
                    memory_info = psutil.virtual_memory()
                    swap_info = psutil.swap_memory()

                    # Update display
                    live.update(create_memory_display(memory_info, swap_info))

                    time.sleep(interval)

                except KeyboardInterrupt:
                    break

        console.print("\n[bold green]Memory monitoring stopped[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error during memory monitoring: {e}[/bold red]")
        raise click.Abort()
