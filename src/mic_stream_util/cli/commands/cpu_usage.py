"""CPU usage monitoring command module."""

from __future__ import annotations

import click
import psutil
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--interval", "-i", type=float, default=1.0, help="Update interval in seconds (default: 1.0)")
@click.option("--cores", "-c", type=int, default=0, help="Number of cores to display (0 = all, default: 0)")
def cpu_usage(interval: float, cores: int):
    """Monitor CPU usage in real-time."""
    console = Console()

    try:
        console.print("[bold blue]CPU Usage Monitor[/bold blue]")
        console.print(f"Update Interval: {interval:.1f}s")
        console.print(f"Display Cores: {'All' if cores == 0 else cores}")
        console.print("Press Ctrl+C to stop")
        console.print("-" * 40)

        def create_cpu_display(cpu_percent: float, cpu_per_core: list[float], memory_percent: float) -> Panel:
            """Create the CPU usage display panel."""
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Metric", style="bold")
            table.add_column("Value", style="bold")
            table.add_column("Bar", style="bold")

            # Overall CPU usage
            cpu_bars = int(cpu_percent * 50 / 100)
            cpu_bar = "█" * min(cpu_bars, 50)
            cpu_color = "red" if cpu_percent > 80 else "yellow" if cpu_percent > 50 else "green"
            table.add_row("CPU Total", f"{cpu_percent:.1f}%", f"[{cpu_color}]{cpu_bar:<50}[/{cpu_color}]")

            # Memory usage
            mem_bars = int(memory_percent * 50 / 100)
            mem_bar = "█" * min(mem_bars, 50)
            mem_color = "red" if memory_percent > 80 else "yellow" if memory_percent > 50 else "green"
            table.add_row("Memory", f"{memory_percent:.1f}%", f"[{mem_color}]{mem_bar:<50}[/{mem_color}]")

            # Per-core CPU usage
            if cores == 0 or cores > len(cpu_per_core):
                display_cores = cpu_per_core
            else:
                display_cores = cpu_per_core[:cores]

            for i, core_usage in enumerate(display_cores):
                core_bars = int(core_usage * 50 / 100)
                core_bar = "█" * min(core_bars, 50)
                core_color = "red" if core_usage > 80 else "yellow" if core_usage > 50 else "green"
                table.add_row(f"Core {i}", f"{core_usage:.1f}%", f"[{core_color}]{core_bar:<50}[/{core_color}]")

            return Panel(table, title="[bold]CPU Usage[/bold]", border_style="blue")

        with Live(create_cpu_display(0.0, [], 0.0), refresh_per_second=2) as live:
            while True:
                try:
                    # Get CPU usage
                    cpu_percent = psutil.cpu_percent(interval=interval)
                    cpu_per_core = psutil.cpu_percent(interval=interval, percpu=True)
                    memory_percent = psutil.virtual_memory().percent

                    # Update display
                    live.update(create_cpu_display(cpu_percent, cpu_per_core, memory_percent))

                except KeyboardInterrupt:
                    break

        console.print("\n[bold green]CPU monitoring stopped[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error during CPU monitoring: {e}[/bold red]")
        raise click.Abort()
