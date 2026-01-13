import time

from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from tpu_demos.biology.medical_imaging import training_loop

CODE_SNIPPET = """
@jax.jit
def train_step(params, opt_state, batch):
    def loss_fn(params):
        logits = model.apply({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        return loss

    grads = jax.grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
"""


def make_header() -> Panel:
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_row("[b white]🚀 Biology Demo: Medical Imaging ResNet-50[/b white]")
    return Panel(grid, style="white on blue", box=box.HEAVY)


def make_code_panel() -> Panel:
    syntax = Syntax(CODE_SNIPPET, "python", theme="monokai", line_numbers=True)
    return Panel(syntax, title="[b]JAX/Flax Implementation[/b]", border_style="green")


def make_metrics_table(step: int, loss: float, throughput: float, status: str) -> Table:
    table = Table(expand=True, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta", justify="right")

    status_style = "green" if status == "RUNNING" else "yellow"
    table.add_row("Status", f"[bold {status_style}]{status}[/]")
    table.add_row("Step", f"{step}")
    table.add_row("Loss", f"{loss:.4f}")
    table.add_row("Throughput", f"{throughput:.1f} img/s")

    return table


def run_dashboard(num_steps: int = 300) -> None:
    layout = Layout()

    layout.split(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3),
    )

    layout["body"].split_row(
        Layout(name="left", ratio=1), Layout(name="right", ratio=1)
    )

    layout["header"].update(make_header())
    layout["left"].update(make_code_panel())

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        expand=True,
    )
    task_id = progress.add_task("[cyan]Training Progress", total=num_steps)
    layout["footer"].update(Panel(progress, border_style="blue"))

    with Live(layout, refresh_per_second=10, screen=True):
        for metrics in training_loop(num_steps=num_steps):
            step = metrics["step"]
            loss = metrics["loss"]
            throughput = metrics["throughput"]
            status = str(metrics["status"])

            # Update Metrics Panel
            metrics_panel = Panel(
                make_metrics_table(step, loss, throughput, status),
                title="[b]Live Telemetry[/b]",
                border_style="red",
            )
            layout["right"].update(metrics_panel)

            # Update Progress
            progress.update(
                task_id,
                completed=step,
                description=f"[cyan]Training... (Loss: {loss:.3f})",
            )

            # Simulate a slight delay for visual pacing if it's too fast on CPU
            if throughput > 1000:
                time.sleep(0.01)


if __name__ == "__main__":
    run_dashboard()
