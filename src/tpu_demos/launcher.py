import json
import subprocess
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from tpu_demos.ui import make_code_panel, make_header, make_metrics_table

console = Console()


class TPULauncher:
    def __init__(
        self, project_id: str, zone: str = "us-west1-c", vm_name: str = "tpu-demo-vm"
    ):
        self.project_id = project_id
        self.zone = zone
        self.vm_name = vm_name

    def _run(self, cmd: list[str], desc: str, quiet: bool = True) -> str:
        cmd_joined = " ".join(cmd)
        console.log(f"[blue]Executing: {cmd_joined}[/]")
        with console.status(f"[bold green]{desc}...") as _:
            try:
                res = subprocess.run(cmd, capture_output=quiet, text=True, check=True)
                console.log(f"[green]✓ {desc} complete.[/green]")
                return res.stdout
            except subprocess.CalledProcessError as e:
                console.log(f"[bold red]✗ {desc} failed![/bold red]")
                if quiet:
                    console.print(f"[red]Error:[/red] {e.stderr}")
                raise e

    def create_vm(self) -> None:
        console.log("[bold cyan]Mission Step 1: Resource Provisioning[/]")
        check_cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            "--zone",
            self.zone,
            "--project",
            self.project_id,
            "--format=json",
        ]
        try:
            raw = subprocess.run(check_cmd, capture_output=True, text=True).stdout
            tpus = json.loads(raw) if raw else []
            if any(t.get("name", "").endswith(self.vm_name) for t in tpus):
                console.log(f"[yellow]Reusing existing VM: {self.vm_name}[/]")
                return
        except Exception as e:
            console.log(f"[yellow]Could not list VMs, attempting creation: {e}[/]")

        create_cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "create",
            self.vm_name,
            "--zone",
            self.zone,
            "--accelerator-type",
            "v5litepod-1",
            "--version",
            "v2-alpha-tpuv5-lite",
            "--project",
            self.project_id,
            "--quiet",
        ]
        self._run(create_cmd, "Provisioning TPU VM")

    def setup_remote(self) -> None:
        console.log("[bold cyan]Mission Step 2: Code Deployment[/]")
        self._run(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "ssh",
                self.vm_name,
                "--zone",
                self.zone,
                "--project",
                self.project_id,
                "--command",
                "mkdir -p ~/tpu-demos",
            ],
            "Preparing remote directory",
        )

        for f in ["src", "pyproject.toml"]:
            self._run(
                [
                    "gcloud",
                    "compute",
                    "tpus",
                    "tpu-vm",
                    "scp",
                    "--recurse",
                    f,
                    f"{self.vm_name}:~/tpu-demos/",
                    "--zone",
                    self.zone,
                    "--project",
                    self.project_id,
                ],
                f"Upload {f}",
            )

        console.log("[bold cyan]Mission Step 3: Dependency Installation[/]")
        pkg_url = "https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        install_cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            self.vm_name,
            "--zone",
            self.zone,
            "--project",
            self.project_id,
            "--command",
            f"pip install 'jax[tpu]' -f {pkg_url} flax optax",
        ]
        self._run(install_cmd, "Installing dependencies via pip")

    def launch_worker(self) -> None:
        console.log("[bold cyan]Mission Step 4: Launching Background Worker[/]")
        # Cleanup old artifacts
        self._run(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "ssh",
                self.vm_name,
                "--zone",
                self.zone,
                "--project",
                self.project_id,
                "--command",
                "rm -f ~/tpu-demos/metrics.json ~/tpu-demos/worker.log",
            ],
            "Cleaning up previous run artifacts",
        )

        cmd_str = (
            "export PYTHONPATH=$HOME/tpu-demos/src && "
            "cd ~/tpu-demos && "
            "nohup python3 -m tpu_demos.biology.medical_imaging "
            "> worker.log 2>&1 &"
        )
        launch_cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            self.vm_name,
            "--zone",
            self.zone,
            "--project",
            self.project_id,
            "--command",
            cmd_str,
        ]
        self._run(launch_cmd, "Starting TPU workload")

    def inspect_logs(self) -> None:
        """Fetches and displays the worker log from the remote VM."""
        console.log("[bold yellow]Fetching worker logs for debugging...[/]")
        try:
            log_cmd = [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "ssh",
                self.vm_name,
                "--zone",
                self.zone,
                "--project",
                self.project_id,
                "--command",
                "cat ~/tpu-demos/worker.log",
            ]
            logs = subprocess.run(
                log_cmd, capture_output=True, text=True, check=False
            ).stdout
            if logs:
                console.print(
                    Panel(logs, title="Remote Worker Log", border_style="red")
                )
            else:
                console.print("[red]No logs found in ~/tpu-demos/worker.log[/]")
        except Exception as e:
            console.print(f"[red]Failed to fetch logs: {e}[/]")

    def poll_and_render(self, num_steps: int = 300) -> None:
        console.log("[bold cyan]Mission Step 5: Real-time Telemetry Active[/]")
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
        task_id = progress.add_task("[cyan]TPU Training", total=num_steps)
        layout["footer"].update(Panel(progress, border_style="blue"))

        received_data = False
        with Live(layout, refresh_per_second=4, screen=True):
            for _ in range(num_steps * 5):  # Poll up to 5x expected time
                try:
                    cat_cmd = [
                        "gcloud",
                        "compute",
                        "tpus",
                        "tpu-vm",
                        "ssh",
                        self.vm_name,
                        "--zone",
                        self.zone,
                        "--project",
                        self.project_id,
                        "--command",
                        "cat ~/tpu-demos/metrics.json",
                    ]
                    # Don't check=True here so we handle empty/missing file gracefully
                    res = subprocess.run(cat_cmd, capture_output=True, text=True)
                    raw = res.stdout
                    if raw:
                        try:
                            m = json.loads(raw)
                            received_data = True
                            step, loss, tp, status = (
                                m["step"],
                                m["loss"],
                                m["throughput"],
                                m["status"],
                            )
                            layout["right"].update(
                                Panel(
                                    make_metrics_table(step, loss, tp, status),
                                    title="Live TPU Telemetry",
                                    border_style="red",
                                )
                            )
                            progress.update(
                                task_id,
                                completed=step,
                                description=f"Training... (Loss: {loss:.3f})",
                            )
                            if step >= num_steps:
                                break
                        except json.JSONDecodeError:
                            pass  # File might be partially written
                except Exception:
                    pass
                time.sleep(1)

        if not received_data:
            console.print("[bold red]❌ No telemetry received![/]")
            # We raise an error here so the main loop catches it and we can inspect logs
            raise RuntimeError("Worker failed to produce metrics.")

    def cleanup(self) -> None:
        delete_cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            self.vm_name,
            "--zone",
            self.zone,
            "--project",
            self.project_id,
            "--quiet",
        ]
        self._run(delete_cmd, "Incinerating resources")


def launch_mission(project_id: str) -> None:
    launcher = TPULauncher(project_id)
    try:
        console.print(Panel("[bold white]🛑 MISSION START[/]", style="red on blue"))
        launcher.create_vm()
        launcher.setup_remote()
        launcher.launch_worker()
        launcher.poll_and_render()
    except KeyboardInterrupt:
        console.print("\n[yellow]Mission aborted by pilot.[/]")
    except Exception as e:
        console.print(f"\n[bold red]CRITICAL FAILURE:[/] {e}")
        # Fetch logs if we failed
        launcher.inspect_logs()
        # Re-raise to ensure non-zero exit code if needed, or just let cleanup happen
    finally:
        console.print(Panel("Initiating Re-entry Protocol...", style="white on red"))
        launcher.cleanup()
        console.print("[bold green]✔ Splashdown confirmed.[/]")
