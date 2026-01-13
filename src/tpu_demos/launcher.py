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
        with console.status(f"[bold green]{desc}...") as _:
            res = subprocess.run(cmd, capture_output=quiet, text=True, check=True)
            return res.stdout

    def create_vm(self) -> None:
        # Simple create-if-not-exists
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
        raw = subprocess.run(check_cmd, capture_output=True, text=True).stdout
        tpus = json.loads(raw)
        if any(t.get("name", "").endswith(self.vm_name) for t in tpus):
            console.log(f"[yellow]Reusing {self.vm_name}[/]")
            return

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
        self._run(create_cmd, "Provisioning TPU")

    def setup_remote(self) -> None:
        # 1. Upload code
        prep_cmd = [
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
        ]
        self._run(prep_cmd, "Prep remote")

        for f in ["src", "pyproject.toml"]:
            scp_cmd = [
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
            ]
            self._run(scp_cmd, f"Upload {f}")

        # 2. Install deps via pip directly
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
        self._run(install_cmd, "Install deps")

    def launch_worker(self) -> None:
        # Start the worker in the background
        cmd_str = (
            "export PYTHONPATH=$HOME/tpu-demos/src && "
            "nohup python3 -m tpu_demos.biology.medical_imaging "
            "> ~/tpu-demos/worker.log 2>&1 &"
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
        self._run(launch_cmd, "Launch worker")

    def poll_and_render(self, num_steps: int = 300) -> None:
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

        with Live(layout, refresh_per_second=4, screen=True):
            for _ in range(num_steps * 2):  # Poll up to 2x expected time
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
                    raw = subprocess.run(cat_cmd, capture_output=True, text=True).stdout
                    if raw:
                        m = json.loads(raw)
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
                except Exception:
                    pass
                time.sleep(1)

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
        self._run(delete_cmd, "Cleanup")


def launch_mission(project_id: str) -> None:
    launcher = TPULauncher(project_id)
    try:
        launcher.create_vm()
        launcher.setup_remote()
        launcher.launch_worker()
        launcher.poll_and_render()
    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted.[/]")
    finally:
        console.print(Panel("Destruction sequence initiated...", style="white on red"))
        launcher.cleanup()
