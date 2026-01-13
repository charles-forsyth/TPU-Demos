import json
import subprocess

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

console = Console()

CODE_SNIPPET = """
@partial(jax.jit, static_argnames=["tx", "model"])
def train_step(params, batch_stats, opt_state, batch, tx, model):
    def loss_fn(params, batch_stats):
        logits, new_model_state = model.apply(
            {"params": params, "batch_stats": batch_stats},
            batch["image"],
            train=True,
            mutable=["batch_stats"],
        )
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        return loss, new_model_state

    grads = jax.grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state)
    return params, new_model_state["batch_stats"], opt_state, loss
"""

WELCOME_TEXT = """
# 🚀 Google Cloud TPU v5e Demo

## Workload: Medical Imaging (ResNet-50)
This demo showcases the power of **JAX**, **Flax**, and **XLA** running on a **TPU v5e**
accelerator.

### The Mission
1.  **Provision** a v5litepod-1 TPU VM.
2.  **Deploy** the ResNet-50 training code.
3.  **Execute** a high-throughput training loop.
4.  **Visualize** the results.
5.  **Incinerate** the resources.
"""


class TPULauncher:
    def __init__(
        self, project_id: str, zone: str = "us-west1-c", vm_name: str = "tpu-demo-vm"
    ):
        self.project_id = project_id
        self.zone = zone
        self.vm_name = vm_name

    def _run(self, cmd: list[str], desc: str, quiet: bool = True) -> None:
        """Runs a command with a spinner."""
        if not quiet:
            # For streaming output, we don't use the spinner context manager
            console.log(f"[blue]Action: {desc}[/]")
            subprocess.run(cmd, check=True, capture_output=False, text=True)
            console.log(f"[green]✓ {desc} complete.[/]")
            return

        with console.status(f"[bold green]{desc}...") as _:
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                console.log(f"[green]✓ {desc} complete.[/]")
            except subprocess.CalledProcessError as e:
                console.log(f"[bold red]✗ {desc} failed![/]")
                if e.stderr:
                    console.print(f"[red]Error Details:[/red]\n{e.stderr}")
                raise e

    def show_welcome(self) -> None:
        console.clear()
        console.print(
            Panel(Markdown(WELCOME_TEXT), title="TPU Demos", border_style="cyan")
        )
        code_panel = Panel(
            Syntax(CODE_SNIPPET, "python", theme="monokai", line_numbers=True),
            title="JAX Code Preview",
            border_style="green",
        )
        console.print(code_panel)
        Prompt.ask("\n[bold yellow]Press Enter to launch mission[/]")

    def provision_vm(self) -> None:
        # Check existence
        check = [
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
            with console.status("[bold green]Checking existing resources...") as _:
                raw = subprocess.run(check, capture_output=True, text=True).stdout
                tpus = json.loads(raw) if raw else []
                if any(t.get("name", "").endswith(self.vm_name) for t in tpus):
                    console.log(f"[yellow]VM '{self.vm_name}' found. Reusing.[/]")
                    return
        except Exception:
            pass

        # Create
        create = [
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
        self._run(create, "Provisioning TPU VM")

    def deploy_code(self) -> None:
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
                "rm -rf ~/tpu-demos && mkdir -p ~/tpu-demos",
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
                f"Uploading {f}",
            )

        pkg_url = "https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        install = [
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
        self._run(install, "Installing dependencies")

    def run_job(self) -> None:
        console.rule("[bold green]Starting Job Execution[/]")
        cmd_str = (
            "export PYTHONPATH=$HOME/tpu-demos/src && "
            "cd ~/tpu-demos && "
            "python3 -u -m tpu_demos.biology.medical_imaging"
        )
        cmd = [
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
            "--ssh-flag=-t",
        ]
        self._run(cmd, "Running Job", quiet=False)

    def display_results(self) -> None:
        console.log("[blue]Downloading results...[/]")
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "scp",
            f"{self.vm_name}:~/tpu-demos/results.json",
            "job_results.json",
            "--zone",
            self.zone,
            "--project",
            self.project_id,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)

            with open("job_results.json") as f:
                data = json.load(f)

            table = Table(title="Mission Results", border_style="green")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Final Loss", f"{data['final_loss']:.4f}")
            table.add_row("Throughput", f"{data['avg_throughput']:.2f} img/s")
            table.add_row("Total Time", f"{data['total_time']:.2f} s")
            table.add_row("Status", "[bold green]SUCCESS[/]")

            console.print(Panel(table, border_style="green"))

        except Exception:
            console.log(
                "[yellow]⚠ Could not retrieve results file (maybe job failed?)[/]"
            )

    def cleanup(self) -> None:
        Prompt.ask("\n[bold yellow]Press Enter to incinerate resources[/]")
        console.rule("[bold red]Cleanup[/]")
        cmd = [
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
        self._run(cmd, "Deleting TPU VM")


def launch_mission(project_id: str) -> None:
    launcher = TPULauncher(project_id)
    try:
        launcher.show_welcome()
        launcher.provision_vm()
        launcher.deploy_code()
        launcher.run_job()
        launcher.display_results()
    except KeyboardInterrupt:
        console.print("\n[yellow]Job cancelled by user.[/]")
    except Exception as e:
        console.print("\n[bold red]Job Failed:[/]")
        console.print(e)
    finally:
        launcher.cleanup()
