import json
import subprocess

from rich.console import Console
from rich.panel import Panel

console = Console()


class TPULauncher:
    def __init__(
        self, project_id: str, zone: str = "us-west1-c", vm_name: str = "tpu-demo-vm"
    ):
        self.project_id = project_id
        self.zone = zone
        self.vm_name = vm_name

    def _run(self, cmd: list[str], desc: str, quiet: bool = True) -> None:
        """Runs a command. If quiet=False, output streams to stdout."""
        console.log(f"[blue]Action: {desc}[/]")
        # For non-quiet commands (like running the job), we want to see output live.
        # subprocess.run with capture_output=False streams to our stdout/stderr
        try:
            subprocess.run(cmd, check=True, capture_output=quiet, text=True)
            console.log(f"[green]✓ {desc} complete.[/]")
        except subprocess.CalledProcessError as e:
            console.log(f"[bold red]✗ {desc} failed![/]")
            if quiet and e.stderr:
                console.print(f"[red]Error Details:[/red]\n{e.stderr}")
            raise e

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
            raw = subprocess.run(check, capture_output=True, text=True).stdout
            tpus = json.loads(raw) if raw else []
            if any(t.get("name", "").endswith(self.vm_name) for t in tpus):
                console.log(f"[yellow]VM '{self.vm_name}' found. Reusing.[/]")
                return
        except Exception:
            pass  # Ignore listing errors, try create

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
        # Cleanup remote dir first to ensure clean state
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

        # Upload
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

        # Install Deps
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
        # We run this WITH output streaming (quiet=False) so user sees logs.
        # We use stdbuf to force unbuffered output if possible, or python -u
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
            "--ssh-flag=-t",  # Force TTY for color/formatting preservation
        ]
        self._run(cmd, "Running Job", quiet=False)

    def download_results(self) -> None:
        console.log("[blue]Downloading results...[/]")
        # We assume the job writes to ~/tpu-demos/results.json
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
            console.log("[green]✓ Results saved to 'job_results.json'[/]")

            # Print summary
            with open("job_results.json") as f:
                data = json.load(f)
                console.print(Panel(json.dumps(data, indent=2), title="Job Summary"))

        except Exception:
            console.log(
                "[yellow]⚠ Could not retrieve results file (maybe job failed?)[/]"
            )

    def cleanup(self) -> None:
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
        launcher.provision_vm()
        launcher.deploy_code()
        launcher.run_job()
        launcher.download_results()
    except KeyboardInterrupt:
        console.print("\n[yellow]Job cancelled by user.[/]")
    except Exception as e:
        console.print(f"\n[bold red]Job Failed:[/] {e}")
    finally:
        launcher.cleanup()
