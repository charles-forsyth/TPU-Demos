import subprocess

from rich.console import Console
from rich.panel import Panel

console = Console()


class TPULauncher:
    def __init__(
        self,
        project_id: str,
        zone: str = "us-west1-c",
        vm_name: str = "tpu-demo-vm",
        accelerator_type: str = "v5litepod-1",
        version: str = "v2-alpha-tpuv5-lite",
    ):
        self.project_id = project_id
        self.zone = zone
        self.vm_name = vm_name
        self.accelerator_type = accelerator_type
        self.version = version

    def _run_command(self, cmd: list[str], description: str) -> None:
        """Runs a shell command and streams output to the console."""
        with console.status(f"[bold green]{description}...") as _:
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                console.log(f"[green]✓ {description} complete.[/green]")
                # console.log(result.stdout) # Verbose logging if needed
            except subprocess.CalledProcessError as e:
                console.log(f"[bold red]✗ {description} failed![/bold red]")
                console.print(f"[red]Error:[/red] {e.stderr}")
                raise e

    def check_gcloud(self) -> None:
        """Verifies gcloud is installed and authenticated."""
        self._run_command(["gcloud", "--version"], "Checking gcloud installation")

    def create_vm(self) -> None:
        """Creates the TPU VM."""
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "create",
            self.vm_name,
            "--zone",
            self.zone,
            "--accelerator-type",
            self.accelerator_type,
            "--version",
            self.version,
            "--project",
            self.project_id,
            "--quiet",
        ]
        self._run_command(
            cmd, f"Provisioning TPU VM '{self.vm_name}' (this may take a few minutes)"
        )

    def delete_vm(self) -> None:
        """Deletes the TPU VM."""
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
        self._run_command(cmd, f"Deleting TPU VM '{self.vm_name}'")

    def deploy_code(self) -> None:
        """Deploys the current project code to the VM."""
        # 1. Install uv on the VM
        install_uv_cmd = [
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
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
        ]
        self._run_command(install_uv_cmd, "Installing 'uv' on remote VM")

        # 2. SCP the current directory to the VM
        files_to_send = ["src", "pyproject.toml", "README.md"]
        for file in files_to_send:
            cmd = [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "scp",
                "--recurse",
                file,
                f"{self.vm_name}:~/tpu-demos/",
                "--zone",
                self.zone,
                "--project",
                self.project_id,
            ]
            self._run_command(cmd, f"Uploading {file}")

    def install_dependencies(self) -> None:
        """Installs dependencies on the VM."""
        # Using uv to install the project
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
            "cd ~/tpu-demos && ~/.cargo/bin/uv sync",
        ]
        self._run_command(cmd, "Installing dependencies on remote VM")

    def run_demo(self, demo_name: str) -> None:
        """Runs the specified demo on the VM."""
        console.print(
            Panel(
                f"[bold white]🚀 Launching {demo_name} on TPU...[/bold white]",
                style="blue",
            )
        )

        # We want to see the output live, so we use subprocess.Popen
        # or run without capture_output
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
            f"cd ~/tpu-demos && ~/.cargo/bin/uv run tpu-demos {demo_name}",
            "--ssh-flag=-t",  # Force TTY for Rich colors
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            console.log("[red]Demo execution failed or interrupted.[/red]")


def launch_mission(project_id: str, demo: str = "biology") -> None:
    launcher = TPULauncher(project_id=project_id)

    try:
        console.print(
            Panel("[bold]🛑 Mission Control Initiated[/bold]", style="red on white")
        )

        # 1. Check Prereqs
        launcher.check_gcloud()

        # 2. Provision
        launcher.create_vm()

        # 3. Setup
        # Create remote dir first
        launcher._run_command(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "ssh",
                launcher.vm_name,
                "--zone",
                launcher.zone,
                "--project",
                launcher.project_id,
                "--command",
                "mkdir -p ~/tpu-demos",
            ],
            "Preparing remote directory",
        )
        launcher.deploy_code()
        launcher.install_dependencies()

        # 4. Execute
        launcher.run_demo(demo)

    except KeyboardInterrupt:
        console.print(
            "\n[bold yellow]⚠️ Mission Aborted by Pilot (Ctrl+C)[/bold yellow]"
        )
    except Exception as e:
        console.print(f"\n[bold red]🔥 Mission Failed:[/bold red] {e}")
    finally:
        # 5. Cleanup
        console.print(
            Panel(
                "[bold]🧹 Re-entry... Resources will be incinerated.[/bold]",
                style="white on red",
            )
        )
        launcher.delete_vm()
        console.print(
            "[bold green]✔ Splashdown confirmed. Resources destroyed.[/bold green]"
        )
