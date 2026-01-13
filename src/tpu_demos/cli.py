import argparse
import os
import subprocess
import sys

from tpu_demos.biology.medical_imaging import run_headless
from tpu_demos.launcher import launch_mission
from tpu_demos.ui import run_dashboard


def get_gcloud_project() -> str | None:
    """Attempts to retrieve the active project from gcloud config."""
    try:
        cmd = ["gcloud", "config", "get-value", "project"]
        # Use stderr=subprocess.DEVNULL to silence warnings if any
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=5
        )
        project = result.stdout.strip()
        if project and project != "(unset)":
            return project
    except Exception:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="TPU Demos CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Biology Dashboard
    subparsers.add_parser("biology", help="Run local Biology demo dashboard")

    # Worker (Remote)
    subparsers.add_parser("worker", help="Run headless worker (internal use)")

    # Launch (Mission Control)
    launch_parser = subparsers.add_parser("launch", help="Launch TPU mission")
    launch_parser.add_argument("project_id", nargs="?", help="Google Cloud Project ID")
    launch_parser.add_argument(
        "--project",
        dest="project_flag",
        help="Google Cloud Project ID (alternative flag)",
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command == "biology":
        run_dashboard()
    elif args.command == "worker":
        run_headless()
    elif args.command == "launch":
        # Resolution order: Argument -> --project flag -> Env Var -> gcloud config
        project_id = (
            args.project_id
            or args.project_flag
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT")
            or get_gcloud_project()
        )

        if not project_id:
            print("Error: Could not determine Google Cloud Project ID.")
            print(
                "Please provide it via argument, --project flag, set "
                "GOOGLE_CLOUD_PROJECT env var, "
                "or ensure 'gcloud config set project' is configured."
            )
            sys.exit(1)

        launch_mission(project_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
