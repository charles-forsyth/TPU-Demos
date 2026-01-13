import os
import sys

from tpu_demos.biology.medical_imaging import run_headless
from tpu_demos.launcher import launch_mission
from tpu_demos.ui import run_dashboard


def get_project_id() -> str | None:
    """Retrieves project ID from arguments or environment variables."""
    if len(sys.argv) >= 3:
        return sys.argv[2]
    return os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: tpu-demos [biology|worker|launch [project_id]]")
        return

    command = sys.argv[1]

    if command == "biology":
        run_dashboard()
    elif command == "worker":
        # Run the headless worker on the node
        run_headless()
    elif command == "launch":
        # Identify project ID
        project_id = get_project_id()
        if not project_id:
            print("Error: project_id required.")
            sys.exit(1)
        launch_mission(project_id)
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
