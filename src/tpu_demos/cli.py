import os
import sys

from tpu_demos.launcher import launch_mission
from tpu_demos.ui import run_dashboard


def get_project_id() -> str | None:
    """Retrieves project ID from arguments or environment variables."""
    # 1. Check command line argument
    if len(sys.argv) >= 3:
        return sys.argv[2]

    # 2. Check environment variables
    return os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: tpu-demos [biology|launch [project_id]]")
        print("\nNote: 'launch' uses GOOGLE_CLOUD_PROJECT if project_id is omitted.")
        return

    command = sys.argv[1]

    if command == "biology":
        run_dashboard()
    elif command == "launch":
        project_id = get_project_id()
        if not project_id:
            print("Error: Missing project_id.")
            print("Usage: tpu-demos launch [project_id]")
            print("Or set GOOGLE_CLOUD_PROJECT environment variable.")
            sys.exit(1)

        launch_mission(project_id)
    else:
        print(f"Unknown command: {command}")
        print("Usage: tpu-demos [biology|launch [project_id]]")


if __name__ == "__main__":
    main()
