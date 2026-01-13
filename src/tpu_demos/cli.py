import sys

from tpu_demos.launcher import launch_mission
from tpu_demos.ui import run_dashboard


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: tpu-demos [biology|launch <project_id>]")
        return

    command = sys.argv[1]

    if command == "biology":
        run_dashboard()
    elif command == "launch":
        if len(sys.argv) < 3:
            print("Usage: tpu-demos launch <project_id>")
            return
        project_id = sys.argv[2]
        launch_mission(project_id)
    else:
        print(f"Unknown command: {command}")
        print("Usage: tpu-demos [biology|launch <project_id>]")


if __name__ == "__main__":
    main()
