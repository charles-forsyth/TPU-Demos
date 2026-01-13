import sys

from tpu_demos.ui import run_dashboard


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "biology":
        run_dashboard()
    else:
        print("Usage: tpu-demos biology")


if __name__ == "__main__":
    main()
