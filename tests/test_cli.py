import os
import sys
from unittest.mock import patch

from tpu_demos.cli import get_gcloud_project, main


def test_get_gcloud_project_success():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "my-gcloud-project\n"
        assert get_gcloud_project() == "my-gcloud-project"


def test_get_gcloud_project_unset():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "(unset)\n"
        assert get_gcloud_project() is None


def test_get_gcloud_project_failure():
    with patch("subprocess.run", side_effect=Exception("Failed")):
        assert get_gcloud_project() is None


@patch("tpu_demos.cli.launch_mission")
def test_cli_launch_arg(mock_launch):
    with patch.object(sys, "argv", ["tpu-demos", "launch", "arg-project"]):
        main()
        mock_launch.assert_called_with("arg-project")


@patch("tpu_demos.cli.launch_mission")
def test_cli_launch_flag(mock_launch):
    args = ["tpu-demos", "launch", "--project", "flag-project"]
    with patch.object(sys, "argv", args):
        main()
        mock_launch.assert_called_with("flag-project")


@patch("tpu_demos.cli.launch_mission")
def test_cli_launch_env(mock_launch):
    with patch.object(sys, "argv", ["tpu-demos", "launch"]):
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "env-project"}):
            main()
            mock_launch.assert_called_with("env-project")


@patch("tpu_demos.cli.launch_mission")
@patch("tpu_demos.cli.get_gcloud_project", return_value="config-project")
def test_cli_launch_config_fallback(mock_get_config, mock_launch):
    with patch.object(sys, "argv", ["tpu-demos", "launch"]):
        with patch.dict(os.environ, {}, clear=True):
            main()
            mock_launch.assert_called_with("config-project")
