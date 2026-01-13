import os
import sys
from unittest.mock import patch

from tpu_demos.cli import get_project_id


def test_get_project_id_from_args():
    with patch.object(sys, "argv", ["tpu-demos", "launch", "my-project-arg"]):
        assert get_project_id() == "my-project-arg"


def test_get_project_id_from_env():
    with patch.object(sys, "argv", ["tpu-demos", "launch"]):
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "my-project-env"}):
            assert get_project_id() == "my-project-env"


def test_get_project_id_from_env_gcp():
    with patch.object(sys, "argv", ["tpu-demos", "launch"]):
        with patch.dict(os.environ, {"GCP_PROJECT": "my-project-gcp"}):
            # Ensure GOOGLE_CLOUD_PROJECT is NOT set
            if "GOOGLE_CLOUD_PROJECT" in os.environ:
                del os.environ["GOOGLE_CLOUD_PROJECT"]
            assert get_project_id() == "my-project-gcp"


def test_get_project_id_none():
    with patch.object(sys, "argv", ["tpu-demos", "launch"]):
        with patch.dict(os.environ, {}, clear=True):
            assert get_project_id() is None
