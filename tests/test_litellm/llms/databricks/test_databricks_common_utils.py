import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm.llms.databricks.common_utils import DatabricksBase, DatabricksException


def test_databricks_validate_environment():
    databricks_base = DatabricksBase()

    with patch.object(
        databricks_base, "_get_databricks_credentials"
    ) as mock_get_credentials:
        try:
            databricks_base.databricks_validate_environment(
                api_key=None,
                api_base="my_api_base",
                endpoint_type="chat_completions",
                custom_endpoint=False,
                headers=None,
            )
        except Exception:
            pass
        mock_get_credentials.assert_called_once()


def test_databricks_missing_api_base_error_points_to_api_base_env_var():
    databricks_base = DatabricksBase()

    with pytest.raises(DatabricksException) as exc_info:
        databricks_base.databricks_validate_environment(
            api_key="my_api_key",
            api_base=None,
            endpoint_type="chat_completions",
            custom_endpoint=True,
            headers=None,
        )

    message = exc_info.value.message
    assert "Missing API Base" in message
    assert "{LLM_PROVIDER}_API_BASE" in message
    assert "{LLM_PROVIDER}_API_KEY" not in message
