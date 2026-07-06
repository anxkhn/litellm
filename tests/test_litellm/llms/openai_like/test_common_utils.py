import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.openai_like.common_utils import OpenAILikeBase, OpenAILikeError


def test_missing_api_base_error_points_to_api_base_env_var():
    openai_like_base = OpenAILikeBase()

    with pytest.raises(OpenAILikeError) as exc_info:
        openai_like_base._validate_environment(
            api_key="my_api_key",
            api_base=None,
            endpoint_type="chat_completions",
            headers=None,
            custom_endpoint=False,
        )

    message = exc_info.value.message
    assert "Missing API Base" in message
    assert "{LLM_PROVIDER}_API_BASE" in message
    assert "{LLM_PROVIDER}_API_KEY" not in message


def test_missing_api_key_error_points_to_api_key_env_var():
    openai_like_base = OpenAILikeBase()

    with pytest.raises(OpenAILikeError) as exc_info:
        openai_like_base._validate_environment(
            api_key=None,
            api_base="https://my-api-base",
            endpoint_type="chat_completions",
            headers=None,
            custom_endpoint=False,
        )

    message = exc_info.value.message
    assert "Missing API Key" in message
    assert "{LLM_PROVIDER}_API_KEY" in message
    assert "{LLM_PROVIDER}_API_BASE" not in message
