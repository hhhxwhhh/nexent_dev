import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sdk.nexent.core.tools import analyze_image_tool
from sdk.nexent.core.tools.analyze_image_tool import AnalyzeImageTool
from sdk.nexent.core.utils.observer import MessageObserver, ProcessType


@pytest.fixture
def mock_storage_client():
    class DummyStorage:
        pass

    return DummyStorage()


@pytest.fixture
def mock_vlm_model():
    return MagicMock()


@pytest.fixture
def mock_prompt_loader(monkeypatch):
    calls = []

    def _fake_get_prompt(template_type, language=None, **_):
        calls.append((template_type, language))
        return {"system_prompt": "Describe {{ query }}"}

    monkeypatch.setattr(
        analyze_image_tool,
        "get_prompt_template",
        _fake_get_prompt,
    )
    return calls


@pytest.fixture
def observer_en():
    observer = MagicMock(spec=MessageObserver)
    observer.lang = "en"
    return observer


@pytest.fixture
def observer_zh():
    observer = MagicMock(spec=MessageObserver)
    observer.lang = "zh"
    return observer


@pytest.fixture
def tool(observer_en, mock_vlm_model, mock_storage_client):
    return AnalyzeImageTool(
        observer=observer_en,
        vlm_model=mock_vlm_model,
        storage_client=mock_storage_client,
    )


class TestAnalyzeImageTool:
    def test_forward_impl_success_with_multiple_images(
        self, tool, mock_vlm_model, mock_prompt_loader
    ):
        mock_vlm_model.analyze_image.side_effect = [
            SimpleNamespace(content="First image analysis"),
            SimpleNamespace(content="Second image analysis"),
        ]

        result = tool._forward_impl([b"img1", b"img2"], "What is shown?")

        assert result == ["First image analysis", "Second image analysis"]
        assert mock_vlm_model.analyze_image.call_count == 2
        for call in mock_vlm_model.analyze_image.call_args_list:
            assert hasattr(call.kwargs["image_input"], "read")
        assert mock_prompt_loader == [("analyze_image", "en")]

    def test_forward_impl_zh_observer_messages(
        self, observer_zh, mock_vlm_model, mock_storage_client, mock_prompt_loader
    ):
        tool = AnalyzeImageTool(
            observer=observer_zh,
            vlm_model=mock_vlm_model,
            storage_client=mock_storage_client,
        )
        mock_vlm_model.analyze_image.return_value = SimpleNamespace(content="描述")

        result = tool._forward_impl([b"img"], "问题")

        assert result == ["描述"]
        assert mock_prompt_loader == [("analyze_image", "zh")]

    @pytest.mark.parametrize(
        "image_list,error_message",
        [
            (None, "image_urls cannot be None"),
            ("not-a-list", "image_urls must be a list of bytes"),
            ([], "image_urls must contain at least one image"),
        ],
    )
    def test_forward_impl_validates_inputs(
        self, tool, image_list, error_message
    ):
        with pytest.raises(ValueError, match=error_message):
            tool._forward_impl(image_list, "question")

    def test_forward_impl_wraps_model_errors(
        self, tool, mock_vlm_model, mock_prompt_loader
    ):
        mock_vlm_model.analyze_image.side_effect = Exception("model failed")

        with pytest.raises(
            Exception,
            match="Error analyzing image: Error understanding image 1: model failed",
        ):
            tool._forward_impl([b"img"], "question")

        mock_vlm_model.analyze_image.assert_called_once()

