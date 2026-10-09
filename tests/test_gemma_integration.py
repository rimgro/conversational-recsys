"""Bundled Gemma protocol and pipeline integration without downloading weights."""
import importlib
from unittest.mock import Mock

from recsys.config import deep_update, load_config
from recsys.llm import build_llm
from recsys.pipeline import Pipeline


def test_pipeline_reconnects_without_reloading_weights(monkeypatch, cfg, data):
    service = importlib.import_module("llm_gemma_service.gemma_service")
    requests = []

    def api(path, payload=None, timeout=10):
        if path == "/health":
            return {"status": "ok"}
        if path == "/v1/models":
            return {"data": [{"id": "gemma"}]}
        assert path == "/v1/chat/completions"
        requests.append((payload, timeout))
        return {"choices": [{"message": {"content": "Enjoy these recommendations."}}],
                "usage": {"completion_tokens": 4}}

    monkeypatch.setattr(service, "api", api)
    monkeypatch.setattr(service, "_read_state", lambda: None)
    launch = Mock(side_effect=AssertionError("A healthy server must not be relaunched"))
    monkeypatch.setattr(service, "_launch", launch)
    layer = load_config("configs/gemma.yaml")
    settings = deep_update(cfg, layer)
    settings = deep_update(settings, {"llm": {"temperature": 0.3,
                                             "gemma_service": {"timeout": 120}}})
    # Recreate the whole pipeline twice, as after editing/reloading its code.
    for _ in range(2):
        pipe = Pipeline.from_config(settings, data.catalog)
        result = pipe.run(data.requests[0])
        assert result.text == "Enjoy these recommendations."
    launch.assert_not_called()
    assert len(requests) == 2
    for payload, timeout in requests:
        assert [m["role"] for m in payload["messages"]] == ["system", "user"]
        assert "Recommended tracks:" in payload["messages"][1]["content"]
        assert payload["max_tokens"] == settings["explainer"]["max_new_tokens"]
        assert payload["temperature"] == 0.3
        assert payload["stream"] is False
        assert timeout == 120


def test_bundled_adapter_defaults_and_plain_prompt(monkeypatch):
    service = importlib.import_module("llm_gemma_service.gemma_service")
    fn = Mock(return_value={"text": " answer "})
    monkeypatch.setattr(service, "measure_chat", fn)
    llm = build_llm({"llm": {"type": "gemma_service", "max_new_tokens": 77}})
    assert llm.generate("question") == "answer"
    fn.assert_called_once_with([{"role": "user", "content": "question"}],
                               max_tokens=77, temperature=0.0, timeout=300)
    llm.generate("question", max_new_tokens=12)
    assert fn.call_args.kwargs["max_tokens"] == 12
