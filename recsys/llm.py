"""LLM, общая для саммари (шаг 1) и описания (шаг 5).

stub           -> StubLLM, ничего не загружает (саммари и описание падают на правила/шаблон)
local          -> LocalLLM, открытая модель через transformers на GPU DataSphere
gemma_service  -> GemmaServiceLLM, готовый сервис в DataSphere: функция prompt -> ответ из модуля проекта
                  (по умолчанию gemma_service.measure_request из /home/jupyter/project/llm_gemma_service)
"""
from __future__ import annotations

import importlib
import json
import re
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

ChatMessages = List[Dict[str, str]]


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, messages: Union[str, ChatMessages], max_new_tokens: Optional[int] = None) -> str:
        ...


class StubLLM(BaseLLM):
    """Заглушка: возвращает фиксированный ответ (по умолчанию пустой)."""

    def __init__(self, response: str = ""):
        self.response = response
        self.calls: List[Any] = []

    def generate(self, messages, max_new_tokens=None) -> str:
        self.calls.append(messages)
        return self.response


class LocalLLM(BaseLLM):
    """HF-модель в процессе ноутбука. Загружается лениво при первом вызове.

    Для DataSphere: конфигурация с GPU, например Qwen/Qwen2.5-7B-Instruct (bf16 ~16 ГБ)
    или Qwen/Qwen2.5-1.5B-Instruct для быстрых проверок.
    """

    def __init__(self, model_name: str, max_new_tokens: int = 512, temperature: float = 0.0,
                 device_map: str = "auto", torch_dtype: str = "auto", load_in_4bit: bool = False):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tok = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: Dict[str, Any] = {"device_map": self.device_map, "torch_dtype": self.torch_dtype}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self._model.eval()

    def generate(self, messages, max_new_tokens=None) -> str:
        self._load()
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        ids = self._tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        ids = ids.to(self._model.device)
        gen: Dict[str, Any] = {"max_new_tokens": max_new_tokens or self.max_new_tokens,
                               "pad_token_id": self._tok.eos_token_id}
        if self.temperature > 0:
            gen.update(do_sample=True, temperature=self.temperature)
        else:
            gen.update(do_sample=False)
        out = self._model.generate(ids, **gen)
        return self._tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


class GemmaServiceLLM(BaseLLM):
    """LLM-сервис DataSphere: function(prompt) из модуля module в папке path. Модуль импортируется при первом
    вызове, поэтому локально (модуля нет) пайплайн работает: ошибка импорта -> шаг падает на шаблон/правила."""

    TEXT_KEYS = ("text", "response", "answer", "output", "content", "generated_text", "result")

    def __init__(self, path: str, module: str = "gemma_service", function: str = "measure_request"):
        self.path, self.module, self.function = path, module, function
        self._fn = None

    def _load(self):
        if self._fn is None:
            if self.path and self.path not in sys.path:
                sys.path.insert(0, self.path)
            self._fn = getattr(importlib.import_module(self.module), self.function)
        return self._fn

    @classmethod
    def _text(cls, result: Any) -> str:
        """Ответ сервиса -> текст: строка как есть; из словаря — поле text / response / answer / ...;
        из кортежа или списка — первая строка (например, (текст, время))."""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            for key in cls.TEXT_KEYS:
                if isinstance(result.get(key), str):
                    return result[key]
        if isinstance(result, (list, tuple)):
            for item in result:
                text = cls._text(item)
                if text:
                    return text
        return ""

    def generate(self, messages, max_new_tokens=None) -> str:
        prompt = messages if isinstance(messages, str) else "\n\n".join(m["content"] for m in messages)
        return self._text(self._load()(prompt)).strip()


def build_llm(cfg: Dict[str, Any]) -> BaseLLM:
    lcfg = cfg.get("llm", {})
    kind = lcfg.get("type", "stub")
    if kind == "stub":
        return StubLLM(lcfg.get("stub_response", ""))
    if kind == "local":
        return LocalLLM(
            model_name=lcfg["model_name"],
            max_new_tokens=lcfg.get("max_new_tokens", 512),
            temperature=lcfg.get("temperature", 0.0),
            device_map=lcfg.get("device_map", "auto"),
            torch_dtype=lcfg.get("torch_dtype", "auto"),
            load_in_4bit=lcfg.get("load_in_4bit", False),
        )
    if kind == "gemma_service":
        g = lcfg.get("gemma_service", {})
        return GemmaServiceLLM(g.get("path", ""), module=g.get("module", "gemma_service"),
                               function=g.get("function", "measure_request"))
    raise ValueError(f"Неизвестный llm.type: {kind}")


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Достаёт первый JSON-объект из ответа модели (в том числе из ```json ... ```)."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
