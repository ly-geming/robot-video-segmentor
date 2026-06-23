from typing import Any, Dict, Type

from .base import VLMBackend
from .dummy import DummyBackend
from .openai_chat import OpenAIChatBackend
from .remote_api import RemoteAPIBackend


BACKENDS: Dict[str, Type[VLMBackend]] = {
    "dummy": DummyBackend,
    "remote_api": RemoteAPIBackend,
    "openai_chat": OpenAIChatBackend,
}


def create_backend(backend_type: str, **kwargs: Any) -> VLMBackend:
    if backend_type == "qwen3vl":
        from .qwen3vl import Qwen3VLBackend

        return Qwen3VLBackend(**kwargs)

    backend_class = BACKENDS.get(backend_type)
    if backend_class is None:
        available = sorted(list(BACKENDS.keys()) + ["qwen3vl"])
        raise ValueError(f"Unknown backend: {backend_type}. Available: {available}")

    return backend_class(**kwargs)
