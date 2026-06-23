"""Robot Video Segmentor - Configuration management."""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from pathlib import Path
import json
import os
import yaml
from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    root: str = Field(..., description="Path to data root directory")
    subset: str = Field(..., description="Subset/directory name")


class RunConfig(BaseModel):
    """Run/output configuration."""
    base_dir: str = Field(default="./runs", description="Base directory for outputs")
    run_id: str = Field(default="default", description="Run identifier")


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8099, description="Server port")
    max_queue: int = Field(default=32, description="Maximum job queue size")
    inflight_timeout_sec: float = Field(default=300.0, description="Timeout for in-flight jobs")
    max_retries_per_job: int = Field(default=5, description="Maximum retries per job")
    auto_exit_after_all_done: bool = Field(default=False, description="Auto exit when all done")


class Qwen3VLConfig(BaseModel):
    """Qwen3VL-specific configuration."""
    model_path: str = Field(
        default="Qwen/Qwen3-VL-32B-Instruct",
        description="Model path or HuggingFace model name"
    )
    device_map: str = Field(default="balanced", description="Device map strategy")


class RemoteAPIConfig(BaseModel):
    """Remote API backend configuration."""
    api_url: str = Field(default="http://127.0.0.1:8080/infer", description="Remote API URL")
    api_key: str = Field(default="", description="API key for remote API")
    timeout_sec: float = Field(default=60.0, description="Request timeout in seconds")
    headers: dict = Field(default_factory=dict, description="Extra headers for remote API")


class OpenAIChatConfig(BaseModel):
    """OpenAI-compatible chat/vision backend configuration."""
    base_url: str = Field(
        default="https://aihubmix.com/v1",
        description="Base URL ending with /v1",
    )
    api_key: str = Field(default="", description="Bearer API key")
    model: str = Field(default="qwen3.5-397b-a17b", description="Model id")
    timeout_sec: float = Field(default=120.0, description="Request timeout in seconds")
    max_tokens: int = Field(
        default=2048,
        description="Output token budget: max_tokens (default) or MiMo maps to max_completion_tokens",
    )
    headers: dict = Field(default_factory=dict, description="Extra headers")


class WorkerConfig(BaseModel):
    """Worker configuration."""
    server_url: str = Field(default="http://127.0.0.1:8099", description="Server URL")
    backend: str = Field(default="dummy", description="VLM backend type")
    qwen3vl: Qwen3VLConfig = Field(default_factory=Qwen3VLConfig)
    remote_api: RemoteAPIConfig = Field(default_factory=RemoteAPIConfig)
    openai_chat: OpenAIChatConfig = Field(default_factory=OpenAIChatConfig)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = ["dummy", "qwen3vl", "remote_api", "openai_chat"]
        if v not in allowed:
            raise ValueError(f"backend must be one of {allowed}, got {v}")
        return v


class WindowingConfig(BaseModel):
    """Video windowing configuration."""
    window_sec: float = Field(default=16.0, description="Window duration in seconds")
    step_sec: float = Field(default=8.0, description="Step size in seconds")
    frames_per_window: int = Field(default=16, description="Frames per window")
    target_width: int = Field(default=720, description="Target frame width")
    target_height: int = Field(default=480, description="Target frame height")
    png_compression: int = Field(default=0, description="PNG compression level (0-9)")


class ProgressConfig(BaseModel):
    """Progress tracking configuration."""
    total_override: int = Field(default=0, description="Override total count (0=auto)")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Log level")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"level must be one of {allowed}, got {v}")
        return v_upper


class Config(BaseModel):
    """Main application configuration."""
    datasets: List[DatasetConfig] = Field(default_factory=list)
    run: RunConfig = Field(default_factory=RunConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    windowing: WindowingConfig = Field(default_factory=WindowingConfig)
    progress: ProgressConfig = Field(default_factory=ProgressConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with env vars if present
        if "DATASETS" in os.environ:
            config.datasets = _parse_datasets_env(os.environ["DATASETS"])
        if "RUN_BASE" in os.environ:
            config.run.base_dir = os.environ["RUN_BASE"]
        if "RUN_ID" in os.environ:
            config.run.run_id = os.environ["RUN_ID"]
        if "PORT" in os.environ:
            config.server.port = int(os.environ["PORT"])
        if "SERVER_URL" in os.environ:
            config.worker.server_url = os.environ["SERVER_URL"]
        if "MODEL_PATH" in os.environ:
            config.worker.qwen3vl.model_path = os.environ["MODEL_PATH"]
        if "BACKEND" in os.environ:
            config.worker.backend = os.environ["BACKEND"]
        if "REMOTE_API_URL" in os.environ:
            config.worker.remote_api.api_url = os.environ["REMOTE_API_URL"]
        if "REMOTE_API_KEY" in os.environ:
            config.worker.remote_api.api_key = os.environ["REMOTE_API_KEY"]
        if "REMOTE_API_TIMEOUT" in os.environ:
            config.worker.remote_api.timeout_sec = float(os.environ["REMOTE_API_TIMEOUT"])
        if "REMOTE_API_HEADERS" in os.environ:
            headers_raw = os.environ["REMOTE_API_HEADERS"]
            headers = json.loads(headers_raw)
            if not isinstance(headers, dict):
                raise ValueError("REMOTE_API_HEADERS must be a JSON object")
            config.worker.remote_api.headers = headers
        if "OPENAI_CHAT_BASE_URL" in os.environ:
            config.worker.openai_chat.base_url = os.environ["OPENAI_CHAT_BASE_URL"]
        if "OPENAI_CHAT_API_KEY" in os.environ:
            config.worker.openai_chat.api_key = os.environ["OPENAI_CHAT_API_KEY"]
        if "OPENAI_CHAT_MODEL" in os.environ:
            config.worker.openai_chat.model = os.environ["OPENAI_CHAT_MODEL"]
        if "OPENAI_CHAT_TIMEOUT" in os.environ:
            config.worker.openai_chat.timeout_sec = float(os.environ["OPENAI_CHAT_TIMEOUT"])
        if "OPENAI_CHAT_MAX_TOKENS" in os.environ:
            config.worker.openai_chat.max_tokens = int(os.environ["OPENAI_CHAT_MAX_TOKENS"])
        if "OPENAI_CHAT_HEADERS" in os.environ:
            headers_raw = os.environ["OPENAI_CHAT_HEADERS"]
            headers = json.loads(headers_raw)
            if not isinstance(headers, dict):
                raise ValueError("OPENAI_CHAT_HEADERS must be a JSON object")
            config.worker.openai_chat.headers = headers
        
        return config
    
    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "Config":
        """Load configuration with priority: file > env > defaults."""
        if path:
            return cls.from_yaml(path)
        
        # Try to find config.yaml in current directory
        default_path = Path("config.yaml")
        if default_path.exists():
            return cls.from_yaml(default_path)
        
        # Fall back to environment variables
        return cls.from_env()


def _parse_datasets_env(spec: str) -> List[DatasetConfig]:
    """Parse DATASETS environment variable."""
    configs = []
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    for p in parts:
        if ":" in p:
            root, subset = p.split(":", 1)
            configs.append(DatasetConfig(root=root.strip(), subset=subset.strip()))
        else:
            data_dir = Path(p.rstrip("/"))
            root = str(data_dir.parent)
            subset = data_dir.name
            configs.append(DatasetConfig(root=root, subset=subset))
    return configs
