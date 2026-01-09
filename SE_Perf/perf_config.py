from dataclasses import dataclass, field
from typing import Any, Literal

from core.global_memory.utils.config import (
    ChromaBackendConfig,
    GlobalMemoryConfig,
    MemoryConfig,
    OpenAIEmbeddingConfig,
)


@dataclass
class PerfRunCLIConfig:
    config: str = "SE/configs/se_configs/dpsk.yaml"
    mode: str = "execute"


@dataclass
class ModelConfig:
    name: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.name is not None:
            out["name"] = self.name
        if self.api_base is not None:
            out["api_base"] = self.api_base
        if self.api_key is not None:
            out["api_key"] = self.api_key
        if self.max_input_tokens is not None:
            out["max_input_tokens"] = self.max_input_tokens
        if self.max_output_tokens is not None:
            out["max_output_tokens"] = self.max_output_tokens
        if self.temperature is not None:
            out["temperature"] = self.temperature
        out.update(self.extras)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ModelConfig":
        known_keys = {"name", "api_base", "api_key", "max_input_tokens", "max_output_tokens", "temperature"}
        extras = {k: v for k, v in (d or {}).items() if k not in known_keys}
        return ModelConfig(
            name=(d or {}).get("name"),
            api_base=(d or {}).get("api_base"),
            api_key=(d or {}).get("api_key"),
            max_input_tokens=(d or {}).get("max_input_tokens"),
            max_output_tokens=(d or {}).get("max_output_tokens"),
            temperature=(d or {}).get("temperature"),
            extras=extras,
        )


@dataclass
class InstancesConfig:
    instances_dir: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {"instances_dir": self.instances_dir}
        out.update(self.extras)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "InstancesConfig":
        extras = {k: v for k, v in (d or {}).items() if k != "instances_dir"}
        return InstancesConfig(instances_dir=str((d or {}).get("instances_dir") or ""), extras=extras)


@dataclass
class LocalMemoryConfig:
    enabled: bool = True
    format_mode: Literal["full", "short"] = "short"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {"enabled": self.enabled, "format_mode": self.format_mode}
        out.update(self.extras)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "LocalMemoryConfig":
        enabled_val = (d or {}).get("enabled")
        enabled = True if enabled_val is None else bool(enabled_val)
        fmt = str((d or {}).get("format_mode") or "short")
        extras = {k: v for k, v in (d or {}).items() if k not in {"enabled", "format_mode"}}
        return LocalMemoryConfig(enabled=enabled, format_mode=fmt, extras=extras)


@dataclass
class StrategyConfig:
    iterations: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {"iterations": self.iterations}
        out.update(self.extras)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "StrategyConfig":
        iterations = list((d or {}).get("iterations") or [])
        extras = {k: v for k, v in (d or {}).items() if k != "iterations"}
        return StrategyConfig(iterations=iterations, extras=extras)


@dataclass
class SEPerfRunSEConfig:
    base_config: str | None = None
    output_dir: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    instances: InstancesConfig = field(default_factory=InstancesConfig)
    max_iterations: int = 1
    num_workers: int = 20
    local_memory: LocalMemoryConfig | None = None
    prompt_config: dict[str, Any] = field(default_factory=dict)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    global_memory_bank: GlobalMemoryConfig | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "base_config": self.base_config,
            "output_dir": self.output_dir,
            "model": self.model.to_dict(),
            "instances": self.instances.to_dict(),
            "max_iterations": self.max_iterations,
            "num_workers": self.num_workers,
            "prompt_config": self.prompt_config,
            "strategy": self.strategy.to_dict(),
        }
        if self.local_memory is not None:
            out["local_memory"] = self.local_memory.to_dict()
        if self.global_memory_bank is not None:
            out["global_memory_bank"] = {
                "enabled": bool(self.global_memory_bank.enabled),
                "embedding_model": {
                    "provider": self.global_memory_bank.embedding_model.provider,
                    "api_base": self.global_memory_bank.embedding_model.api_base,
                    "api_key": self.global_memory_bank.embedding_model.api_key,
                    "model": self.global_memory_bank.embedding_model.model,
                    "request_timeout": self.global_memory_bank.embedding_model.request_timeout,
                },
                "memory": {
                    "backend": self.global_memory_bank.memory.backend,
                    "chroma": {
                        "collection_name": self.global_memory_bank.memory.chroma.collection_name,
                        "persist_path": self.global_memory_bank.memory.chroma.persist_path,
                    },
                },
            }
        out.update(self.extras)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SEPerfRunSEConfig":
        base_config = (d or {}).get("base_config")
        output_dir = str((d or {}).get("output_dir") or "")
        model = ModelConfig.from_dict((d or {}).get("model") or {})
        instances = InstancesConfig.from_dict((d or {}).get("instances") or {})
        mi_val = (d or {}).get("max_iterations", 10)
        try:
            max_iterations = int(mi_val)
        except Exception:
            max_iterations = 10
        nw_val = (d or {}).get("num_workers", 1)
        try:
            num_workers = int(nw_val)
        except Exception:
            num_workers = 1
        lm_dict = (d or {}).get("local_memory")
        local_memory = LocalMemoryConfig.from_dict(lm_dict) if isinstance(lm_dict, dict) else None
        prompt_config = (d or {}).get("prompt_config") or {}
        strategy = StrategyConfig.from_dict((d or {}).get("strategy") or {})
        known = {
            "base_config",
            "output_dir",
            "model",
            "instances",
            "max_iterations",
            "num_workers",
            "local_memory",
            "prompt_config",
            "strategy",
            "global_memory_bank",
        }
        extras = {k: v for k, v in (d or {}).items() if k not in known}
        gmb_dict = (d or {}).get("global_memory_bank") or None
        gmb = None
        if isinstance(gmb_dict, dict):
            enabled_val = gmb_dict.get("enabled")
            enabled = True if enabled_val is None else bool(enabled_val)
            em_raw = gmb_dict.get("embedding_model") or {}
            em_cfg = OpenAIEmbeddingConfig(
                provider=str(em_raw.get("provider") or "openai"),
                api_base=em_raw.get("api_base") or em_raw.get("base_url"),
                api_key=em_raw.get("api_key"),
                model=em_raw.get("model"),
                request_timeout=em_raw.get("request_timeout"),
            )
            m_raw = gmb_dict.get("memory") or {}
            c_raw = m_raw.get("chroma") or {}
            chroma_cfg = ChromaBackendConfig(
                collection_name=str(c_raw.get("collection_name") or "global_memory"),
                persist_path=c_raw.get("persist_path"),
            )
            mem_cfg = MemoryConfig(backend=str(m_raw.get("backend") or "chroma"), chroma=chroma_cfg)
            gmb = GlobalMemoryConfig(enabled=enabled, embedding_model=em_cfg, memory=mem_cfg)
        return SEPerfRunSEConfig(
            base_config=base_config,
            output_dir=output_dir,
            model=model,
            instances=instances,
            max_iterations=max_iterations,
            num_workers=num_workers,
            local_memory=local_memory,
            prompt_config=prompt_config,
            strategy=strategy,
            extras=extras,
            global_memory_bank=gmb,
        )
