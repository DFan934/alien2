# ---------------------------
# FILE: prediction_engine/schemas.py
# ---------------------------
"""Pydantic schema – runtime parameters & IO directories."""
from pathlib import Path
from typing import Literal
from typing import Any

from pydantic import BaseModel, Field, validator


class EngineConfig(BaseModel):
    # model hyper‑params
    knn_k: int = Field(15, ge=1)
    ann_backend: Literal["sklearn", "faiss"] = "faiss"
    ann_backend_kwargs: dict[str, Any] = Field(default_factory=dict)
    min_sharpe: float = 0.30  # entry filter
    drift_threshold: float = 0.12

    # IO
    weight_dir: Path = Path("weights/")
    logs_dir: Path = Path("logs/")
    signal_dir: Path = Path("signals/")

    class Config:
        extra = "forbid"

    @validator("weight_dir", "logs_dir", "signal_dir", pre=True)
    def _expand(cls, v):  # noqa: D401
        p = Path(v).expanduser().absolute()
        p.mkdir(parents=True, exist_ok=True)
        return p

