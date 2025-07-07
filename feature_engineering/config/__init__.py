# ===========================================================================
# feature_engineering/config/__init__.py  (NEW)
# ---------------------------------------------------------------------------
"""Pydantic settings loader for *fe_config.yaml* (optional).

Search order:
1. Explicit env‑var FE_CONFIG_PATH
2. Project‑root/fe_config.yaml
3. Package‑root/fe_config.yaml
If no file exists, we fall back to built‑in defaults.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Settings model (all fields optional with sensible defaults)
# ---------------------------------------------------------------------------
class _FEConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    # Paths
    parquet_root: Optional[Path] = None   # raw minute bars
    feature_root: Optional[Path] = None   # engineered output

    # Feature‑eng params
    pca_variance: float = 0.95
    row_nan_frac: float = 0.20
    impute_strategy: str = Field("mean", pattern="^(mean|median)$")


# ---------------------------------------------------------------------------
# Locate config file
# ---------------------------------------------------------------------------
def _candidate_paths() -> list[Path]:
    env = os.getenv("FE_CONFIG_PATH")
    if env:
        yield Path(env).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    yield project_root / "fe_config.yaml"
    yield Path(__file__).with_suffix(".yaml")  # package‑local


def _load_yaml(path: Path) -> dict:  # pragma: no cover – tiny helper
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


_raw: dict = {}
_origin: Optional[Path] = None
for p in _candidate_paths():
    if p.is_file():
        try:
            _raw = _load_yaml(p)
            _origin = p
            break
        except Exception as e:  # log & continue
            print(f"[FE‑config] Failed to load {p}: {e}")
    else:
        # Skip directories (avoids PermissionError when path=".")
        continue

if _origin:
    print(f"[FE‑config] Loaded settings from {_origin}")
else:
    print("[FE‑config] No YAML found – using defaults.")

settings = _FEConfig(**_raw)  # type: ignore[arg-type]
