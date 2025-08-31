# --- scanner/config.py (new)
"""
Centralised loader for scanner‑phase YAML config.

The YAML schema is intentionally flat:

```yaml
gap:
  pct: 0.02
rvol:
  thresh: 2.0
  lookback: 20
news:
  min_score: 0.0         # −1 … +1   (skip if < min_score)
liquidity:
  adv_cap_pct: 20        # KellySizer parameter
```
"""
from __future__ import annotations

import yaml, importlib.resources as pkg
from pathlib import Path
from typing import Any, Dict

_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "gap":      {"pct": 0.02},
    "rvol":     {"thresh": 2.0, "lookback": 20},
    "liquidity":{"adv_cap_pct": 20.0},
    "composite":{"mode": "AND"},
}



def load(path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return _DEFAULTS
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    with cfg_path.open("r", encoding="utf‑8") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    # shallow‑merge defaults
    #out: Dict[str, Dict[str, Any]] = {k: {**v, **user_cfg.get(k, {})}
    #                                  for k, v in _DEFAULTS.items()}

    # shallow-merge defaults including 'composite.mode'
    out: Dict[str, Dict[str, Any]] = {
        k: {**v, **user_cfg.get(k, {})} for k, v in _DEFAULTS.items()
                                                             }

    return out



def load_cfg(path: str | None = None) -> dict:
    if path is None:
        # default asset inside package -> config/scanner.yaml
        with pkg.files("config").joinpath("scanner.yaml").open() as fh:
            return yaml.safe_load(fh)
    with open(path, "r") as fh:
        return yaml.safe_load(fh)
