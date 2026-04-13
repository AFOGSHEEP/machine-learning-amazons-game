from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import json
from typing import Any

from src.envs.amazons_env import Action, MiniAmazonsEnv


def prune_actions_by_score(
    env: MiniAmazonsEnv,
    player: int,
    legal_actions: list[Action],
    top_k: int = 0,
    keep_ratio: float = 1.0,
) -> list[Action]:
    """
    Keep only top scored actions to reduce branching.
    top_k<=0 and keep_ratio>=1 means no pruning.
    """
    if len(legal_actions) <= 1:
        return legal_actions
    if top_k <= 0 and keep_ratio >= 1.0:
        return legal_actions

    scored = [(env.estimate_action_score(a, player), a) for a in legal_actions]
    scored.sort(key=lambda x: x[0], reverse=True)

    ratio_n = int(len(legal_actions) * keep_ratio)
    if ratio_n <= 0:
        ratio_n = 1
    if top_k > 0:
        keep_n = max(1, min(top_k, ratio_n))
    else:
        keep_n = max(1, ratio_n)
    return [a for _, a in scored[:keep_n]]


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    return {"repr": repr(cfg)}


def save_run_metadata(
    out_path: str,
    run_type: str,
    config: Any,
    extra: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "run_type": run_type,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": _to_plain_dict(config),
    }
    if extra:
        payload["extra"] = extra
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    run_id = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
    payload["run_id"] = run_id

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_id


def save_case_record(
    out_path: str,
    metadata: dict[str, Any],
    steps: list[dict[str, Any]],
) -> str:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metadata": metadata,
        "steps": steps,
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    case_id = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
    payload["case_id"] = case_id

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return case_id
