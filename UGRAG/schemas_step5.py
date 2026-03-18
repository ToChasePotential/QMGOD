
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ScoreItem:
    doc_id: str
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DiffSeqResult:
    ranked: List[ScoreItem]
    diffs: List[float]
    rel_diffs: List[float]
    k_prime: int
    kept: List[ScoreItem]
