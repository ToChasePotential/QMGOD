
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class GlobalEntity:
    gid: str
    name: str
    layer: int
    sources: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GlobalRelation:
    gid: str
    head: str
    tail: str
    relation: str
    description: str = ""
    layer: int = 0
    sources: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GlobalUnionGraph:
    E: Dict[str, GlobalEntity] = field(default_factory=dict)
    R: Dict[str, GlobalRelation] = field(default_factory=dict)
    evidence_map: Dict[str, Any] = field(default_factory=dict)
    maps: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TokenPlan:
    total_budget: int
    per_layer_budget: Dict[int, int] = field(default_factory=dict)

    per_layer_entity_budget: Dict[int, int] = field(default_factory=dict)
    per_layer_relation_budget: Dict[int, int] = field(default_factory=dict)
    per_entity_budget: Dict[str, int] = field(default_factory=dict)
    per_relation_budget: Dict[str, int] = field(default_factory=dict)

    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PromptPack:
    prompt_raw: str
    prompt_revised: str
    token_plan: TokenPlan
    meta: Dict[str, Any] = field(default_factory=dict)
