from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class EntityRow:
    eid: str
    name: str
    entity_type: str = ""   
    description: str = ""
    layer: str = ""         
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RelationRow:
    rid: str
    head: str
    tail: str
    relation: str
    description: str = ""
    layer: str = ""     
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EntitiesTable:
    rows: Dict[str, EntityRow] = field(default_factory=dict)

    def add(self, row: EntityRow) -> None:
        self.rows[row.eid] = row

@dataclass
class RelationsTable:
    rows: Dict[str, RelationRow] = field(default_factory=dict)

    def add(self, row: RelationRow) -> None:
        self.rows[row.rid] = row

@dataclass
class QuestionSubGraph:
    layer: str  
    E: List[str] = field(default_factory=list)
    R: List[str] = field(default_factory=list)
