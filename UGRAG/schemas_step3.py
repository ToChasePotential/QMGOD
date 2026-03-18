
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class SourceChunk:
 
    chunk_id: int
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocEntityRow:

    eid: str
    name: str
    description: str = ""
    description1: str = "" 
    source_chunk_ids: List[int] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocRelationRow:
    rid: str
    head: str
    tail: str
    relation: str
    description: str = ""
    description1: str = ""
    source_chunk_ids: List[int] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
