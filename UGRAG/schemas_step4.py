
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class MatchE:
    q_eid: str
    d_eid: str
    s_E: float
    evidence: str = ""

@dataclass
class MatchR:
    q_rid: str
    d_rid: str
    s_R: float
    evidence: str = ""

@dataclass
class AlignEdge:
    rid: str
    head: str   # q_eid
    tail: str   # d_eid
    relation: str = "align"
    description: str = ""
    layer: str = "align"
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UnionSubGraph:
    doc_id: str

    G_que_ell_E: List[str]
    G_que_ell_R: List[str]
    G_que_h_E: List[str]
    G_que_h_R: List[str]

    d_E_ids: List[str]
    d_R_ids: List[str]


    kept_d_E_ids: List[str] = field(default_factory=list) 
    kept_d_R_ids: List[str] = field(default_factory=list)   #
    bridge_E_ids: List[str] = field(default_factory=list)   
    bridge_R_ids: List[str] = field(default_factory=list)   

    M_i_E: List[MatchE] = field(default_factory=list)
    M_i_R: List[MatchR] = field(default_factory=list)
    R_i_align: List[AlignEdge] = field(default_factory=list)

    backtrace_map: Dict[str, Any] = field(default_factory=dict)
