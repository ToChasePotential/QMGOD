
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import math
import re

from .schemas_step4 import UnionSubGraph
from .schemas_step5 import DiffSeqResult, ScoreItem
from .schemas_step6 import GlobalUnionGraph, GlobalEntity, GlobalRelation, TokenPlan, PromptPack


def _try_llmlingua_compress(text: str, rate: float = 0.5) -> Optional[str]:
    """
    If microsoft/LLMLingua is installed, compress the prompt to reduce cost.
    Returns None if unavailable.
    """
    try:
        from llmlingua import PromptCompressor

        compressor = PromptCompressor()
        try:
            out = compressor.compress_prompt(text, rate=rate)
        except TypeError:
            out = compressor.compress_prompt(text)
        if isinstance(out, dict):
            return out.get("compressed_prompt") or out.get("compressed") or out.get("prompt")
        if isinstance(out, str):
            return out
    except Exception:
        return None
    return None

def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _layer_from_strength(s: float) -> int:

    if s >= 0.85:
        return 0
    if s >= 0.65:
        return 1
    return 2

def _allocate_token_budget(total_budget: int, max_layer: int) -> Dict[int, int]:

    if total_budget <= 0:
        return {0: 0}

    weights = []
    for L in range(max_layer + 1):

        weights.append(0.55 * (0.6 ** L))
    s = sum(weights)
    weights = [w / s for w in weights]

    per_layer = {}
    remain = total_budget
    for L, w in enumerate(weights):
        b = int(round(total_budget * w))
        per_layer[L] = b
        remain -= b

    if remain != 0:
        per_layer[0] = max(0, per_layer.get(0, 0) + remain)
    return per_layer

def step6_global_union_dedup_budget_self_think_generate(
    union_subgraphs: List[UnionSubGraph],
    res5: DiffSeqResult,
    total_budget: int = 1200,
    llm_generate=None,
) -> PromptPack:
 

    kept_doc_ids = [it.doc_id for it in res5.kept]
    kept_union = [u for u in union_subgraphs if u.doc_id in kept_doc_ids]

    G_global = build_global_union_graph(kept_union)

    token_plan = build_token_plan(G_global, total_budget=total_budget)

    prompt_raw = build_prompt_from_global_union_graph(G_global, token_plan)

    if use_llmlingua:
        c = _try_llmlingua_compress(prompt_raw, rate=llmlingua_rate)
        if c:
            prompt_raw = c

    prompt_revised = self_think_revision(prompt_raw, G_global)

    if use_llmlingua:
        c2 = _try_llmlingua_compress(prompt_revised, rate=llmlingua_rate)
        if c2:
            prompt_revised = c2

    return PromptPack(
        prompt_raw=prompt_raw,
        prompt_revised=prompt_revised,
        token_plan=token_plan,
        meta={"kept_doc_ids": kept_doc_ids, "global_stats": {"|E|": len(G_global.E), "|R|": len(G_global.R)}},
    )

def build_global_union_graph(kept_union: List[UnionSubGraph]) -> GlobalUnionGraph:


    G = GlobalUnionGraph()

    for u in kept_union:
        for k, v in (u.backtrace_map or {}).items():

            if k not in G.evidence_map:
                G.evidence_map[k] = v


    maps = {}


    name2gid: Dict[str, str] = {}

    gid_counter = 0
    rid_counter = 0

    for u in kept_union:
        doc_id = u.doc_id
        maps[doc_id] = {"entity": {}, "relation": {}, "align": []}

        for m in (u.M_i_E or []):

            for local_eid, strength in [(m.q_eid, m.s_E), (m.d_eid, m.s_E)]:
                nm = _norm_name(local_eid)
                if nm not in name2gid:
                    gid_counter += 1
                    gid = f"gE{gid_counter}"
                    name2gid[nm] = gid
                    G.E[gid] = GlobalEntity(
                        gid=gid,
                        name=local_eid,
                        layer=_layer_from_strength(strength),
                        sources=[{"doc_id": doc_id, "type": "entity", "local_id": local_eid, "s_E": strength}],
                        meta={"norm": nm},
                    )
                else:
                    gid = name2gid[nm]
 
                    G.E[gid].layer = min(G.E[gid].layer, _layer_from_strength(strength))
                    G.E[gid].sources.append({"doc_id": doc_id, "type": "entity", "local_id": local_eid, "s_E": strength})

                maps[doc_id]["entity"][local_eid] = gid

   
        for ae in (u.R_i_align or []):
            rid_counter += 1
            gr_id = f"gR{rid_counter}"

            h = maps[doc_id]["entity"].get(ae.head)
            t = maps[doc_id]["entity"].get(ae.tail)
            if h is None or t is None:
                continue

            G.R[gr_id] = GlobalRelation(
                gid=gr_id,
                head=h,
                tail=t,
                relation="align",
                description=ae.description,
                layer=0,
                sources=[{"doc_id": doc_id, "type": "align", "local_rid": ae.rid}],
                meta={"doc_id": doc_id},
            )
            maps[doc_id]["align"].append(gr_id)

        for mr in (u.M_i_R or []):
            rid_counter += 1
            gr_id = f"gR{rid_counter}"

            if not u.M_i_E:
                continue
            anchor = u.M_i_E[0]
            h = maps[doc_id]["entity"].get(anchor.q_eid)
            t = maps[doc_id]["entity"].get(anchor.d_eid)
            if h is None or t is None:
                continue
            layer = _layer_from_strength(mr.s_R)
            G.R[gr_id] = GlobalRelation(
                gid=gr_id,
                head=h,
                tail=t,
                relation="relation_match",
                description=mr.evidence,
                layer=layer,
                sources=[{"doc_id": doc_id, "type": "relation_match", "q_rid": mr.q_rid, "d_rid": mr.d_rid, "s_R": mr.s_R}],
                meta={"doc_id": doc_id},
            )
            maps[doc_id]["relation"][mr.q_rid] = gr_id

    G.maps = maps
    return G


def build_token_plan(G_global: GlobalUnionGraph, total_budget: int) -> TokenPlan:
    if not G_global.E:
        return TokenPlan(total_budget=total_budget, per_layer_budget={0: total_budget}, meta={"max_layer": 0})

    max_layer = max(e.layer for e in G_global.E.values())
    per_layer = _allocate_token_budget(total_budget, max_layer=max_layer)

    return TokenPlan(total_budget=total_budget, per_layer_budget=per_layer, meta={"max_layer": max_layer})

def build_prompt_from_global_union_graph(G_global, token_plan, x_que: str = "") -> str:


    E_by_layer: Dict[int, List[GlobalEntity]] = {}
    R_by_layer: Dict[int, List[GlobalRelation]] = {}

    for e in G_global.E.values():
        E_by_layer.setdefault(e.layer, []).append(e)
    for r in G_global.R.values():
        R_by_layer.setdefault(r.layer, []).append(r)

    for L in E_by_layer:
        E_by_layer[L].sort(key=lambda x: x.gid)
    for L in R_by_layer:
        R_by_layer[L].sort(key=lambda x: x.gid)

    lines = []
    lines.append("System Instruction:")
    lines.append("You must answer with evidence-backed statements. Use the Evidence Map for citations.")
    lines.append("")
    lines.append("Global Union Graph (layered):")
    lines.append(f"- #Entities: {len(G_global.E)}; #Relations: {len(G_global.R)}")
    lines.append(f"- Token Plan: {token_plan.per_layer_budget}")
    lines.append("")

    max_layer = token_plan.meta.get("max_layer", 0)
    for L in range(max_layer + 1):
        b = token_plan.per_layer_budget.get(L, 0)
        lines.append(f"[Layer {L}] (budget≈{b})")
        ents = E_by_layer.get(L, [])
        rels = R_by_layer.get(L, [])
        if not ents and not rels:
            lines.append("(empty)")
            lines.append("")
            continue

        lines.append("Entities:")
        for e in ents[: min(len(ents), 30)]:
            # show a short source pointer
            src = e.sources[0] if e.sources else {}
            lines.append(f"- {e.gid}: {e.name} | src={src}")
        lines.append("Relations:")
        for r in rels[: min(len(rels), 40)]:
            src = r.sources[0] if r.sources else {}
            lines.append(f"- {r.gid}: ({r.head})->({r.tail}) {r.relation} | src={src}")
        lines.append("")

    lines.append("Evidence Map Keys (Description1):")

    keys = list(G_global.evidence_map.keys())[:20]
    for k in keys:
        v = G_global.evidence_map[k]
        lines.append(f"- {k}: chunk_ids={v.get('chunk_ids')} preview='{v.get('chunk_preview')}'")
    lines.append("")
    lines.append("User Question:")
    lines.append(x_que if x_que else "(empty)")
    lines.append("")
    lines.append("Task:")
    lines.append("Generate a grounded answer. For each important claim, attach evidence pointers from Evidence Map (Description1) or align-edge descriptions.")
    return "\n".join(lines)


def self_think_revision(prompt_raw: str, G_global: GlobalUnionGraph) -> str:


    lines = prompt_raw.splitlines()
    out = []
    for line in lines:

        m = re.search(r"(doc\w+\|[ER]\|[^\s]+)", line)
        if m:
            key = m.group(1)
            if key not in G_global.evidence_map:
                continue
        out.append(line)

    if len(G_global.E) == 0:
        out.append("")
        out.append("[self-think] Warning: Global Union Graph is empty; retrieval/alignment may have failed. Consider lowering tau_E or enriching Recog extraction.")
    return "\n".join(out)