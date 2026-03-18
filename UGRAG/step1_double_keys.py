from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any
import json
import re


def _try_lightrag_keywords(text: str, topk: int = 8) -> Optional[List[str]]:

    try:
        import importlib
        candidates = [
            ("lightrag", ["extract_keywords", "get_keywords", "keywords"]),
            ("lightrag.utils", ["extract_keywords", "get_keywords", "keywords"]),
            ("lightrag.utils.text", ["extract_keywords", "get_keywords", "keywords"]),
            ("lightrag.text_utils", ["extract_keywords", "get_keywords", "keywords"]),
        ]
        for mod_name, fn_names in candidates:
            try:
                m = importlib.import_module(mod_name)
            except Exception:
                continue
            for fn_name in fn_names:
                if hasattr(m, fn_name):
                    fn = getattr(m, fn_name)
                    try:
                        out = fn(text, topk=topk)
                    except Exception:
                        out = fn(text)
                    if isinstance(out, (list, tuple)):
                        return list(out)[:topk]
        return None
    except Exception:
        return None



@dataclass
class KeyRelation:
    head: str
    tail: str
    relation: str           
    description: str = ""
    layer: str = ""           
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DoubleKeys:
    low_level_keys: List[str] = field(default_factory=list)
    high_level_keys: List[str] = field(default_factory=list)
    keys_relations: List[KeyRelation] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


def step1_double_keys(
    x_ins: str,
    x_que: str,
    llm_generate: Optional[Callable[[str], str]] = None,
    use_lightrag: bool = False,
    lightrag_topk: int = 8,
) -> DoubleKeys:

    if llm_generate is not None:
        prompt = _build_double_keys_prompt(x_ins=x_ins, x_que=x_que)
        raw = llm_generate(prompt)
        dk = _parse_double_keys_output(raw)
        dk.meta["mode"] = "llm"
        dk.meta["prompt"] = prompt
        dk.meta["raw"] = raw
        return dk

    if use_lightrag:
        kws = _try_lightrag_keywords(x_que, topk=lightrag_topk)
        if kws:
            low = kws[: max(1, min(6, len(kws)))]
            high = list(dict.fromkeys([k.split()[-1] for k in kws[: max(1, min(4, len(kws)))] ]))
            dk = DoubleKeys(low_level_keys=low, high_level_keys=high, keys_relations=[])
            dk.meta["mode"] = "lightrag"
            return dk

    dk = _heuristic_double_keys(x_que=x_que)
    dk.meta["mode"] = "heuristic"
    return dk


def _build_double_keys_prompt(x_ins: str, x_que: str) -> str:
    return (
        "You are implementing UnionGraph-RAG.\n"
        "Given system instruction x^{ins} and user query x_{que}, generate Double-keys and keys structure relations.\n\n"
        "Return STRICT JSON ONLY with the following schema:\n"
        "{\n"
        '  "low_level_keys": ["..."],\n'
        '  "high_level_keys": ["..."],\n'
        '  "keys_relations": [\n'
        '    {"head":"...","tail":"...","relation":"...","description":"...","layer":"ell|h|cross"}\n'
        "  ]\n"
        "}\n\n"
        "Guidelines:\n"
        "- low_level_keys: concrete entities, constraints, numbers, fine-grained details.\n"
        "- high_level_keys: topics/intents/structure that guides which category to search.\n"
        "- keys_relations: explicit structure among keys (within low/high layers or cross-layer).\n"
        "- Output JSON only. No extra text.\n\n"
        f"x^{ins}: {x_ins}\n"
        f"x_{que}: {x_que}\n"
    )


def _parse_double_keys_output(raw: str) -> DoubleKeys:
    raw_s = raw.strip()

    try:
        obj = json.loads(raw_s)
        low = [str(x).strip() for x in obj.get("low_level_keys", []) if str(x).strip()]
        high = [str(x).strip() for x in obj.get("high_level_keys", []) if str(x).strip()]

        rels: List[KeyRelation] = []
        for r in (obj.get("keys_relations", []) or []):
            head = str(r.get("head", "")).strip()
            tail = str(r.get("tail", "")).strip()
            relation = str(r.get("relation", "")).strip()
            description = str(r.get("description", "")).strip()
            layer = str(r.get("layer", "")).strip()
            if head and tail and relation:
                rels.append(KeyRelation(head=head, tail=tail, relation=relation, description=description, layer=layer))

        return DoubleKeys(low_level_keys=low, high_level_keys=high, keys_relations=rels)
    except Exception:
        pass

    low = _extract_bracket_list(raw_s, label="low-level keys")
    high = _extract_bracket_list(raw_s, label="high-level keys")
    return DoubleKeys(low_level_keys=low, high_level_keys=high, keys_relations=[])


def _extract_bracket_list(raw: str, label: str) -> List[str]:
    pattern = re.compile(rf"{re.escape(label)}\s*:\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)
    m = pattern.search(raw)
    if not m:
        return []
    inside = m.group(1)
    items = [re.sub(r"^['\"\s]+|['\"\s]+$", "", x.strip()) for x in inside.split(",")]
    return [x for x in items if x]


def _heuristic_double_keys(x_que: str) -> DoubleKeys:

    low: List[str] = []
    high: List[str] = []

    low.extend(re.findall(r"\b\d+(?:\.\d+)?\b", x_que))


    toks = [t for t in re.split(r"[^A-Za-z0-9_\u4e00-\u9fff]+", x_que) if t]
    seen = set()
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        if len(t) <= 1:
            continue
        if len(t) >= 4:
            low.append(t)
        else:
            high.append(t)

    low = _dedup_keep_order(low)
    high = _dedup_keep_order(high)

    rels: List[KeyRelation] = []
    if high:
        for lk in low[: min(len(low), 8)]:
            rels.append(KeyRelation(head=lk, tail=high[0], relation="belongs_to", description="heuristic", layer="cross"))

    return DoubleKeys(low_level_keys=low, high_level_keys=high, keys_relations=rels)


def _dedup_keep_order(xs: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in xs:
        x = str(x).strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out