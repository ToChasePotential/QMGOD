from typing import Any

def _normalize_step1_doublekeys(out: Any):
    """
    Accept Step1 output as:
      - tuple/list of len>=3
      - dict
      - object (e.g., DoubleKeys) with attributes
    Return: (low_level_keys, high_level_keys, keys_relation)
    """
    if isinstance(out, (list, tuple)):
        if len(out) >= 3:
            return out[0], out[1], out[2]
        raise ValueError(f"Step1 returned tuple/list but len={len(out)} < 3")

    if isinstance(out, dict):
        low = out.get("low_level_keys", out.get("low_keys", out.get("low", None)))
        high = out.get("high_level_keys", out.get("high_keys", out.get("high", None)))
        rel = out.get("keys_relation", out.get("relation", out.get("keys_rel", None)))
        if low is None or high is None:
            raise ValueError(f"Step1 dict missing keys: got={list(out.keys())}")
        return low, high, rel

    for low_nm in ("low_level_keys", "low_keys", "low"):
        if hasattr(out, low_nm):
            low = getattr(out, low_nm)
            break
    else:
        low = None

    for high_nm in ("high_level_keys", "high_keys", "high"):
        if hasattr(out, high_nm):
            high = getattr(out, high_nm)
            break
    else:
        high = None

    rel = None
    for rel_nm in ("keys_relation", "relation", "keys_rel", "rel"):
        if hasattr(out, rel_nm):
            rel = getattr(out, rel_nm)
            break

    if (low is None or high is None) and hasattr(out, "model_dump"):
        d = out.model_dump()
        return _normalize_step1_doublekeys(d)

    if low is None or high is None:
        raise TypeError(f"Step1 returned unsupported type: {type(out)} (cannot extract low/high keys)")
    return low, high, rel



import os
import json
import time
import inspect
import importlib
from typing import Any, Dict, List, Optional, Callable


def _now() -> float:
    return time.time()

def _ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _write_text(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _import_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

def resolve_func(module: Any, candidate_names: List[str]) -> Optional[Callable]:
    if module is None:
        return None
    for nm in candidate_names:
        fn = getattr(module, nm, None)
        if callable(fn):
            return fn
    return None

def call_with_supported_kwargs(fn: Callable, /, **kwargs):
    sig = inspect.signature(fn)
    supported = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            supported[k] = v
    return fn(**supported)


def _normalize_step3_out(doc_id: str, out: Any) -> Dict[str, Any]:

    if isinstance(out, dict):
        d = dict(out)
        d.setdefault("doc_id", doc_id)
        return d

    source_chunks_i, E_i, R_i = [], [], []
    if isinstance(out, (list, tuple)):
        if len(out) >= 3:
            source_chunks_i, E_i, R_i = out[0], out[1], out[2]
    return {
        "doc_id": doc_id,
        "source_chunks_i": _dump(source_chunks_i),
        "E_i": _dump(E_i),
        "R_i": _dump(R_i),
        # keep compatibility fields (optional)
        "Entities_doc_i": None,
        "Relations_doc_i": None,
        "Description1_i": None,
    }

def _dump(obj: Any) -> Any:
    """Best-effort dump for json serialization."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dump(x) for x in obj]
    return obj

try:
    from schemas_step2 import EntitiesTable, RelationsTable, EntityRow, RelationRow
except Exception:
    class EntitiesTable:
        def __init__(self, rows=None): self.rows = rows or []
    class RelationsTable:
        def __init__(self, rows=None): self.rows = rows or []
    class EntityRow:
        def __init__(self, eid, name, entity_type="entity", description="", meta=None):
            self.eid=eid; self.name=name; self.entity_type=etype; self.description=description; self.meta=chunk_ids or []
    class RelationRow:
        def __init__(self, rid, head, tail, relation="related_to", description="", meta=None):
            self.rid=rid; self.head=head; self.tail=tail; self.relation=relation; self.description=description; self.meta=chunk_ids or []


DEFAULT_RECOG_CACHE_DIR = "recog_cache"


def _heuristic_recog_dict(doc_id: str, x_i_doc: str, max_chunks: int = 12) -> Dict[str, Any]:

    text = (x_i_doc or "").strip()


    if not text:
        chunks = [""]
    else:
        chunks = [c.strip() for c in text.replace("\r", "\n").split("\n") if c.strip()]
        if not chunks:
            chunks = [text]
    chunks = chunks[:max_chunks]

    toks = []
    for w in text.replace("\n", " ").split():
        w = w.strip().strip(",.()[]{}:;\"'").strip()
        if not w:
            continue
        if len(w) >= 4:
            toks.append(w)
        if len(toks) >= 8:
            break
    if not toks:
        toks = ["doc", "context"]

    Entities_doc_i = EntitiesTable()
    Relations_doc_i = RelationsTable()

    eids = []
    for idx, name in enumerate(toks, start=1):
        eid = f"{doc_id}_e{idx}"
        eids.append(eid)
        Entities_doc_i.add(
            EntityRow(
                eid=eid,
                name=name,
                entity_type="entity",
                description=f"heuristic entity from {doc_id}",
                layer="doc",
                meta={"source_chunk_ids": [min(idx - 1, len(chunks) - 1)]},
            )
        )

    for idx in range(1, len(eids)):
        rid = f"{doc_id}_r{idx}"
        Relations_doc_i.add(
            RelationRow(
                rid=rid,
                head=eids[idx - 1],
                tail=eids[idx],
                relation="related_to",
                description=f"heuristic link in {doc_id}",
                layer="doc",
                meta={"source_chunk_ids": [min(idx - 1, len(chunks) - 1)]},
            )
        )

    Description1 = {}
    for eid in eids:
        Description1[eid] = {"source_chunk_ids": Entities_doc_i.rows[eid].meta.get("source_chunk_ids", []), "text": ""}

    return {
        "doc_id": doc_id,
        "source_chunks": [{"chunk_id": i, "text": c, "meta": {}} for i, c in enumerate(chunks)],
        "E_i": Entities_doc_i,
        "R_i": Relations_doc_i,
        "Description1": Description1,
    }

def run_step3_recog_cached(
    *,
    x_ins: str,
    doc_id: str,
    x_i_doc: str,
    llm_generate: Optional[Callable],
    cache_dir: str = DEFAULT_RECOG_CACHE_DIR,
    enable_cache: bool = True,
) -> Dict[str, Any]:
    cache_dir = cache_dir or DEFAULT_RECOG_CACHE_DIR
    cache_path = os.path.join(cache_dir, f"{doc_id}.json")

    if enable_cache and os.path.exists(cache_path):
        try:
            return json.loads(_read_text(cache_path))
        except Exception:
            pass  


    try:
        m_step3 = _import_module("step3_recog")
        Recog_fn = getattr(m_step3, "Recog", None) if m_step3 else None
        Step3Config = getattr(m_step3, "Step3Config", None) if m_step3 else None

        if callable(Recog_fn):
            cfg = None
            if Step3Config is not None:
                try:
                    cfg = Step3Config(cache_dir=cache_dir, enable_cache=enable_cache)
                except Exception:
                    cfg = None

            out = call_with_supported_kwargs(
                Recog_fn,
                x_ins=x_ins,
                x_i_doc=x_i_doc,
                x_doc_i=x_i_doc,  
                doc_id=doc_id,
                cfg=cfg,
                llm_generate=llm_generate,
            )
            d = _normalize_step3_out(doc_id, out)

            if enable_cache:
                _ensure_dir(cache_dir)
                _write_text(cache_path, json.dumps(d, ensure_ascii=False, indent=2))
            return d

    except Exception:

        pass

    d = _heuristic_recog_dict(doc_id, x_i_doc)
    if enable_cache:
        _ensure_dir(cache_dir)
        _write_text(cache_path, json.dumps(d, ensure_ascii=False, indent=2))
    return d



def uniongraph_rag_pipeline(
    *,
    x_ins: str,
    x_doc: List[str],
    x_que: str,
    llm_generate: Optional[Callable] = None,
    recog_cache_dir: str = DEFAULT_RECOG_CACHE_DIR,
    recog_enable_cache: bool = True,
) -> Dict[str, Any]:

    t0 = _now()

   
    m_step1 = _import_module("step1_double_keys")
    fn_step1 = resolve_func(m_step1, ["step1_double_keys", "double_keys", "run_step1"])
    if fn_step1 is None:
        raise ImportError("Step1 function not found in step1_double_keys.py")

    out1 = call_with_supported_kwargs(
        fn_step1,
        x_ins=x_ins,
        x_que=x_que,
        llm_generate=llm_generate,
)

    double_keys = out1
    low_level_keys, high_level_keys, keys_relation = _normalize_step1_doublekeys(out1)


  
    m_step2 = _import_module("step2_question_graph")
    fn_step2 = resolve_func(m_step2, ["step2_build_question_subgraphs", "step2_build_question_graph", "build_question_graph", "step2_question_graph"])
    if fn_step2 is None:
        G_que_ell = None
        G_que_h = None
        Entities_que = EntitiesTable(rows=[])
        Relations_que = RelationsTable(rows=[])
    else:
        out2 = call_with_supported_kwargs(
            fn_step2,
            x_ins=x_ins, x_que=x_que,
            double_keys=double_keys,
            low_level_keys=low_level_keys,
            high_level_keys=high_level_keys,
            keys_relation=keys_relation,
            keys_relation_list=keys_relation,
            keys_relation_obj=keys_relation,
            llm_generate=llm_generate,
        )
        if isinstance(out2, dict):
            G_que_ell = out2.get("G_que_ell") or out2.get("G_que_l") or out2.get("G_que_low")
            G_que_h = out2.get("G_que_h") or out2.get("G_que_high")
            Entities_que = out2.get("Entities_que", EntitiesTable(rows=[]))
            Relations_que = out2.get("Relations_que", RelationsTable(rows=[]))
        else:
            G_que_ell, G_que_h, Entities_que, Relations_que = out2


    doc_ids = [f"doc{i}" for i in range(len(x_doc))]
    recog_packs: Dict[str, Dict[str, Any]] = {}
    for doc_id, x_i_doc in zip(doc_ids, x_doc):
        recog_packs[doc_id] = run_step3_recog_cached(
            x_ins=x_ins,
            doc_id=doc_id,
            x_i_doc=x_i_doc,
            llm_generate=llm_generate,
            cache_dir=recog_cache_dir,
            enable_cache=recog_enable_cache,
        )

 
    m_step4 = _import_module("step4_align")
    fn_step4 = resolve_func(m_step4, ["step4_align_write_back_align_edges", "step4_align_entity_relation_union", "step4_align", "align_entity_relation_union"])
    if fn_step4 is None:
        raise ImportError("Step4 function not found in step4_align.py")

    union_subgraphs = []
    align_out_by_doc = {}

    for doc_id in doc_ids:
        pack = recog_packs[doc_id]

       
        out4 = None
        try:
                 
                    out4 = fn_step4(
                        doc_id,
                        G_que_ell,
                        G_que_h,
                        Entities_que,
                        Relations_que,
                        pack.get("source_chunks_i") or pack.get("source_chunks") or [],
                        pack.get("E_i") or pack.get("E") or [],
                        pack.get("R_i") or pack.get("R") or [],
                        tau_E=tau_E,
                        tau_R=tau_R,
                        B_i=B_i,
                        high_level_keys=high_level_keys,
                        enable_high_level_gate=enable_high_level_gate,
                    )
        except TypeError:
                   
                    out4 = call_with_supported_kwargs(
                        fn_step4,
                        doc_id=doc_id,
                        G_que_ell=G_que_ell,
                        G_que_h=G_que_h,
                        Entities_que=Entities_que,
                        Relations_que=Relations_que,
                        source_chunks_i=pack.get("source_chunks_i") or pack.get("source_chunks") or [],
                        E_i=pack.get("E_i") or pack.get("E") or [],
                        R_i=pack.get("R_i") or pack.get("R") or [],
                        tau_E=tau_E,
                        tau_R=tau_R,
                        B_i=B_i,
                        high_level_keys=high_level_keys,
                        enable_high_level_gate=enable_high_level_gate,
                    )

    
        union_subgraphs.append(out4)
        if isinstance(out4, dict):
            align_out_by_doc[doc_id] = out4.get("align_out", out4.get("align_out_by_doc", {}))
        else:
            align_out_by_doc[doc_id] = None

    # Step5
    m_step5 = _import_module("step5_compress")
    fn_step5 = resolve_func(
        m_step5,
        ["step5_score_rank_diffseq_adaptive_kprime", "rank_and_select", "run_step5", "step5_compress"],
    )

    if fn_step5 is None:
    
        kept_doc_ids = doc_ids[:1]
        k_prime = 1
        scores = {d: 0.0 for d in doc_ids}
    else:
        out5 = call_with_supported_kwargs(fn_step5, union_subgraphs=union_subgraphs, eta=0.1)
        if hasattr(out5, "model_dump"):
            out5 = out5.model_dump()
        if isinstance(out5, dict):
            kept_doc_ids = out5.get("kept_doc_ids", [])
            k_prime = out5.get("k_prime", len(kept_doc_ids))
            scores = out5.get("scores", {})
        else:
        
            try:
                scores, ranked, kept_doc_ids, k_prime = out5[0], out5[1], out5[2], out5[3]
            except Exception:
                kept_doc_ids = doc_ids[:1]
                k_prime = 1
                scores = {d: 0.0 for d in doc_ids}

    kept_union = [u for u, d in zip(union_subgraphs, doc_ids) if d in kept_doc_ids]


    m_step6 = _import_module("step6_global_union")
    fn_step6 = resolve_func(
        m_step6,
        ["step6_global_union_dedup_budget_self_think_generate", "step6_global_union", "build_global_union_graph"],
    )
    if fn_step6 is None:
        raise ImportError("Step6 function not found in step6_global_union.py")

    out6 = call_with_supported_kwargs(
        fn_step6,
        x_ins=x_ins,
        x_que=x_que,
        union_subgraphs=kept_union,
        recog_packs=recog_packs,
        llm_generate=llm_generate,
    )

    global_union = None
    token_plan = None
    prompt = ""

    if hasattr(out6, "model_dump"):
        out6 = out6.model_dump()

    if isinstance(out6, dict):
        global_union = out6.get("global_union", out6.get("GlobalUnionGraph", None))
        token_plan = out6.get("token_plan", out6.get("TokenPlan", None))
        prompt = out6.get("prompt", out6.get("prompt_revised", out6.get("prompt_initial", "")))
    else:
      
        try:
            global_union, token_plan, prompt = out6
        except Exception:
            prompt = ""

    t1 = _now()

    return {
        "Step1": {
            "low_level_keys": low_level_keys,
            "high_level_keys": high_level_keys,
            "keys_relation": keys_relation,
        },
        "Step2": {
            "Entities_que": _dump(Entities_que),
            "Relations_que": _dump(Relations_que),
        },
        "Step3": {
            "recog_packs": recog_packs,
            "cache_dir": recog_cache_dir,
            "enable_cache": recog_enable_cache,
        },
        "Step4": {
            "union_subgraphs": _dump(union_subgraphs),
            "align_out_by_doc": _dump(align_out_by_doc),
        },
        "Step5": {
            "scores": _dump(scores),
            "k_prime": k_prime,
            "kept_doc_ids": kept_doc_ids,
        },
        "Step6": {
            "global_union": _dump(global_union),
            "token_plan": _dump(token_plan),
            "prompt": prompt,
        },
        "scores": _dump(scores),
        "k_prime": k_prime,
        "kept_doc_ids": kept_doc_ids,
        "global_union": _dump(global_union),
        "token_plan": _dump(token_plan),
        "prompt": prompt,
        "prompt_preview": (prompt or "")[:800],
        "meta": {
            "time_sec": round(t1 - t0, 4),
            "n_docs": len(x_doc),
        },
    }


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    cwd = os.getcwd()
    data_path = os.path.join(cwd, "toy_dataset.jsonl")
    out_dir = os.path.join(cwd, "run_outputs")
    _ensure_dir(out_dir)

    print("CWD:", cwd)
    print("DATA:", data_path)
    print("OUT :", out_dir)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data not found: {data_path}")

    first = None
    for ex in _iter_jsonl(data_path):
        first = ex
        break
    if first is None:
        raise RuntimeError("toy_dataset.jsonl is empty")

    x_ins = first.get("x_ins", "S_ins")
    x_que = first.get("x_que", "")
    x_doc = first.get("x_doc", [])
    if not isinstance(x_doc, list):
        raise ValueError("x_doc must be a list[str]")

    out = uniongraph_rag_pipeline(
        x_ins=x_ins,
        x_doc=x_doc,
        x_que=x_que,
        llm_generate=None,
        recog_cache_dir=DEFAULT_RECOG_CACHE_DIR,
        recog_enable_cache=True,
    )

    save_path = os.path.join(out_dir, "results.jsonl")
    _write_text(save_path, json.dumps(out, ensure_ascii=False, indent=2))
    print("DONE.")
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
