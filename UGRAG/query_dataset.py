from __future__ import annotations

import os
import sys
import json
import time
import inspect
import importlib
from typing import Any, Dict, List, Optional

CWD = os.getcwd()
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "toy_dataset.json")
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "run_outputs")

TAU_E_DEFAULT = 0.55
TAU_R_DEFAULT = 0.55

ETA_DEFAULT = 0.35
EPS_DEFAULT = 1e-9

def info(msg: str):
    print(f"[INFO] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def dbg(tag: str, msg: str):
    print(f"[{tag}] {msg}")


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def safe_import(module_name: str):
    try:
        m = importlib.import_module(module_name)
        path = getattr(m, "__file__", None)
        if path:
            print(f"[LOADED] {module_name} from: {path}")
        else:
            print(f"[LOADED] {module_name}")
        return m
    except Exception as e:
        warn(f"Failed to import {module_name}: {e}")
        return None


def resolve_func(module, cand_names: List[str]):
    if module is None:
        return None
    for n in cand_names:
        if hasattr(module, n) and callable(getattr(module, n)):
            return getattr(module, n)
    return None


def call_with_supported_kwargs(fn, **kwargs):
    """
    Call fn with only supported kwargs (prevents unexpected keyword errors).
    If fn raises TypeError for missing required args, we re-raise.
    """
    if fn is None:
        return None
    sig = inspect.signature(fn)
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return fn(**accepted)

TOY_FALLBACK = {
    "queries": [
        {
            "qid": "q0",
            "x_ins": "Answer the question based only on the documents. If conflicts exist, prefer the most supported statement.",
            "x_que": "What is the capital of France?",
            "docs": [
                {"doc_id": "doc0", "text": "France's capital city is Paris."},
                {"doc_id": "doc1", "text": "France is a country in Europe. Paris is its capital."},
                {"doc_id": "doc2", "text": "Berlin is the capital of Germany."},
            ],
        }
    ]
}


def load_json_or_jsonl(path: str) -> Dict[str, Any]:
    if not path:
        warn("DATA path is empty -> use fallback toy dataset.")
        return TOY_FALLBACK

    if not os.path.exists(path):
        warn(f"Dataset file not found: {path} -> use fallback toy dataset.")
        return TOY_FALLBACK

    try:
        if os.path.getsize(path) == 0:
            warn(f"Dataset file is empty: {path} -> use fallback toy dataset.")
            return TOY_FALLBACK
    except Exception:
        pass

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".jsonl":
            queries = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    queries.append(json.loads(s))
            if not queries:
                warn(f"JSONL has no valid lines: {path} -> use fallback toy dataset.")
                return TOY_FALLBACK
            return {"queries": queries}

        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                warn(f"Dataset file is empty: {path} -> use fallback toy dataset.")
                return TOY_FALLBACK
            obj = json.loads(content)

        if isinstance(obj, list):
            return {"queries": obj}
        if isinstance(obj, dict):
            if "queries" in obj and isinstance(obj["queries"], list):
                return {"queries": obj["queries"]}
            if "results" in obj and isinstance(obj["results"], list):
                return {"queries": obj["results"]}
            if "x_que" in obj and "docs" in obj:
                return {"queries": [obj]}
            warn(f"Unrecognized JSON schema in {path} -> use fallback toy dataset.")
            return TOY_FALLBACK

        warn(f"Unrecognized dataset type in {path} -> use fallback toy dataset.")
        return TOY_FALLBACK

    except json.JSONDecodeError as e:
        warn(f"JSON decode failed: {e} -> use fallback toy dataset.")
        return TOY_FALLBACK
    except Exception as e:
        warn(f"Load dataset failed: {e} -> use fallback toy dataset.")
        return TOY_FALLBACK

def build_q_graph_for_step4(q_graph: dict) -> dict:
    """Make q_graph compatible with different step4 implementations."""
    q_low = q_graph.get("low", {}) or {}
    q_high = q_graph.get("high", {}) or {}

    E_flat = (q_low.get("E", []) or []) + (q_high.get("E", []) or [])
    R_flat = (q_low.get("R", []) or []) + (q_high.get("R", []) or [])

    return {
        "low": q_low,
        "high": q_high,
        "E": E_flat,
        "R": R_flat,
        "entities": E_flat,
        "relations": R_flat,
        "Entities": E_flat,
        "Relations": R_flat,
    }


def build_d_graph_for_step4(recog_pack: dict) -> dict:
    E = recog_pack.get("E_i", []) or []
    R = recog_pack.get("R_i", []) or []
    D1 = recog_pack.get("Description1_i", {}) or {}
    chunks = recog_pack.get("source_chunks_i", []) or []

    return {
        "E_i": E,
        "R_i": R,
        "Description1_i": D1,
        "source_chunks_i": chunks,
        "E": E,
        "R": R,
        "entities": E,
        "relations": R,
        "Entities": E,
        "Relations": R,
        "desc_map": D1,
    }

def safe_name(e: Dict[str, Any]) -> str:
    return str(e.get("name", "")).strip()


def merge_entities_unique(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for e in entities:
        key = safe_name(e).lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def build_align_edges_from_matches(doc_id: str, M_i_E: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    edges = []
    for j, m in enumerate(M_i_E or []):
        qid = m.get("q_eid") or m.get("q") or m.get("qid") or m.get("head")
        did = m.get("d_eid") or m.get("d") or m.get("did") or m.get("tail")
        if not qid or not did:
            continue
        score = m.get("score", "")
        edges.append(
            {
                "id": f"{doc_id}_align_{j}",
                "head": qid,
                "tail": did,
                "type": "align",
                "description": f"entity-align score={score}".strip(),
            }
        )
    return edges


def build_union_subgraph(doc_id: str, q_graph4: dict, recog_pack: dict, align_out: dict) -> Dict[str, Any]:

    qE = q_graph4.get("E", []) or []
    qR = q_graph4.get("R", []) or []

    dE = recog_pack.get("E_i", []) or []
    dR = recog_pack.get("R_i", []) or []
    D1 = recog_pack.get("Description1_i", {}) or {}

    M_i_E = align_out.get("M_i_E", []) or []
    align_edges = align_out.get("R_i_align", None)
    if align_edges is None:
        align_edges = build_align_edges_from_matches(doc_id, M_i_E)

    E_prime = merge_entities_unique(list(qE) + list(dE))
    R_prime = list(qR) + list(dR) + list(align_edges)

    return {
        "doc_id": doc_id,
        "E_prime_i": E_prime,
        "R_prime_i": R_prime,
        "Description1_prime_i": D1,  
    }


def run_step2(step2_module, x_ins: str, x_que: str, llm_generate=None) -> Dict[str, Any]:

    fn = resolve_func(step2_module, [
        "step2_build_question_graph",
        "build_question_graph",
        "run_step2",
        "step2_question_graph",
        "question_graph",
    ])

    if fn is not None:
        try:
            out = call_with_supported_kwargs(fn, x_ins=x_ins, x_que=x_que, llm_generate=llm_generate)
            if isinstance(out, dict):
                if "low" in out and "high" in out:
                    return out
                if "E" in out or "R" in out:
                    return {"low": {"E": out.get("E", []), "R": out.get("R", [])}, "high": {"E": [], "R": []}}
        except Exception as e:
            warn(f"Step2 failed -> fallback. err={e}")

    import re
    words = re.findall(r"[A-Z][a-zA-Z]{2,}", x_que or "")
    lowE = [{"id": f"qL_e{i}", "name": w, "type": "QuestionEntity"} for i, w in enumerate(words[:10])]
    lowR = []
    for i in range(len(lowE) - 1):
        lowR.append({"id": f"qL_r{i}", "head": lowE[i]["id"], "tail": lowE[i + 1]["id"], "type": "question_co_occurrence"})

    kws = []
    for t in (x_que or "").lower().split():
        t = "".join(ch for ch in t if ch.isalnum())
        if t and t not in kws:
            kws.append(t)
    highE = [{"id": f"qH_e{i}", "name": w, "type": "QuestionTopic"} for i, w in enumerate(kws[:8])]
    highR = []
    for i in range(len(highE) - 1):
        highR.append({"id": f"qH_r{i}", "head": highE[i]["id"], "tail": highE[i + 1]["id"], "type": "semantic_flow"})

    return {"low": {"E": lowE, "R": lowR}, "high": {"E": highE, "R": highR}}


def run_step3(step3_module, x_ins: str, doc_id: str, x_i_doc: str, llm_generate=None) -> Dict[str, Any]:

    RecogCls = getattr(step3_module, "Recog", None) if step3_module else None
    Step3Config = getattr(step3_module, "Step3Config", None) if step3_module else None

    if RecogCls is None:
        return {"source_chunks_i": [x_i_doc], "E_i": [], "R_i": [], "Description1_i": {}}

    cfg = None
    try:
        cfg = Step3Config() if callable(Step3Config) else None
    except Exception:
        cfg = None

    try:
        recog = RecogCls(x_ins=x_ins, x_i_doc=x_i_doc, doc_id=doc_id, cfg=cfg, llm_generate=llm_generate)
        out = recog()
    except Exception as e:
        warn(f"Step3 failed on {doc_id} -> fallback minimal. err={e}")
        out = {}

    if not isinstance(out, dict):
        return {"source_chunks_i": [x_i_doc], "E_i": [], "R_i": [], "Description1_i": {}}

    pack = {
        "source_chunks_i": out.get("source_chunks_i") or out.get("source_chunks") or [x_i_doc],
        "E_i": out.get("E_i") or out.get("E") or [],
        "R_i": out.get("R_i") or out.get("R") or [],
        "Description1_i": out.get("Description1_i") or out.get("Description1") or {},
    }

    pack["source_chunks"] = pack["source_chunks_i"]
    pack["E"] = pack["E_i"]
    pack["R"] = pack["R_i"]
    pack["Description1"] = pack["Description1_i"]
    return pack


def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum() or ch in (" ", "_", "-"):
            out.append(ch)
    return "".join(out).strip()


def _fallback_entity_match(q_graph4: dict, d_graph4: dict) -> List[Dict[str, Any]]:
    qE = q_graph4.get("E", []) or []
    dE = d_graph4.get("E", []) or []

    stop = {"what", "which", "who", "where", "when", "why", "how"}
    q_map = {}
    for e in qE:
        name = _norm_name(e.get("name", ""))
        if not name or name in stop:
            continue
        q_map[name] = e.get("id")

    d_map = {}
    for e in dE:
        name = _norm_name(e.get("name", ""))
        if not name:
            continue
        d_map[name] = e.get("id")

    matches = []
    for name, qid in q_map.items():
        if name in d_map:
            matches.append({"q_eid": qid, "d_eid": d_map[name], "score": 1.0})
    return matches


def run_step4(step4_module, q_graph4: dict, d_graph4: dict, doc_id: str, tau_E: float, tau_R: float) -> Dict[str, Any]:

    fn = resolve_func(step4_module, [
        "step4_align",
        "align",
        "run_step4",
        "align_graphs",
    ])

    out = None
    if fn is not None:
        try:
            out = call_with_supported_kwargs(
                fn,
                q_graph=q_graph4,
                d_graph=d_graph4,
                doc_id=doc_id,
                tau_E=tau_E,
                tau_R=tau_R,
            )
        except Exception as e:
            warn(f"Step4 failed on {doc_id} -> fallback. err={e}")
            out = None

    if not isinstance(out, dict):
        out = {"doc_id": doc_id, "M_i_E": [], "M_i_R": []}
    out.setdefault("doc_id", doc_id)
    out.setdefault("M_i_E", [])
    out.setdefault("M_i_R", [])

    fb = _fallback_entity_match(q_graph4, d_graph4)

    if fb:
        seen = set()
        merged = []
        for m in (out.get("M_i_E", []) or []) + fb:
            qid = m.get("q_eid")
            did = m.get("d_eid")
            if not qid or not did:
                continue
            key = (qid, did)
            if key in seen:
                continue
            seen.add(key)
            merged.append(m)
        out["M_i_E"] = merged

    return out



def run_step5(
    step5_module,
    x_ins: str,
    x_que: str,
    q_graph: Dict[str, Any],
    recog_packs: Dict[str, Any],
    align_out_by_doc: Dict[str, Any],
    union_subgraphs: Dict[str, Any],
    eta: float = ETA_DEFAULT,
    eps: float = EPS_DEFAULT,
) -> Dict[str, Any]:

    fn = resolve_func(step5_module, [
        "step5_score_rank_diffseq_adaptive_kprime",
        "step5_compress",
        "compress",
        "run_step5",
        "rank_and_select",
    ])

    if fn is not None:
        try:
            out = call_with_supported_kwargs(
                fn,
                x_ins=x_ins,
                x_que=x_que,
                q_graph=q_graph,
                recog_packs=recog_packs,
                align_out_by_doc=align_out_by_doc,
                union_subgraphs=union_subgraphs,
                eta=eta,
                eps=eps,
            )
            if isinstance(out, dict):
                return out
        except Exception as e:
            warn(f"Step5 failed -> fallback ranking. err={e}")

    qE = (q_graph.get("low", {}).get("E", []) or []) + (q_graph.get("high", {}).get("E", []) or [])
    q_names = {safe_name(e).lower() for e in qE if safe_name(e)}
    q_names = {n for n in q_names if n}

    ranked_items = []
    for doc_id, usg in (union_subgraphs or {}).items():
        R = usg.get("R_prime_i", []) or []
        E = usg.get("E_prime_i", []) or []
        d_names = {safe_name(e).lower() for e in E if safe_name(e)}
        d_names = {n for n in d_names if n}

        hits = len([n for n in q_names if n in d_names])
        q_cov = hits / max(1, len(q_names))

        align_cnt = sum(1 for r in R if str(r.get("type", "")).lower() == "align")

        pack = recog_packs.get(doc_id, {}) or {}
        chunks = pack.get("source_chunks_i", []) or []
        chunk_n = len(chunks)

        score = 2.0 * align_cnt + 3.0 * q_cov + 0.2 * chunk_n
        ranked_items.append({
            "doc_id": doc_id,
            "score": float(score),
            "Ei": int(len(pack.get("E_i", []) or [])),
            "Ri": int(len(pack.get("R_i", []) or [])),
            "M_E": int(align_cnt),
            "M_R": int(len((align_out_by_doc.get(doc_id, {}) or {}).get("M_i_R", []) or [])),
            "q_cov": float(q_cov),
            "q_hits": int(hits),
            "chunks": int(chunk_n),
        })

    ranked_items.sort(key=lambda x: x["score"], reverse=True)
    ranked = [[it["doc_id"], it["score"], it["Ei"], it["Ri"]] for it in ranked_items]


    if len(ranked_items) <= 2:
        k_prime = len(ranked_items)
    else:
        s0 = ranked_items[0]["score"]
        s1 = ranked_items[1]["score"]
        s2 = ranked_items[2]["score"]
 
        k_prime = 2 if (s1 - s2) > 0.15 else 3

    kept_doc_ids = [it["doc_id"] for it in ranked_items[:k_prime]]

    diffseq = []
    for i in range(max(0, len(ranked_items) - 1)):
        s_i = ranked_items[i]["score"]
        s_next = ranked_items[i + 1]["score"]
        drop = s_i - s_next
        rel_drop = drop / max(eps, abs(s_i))
        diffseq.append({"i": i, "s_i": s_i, "s_next": s_next, "drop": drop, "rel_drop": rel_drop})

    return {
        "k_prime": int(k_prime),
        "kept_doc_ids": kept_doc_ids,
        "ranked": ranked,
        "ranked_items": ranked_items,
        "diffseq": diffseq,
    }


def run_step6(step6_module, kept_union_subgraphs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build global union graph.
    If step6 not found -> simple dedup merge.
    """
    fn = resolve_func(step6_module, [
        "step6_global_union",
        "build_global_union",
        "run_step6",
        "global_union",
    ])

    if fn is not None:
        try:
            out = call_with_supported_kwargs(
                fn,
                kept_union_subgraphs=kept_union_subgraphs,
                union_subgraphs=kept_union_subgraphs,
            )
            if isinstance(out, dict):
                return out
        except Exception as e:
            warn(f"Step6 failed -> fallback merge. err={e}")

    allE = []
    allR = []
    for _, usg in kept_union_subgraphs.items():
        allE.extend(usg.get("E_prime_i", []) or [])
        allR.extend(usg.get("R_prime_i", []) or [])
    gE = merge_entities_unique(allE)
    gR = allR
    return {"global_union_graph": {"E": gE, "R": gR}}


def main():
    data_path = os.environ.get("DATA", DEFAULT_DATA)
    out_dir = os.environ.get("OUT", DEFAULT_OUT)
    os.makedirs(out_dir, exist_ok=True)

    info(f"CWD: {CWD}")
    info(f"DATA: {data_path}")
    info(f"OUT : {out_dir}")


    llm_generate = None
    try:
        from chugao.llm_deepseek import build_deepseek_llm_generate
        llm_generate = build_deepseek_llm_generate(
            model="deepseek-chat",
            max_tokens=1200,
            temperature=0.1,
        )
    except Exception as e:
        warn(f"LLM init failed -> run without LLM. err={e}")
        llm_generate = None
    print("[DBG] llm_generate ready:", llm_generate is not None)

    step2_m = safe_import("chugao.step2_question_graph") or safe_import("step2_question_graph")
    step3_m = safe_import("chugao.step3_recog") or safe_import("step3_recog")
    step4_m = safe_import("chugao.step4_align") or safe_import("step4_align")
    step5_m = safe_import("chugao.step5_compress") or safe_import("step5_compress")
    step6_m = safe_import("chugao.step6_global_union") or safe_import("step6_global_union")

    data = load_json_or_jsonl(data_path)
    queries = data.get("queries", []) or []

    info(f"start pipeline @ {now_str()}")
    info(f"#queries={len(queries)}")

    results = []
    for qi, q in enumerate(queries):
        qid = q.get("qid", f"q{qi}")
        x_ins = q.get("x_ins", "") or ""
        x_que = q.get("x_que", "") or ""
        docs = q.get("docs", []) or []
        n_docs = len(docs)

        info(f"running {qid} ({qi+1}/{len(queries)}) ...")
        info(f"#docs={n_docs} | x_que_len={len(x_que)}")

  
        q_graph = run_step2(step2_m, x_ins=x_ins, x_que=x_que, llm_generate=llm_generate)
        q_graph4 = build_q_graph_for_step4(q_graph)


        recog_packs: Dict[str, Any] = {}
        for d in docs:
            doc_id = d.get("doc_id", "")
            text = d.get("text", "") or ""
            recog_packs[doc_id] = run_step3(step3_m, x_ins=x_ins, doc_id=doc_id, x_i_doc=text, llm_generate=llm_generate)

        align_out_by_doc: Dict[str, Any] = {}
        union_subgraphs: Dict[str, Any] = {}

        for d in docs:
            doc_id = d.get("doc_id", "")
            pack = recog_packs.get(doc_id, {})
            d_graph4 = build_d_graph_for_step4(pack)

            dbg(
                "DBG4_IN",
                f"{doc_id}: |EQ|={len(q_graph4.get('E', []))} |RQ|={len(q_graph4.get('R', []))} "
                f"|Ei|={len(d_graph4.get('E', []))} |Ri|={len(d_graph4.get('R', []))} |CH|={len(d_graph4.get('source_chunks_i', []))}",
            )

            out4 = run_step4(step4_m, q_graph4=q_graph4, d_graph4=d_graph4, doc_id=doc_id, tau_E=TAU_E_DEFAULT, tau_R=TAU_R_DEFAULT)
            out4.setdefault("doc_id", doc_id)
            out4.setdefault("M_i_E", [])
            out4.setdefault("M_i_R", [])

            if "R_i_align" not in out4:
                out4["R_i_align"] = build_align_edges_from_matches(doc_id, out4.get("M_i_E", []))

            align_out_by_doc[doc_id] = out4

            usg = build_union_subgraph(doc_id, q_graph4=q_graph4, recog_pack=pack, align_out=out4)
            union_subgraphs[doc_id] = usg


        step5_out = call_with_supported_kwargs(
            resolve_func(step5_m, ["step5_compress", "step5_score_rank_diffseq_adaptive_kprime"]),
            x_ins=x_ins,
            x_que=x_que,
            q_graph=q_graph,
            recog_packs=recog_packs,
            align_out_by_doc=align_out_by_doc,
            union_subgraphs=union_subgraphs,
            eta=0.35,
            eps=1e-9,
        )


        kept_doc_ids = step5_out.get("kept_doc_ids", None)
        if not kept_doc_ids:
            ranked = step5_out.get("ranked", []) or []
            kept_doc_ids = [x[0] for x in ranked] if ranked else list(union_subgraphs.keys())

        kept_union_subgraphs = {did: union_subgraphs[did] for did in kept_doc_ids if did in union_subgraphs}


        step6_out = run_step6(step6_m, kept_union_subgraphs=kept_union_subgraphs)


        results.append(
            {
                "qid": qid,
                "x_ins": x_ins,
                "x_que": x_que,
                "docs": docs,

                "q_graph": q_graph,
                "recog_packs": recog_packs,
                "align_out_by_doc": {
                    did: {
                        "doc_id": did,
                        "M_i_E": align_out_by_doc[did].get("M_i_E", []),
                        "M_i_R": align_out_by_doc[did].get("M_i_R", []),
                        "R_i_align": align_out_by_doc[did].get("R_i_align", []),
                    }
                    for did in align_out_by_doc
                },
                "union_subgraphs": union_subgraphs,

                "step5": step5_out,
                "step6": step6_out,

                "meta": {
                    "time": now_str(),
                    "tau_E": TAU_E_DEFAULT,
                    "tau_R": TAU_R_DEFAULT,
                    "n_docs": n_docs,
                },
            }
        )

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "meta": {"time": now_str()}}, f, ensure_ascii=False, indent=2)

    print("DONE.")
    print(f"Saved: {out_path}")


if __name__ == "__main__":

    proj_root = os.path.dirname(os.path.dirname(__file__))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    main()
