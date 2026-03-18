
from __future__ import annotations
from typing import Any, Dict, List, Tuple


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _get_q_entities(q_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    low = (q_graph.get("low") or {})
    high = (q_graph.get("high") or {})
    qE = (low.get("E") or []) + (high.get("E") or [])
    return qE


def _qid_set_from_matches(M_i_E: List[Dict[str, Any]]) -> set:
    s = set()
    for m in (M_i_E or []):
        qid = m.get("q_eid") or m.get("q") or m.get("qid") or m.get("head")
        if qid:
            s.add(str(qid))
    return s


def _build_align_edges(doc_id: str, M_i_E: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    edges = []
    for j, m in enumerate(M_i_E or []):
        qid = m.get("q_eid") or m.get("q") or m.get("qid") or m.get("head")
        did = m.get("d_eid") or m.get("d") or m.get("did") or m.get("tail")
        if not qid or not did:
            continue
        edges.append(
            {
                "id": f"{doc_id}_align_{j}",
                "head": qid,
                "tail": did,
                "type": "align",
                "description": f"entity-align score={m.get('score', '')}".strip(),
            }
        )
    return edges


def _score_one_doc(
    doc_id: str,
    q_graph: Dict[str, Any],
    recog_packs: Dict[str, Any],
    align_out_by_doc: Dict[str, Any],
) -> Dict[str, Any]:
    pack = recog_packs.get(doc_id, {}) or {}
    aout = align_out_by_doc.get(doc_id, {}) or {}

    M_i_E = aout.get("M_i_E") or []
    M_i_R = aout.get("M_i_R") or []
    R_i_align = aout.get("R_i_align")
    if R_i_align is None:
        R_i_align = _build_align_edges(doc_id, M_i_E)

  
    chunks = pack.get("source_chunks_i") or pack.get("source_chunks") or []
    n_chunks = len(chunks)

    
    qE = _get_q_entities(q_graph)
    q_ids = [str(e.get("id")) for e in qE if e.get("id")]
    q_ids_set = set(q_ids)
    hit_qids = _qid_set_from_matches(M_i_E)
    q_hits = len(q_ids_set.intersection(hit_qids)) if q_ids_set else 0
    q_cov = (q_hits / max(1, len(q_ids_set))) if q_ids_set else 0.0

 
    n_me = len(M_i_E)
    n_mr = len(M_i_R)
    n_align = len(R_i_align)

    score = 6.0 * q_cov + 2.0 * n_me + 0.5 * n_align + 0.2 * n_chunks + 0.3 * n_mr

    return {
        "doc_id": doc_id,
        "score": float(score),
        "Ei": int(pack.get("E_i") and len(pack.get("E_i")) or 0),
        "Ri": int(pack.get("R_i") and len(pack.get("R_i")) or 0),
        "M_E": int(n_me),
        "M_R": int(n_mr),
        "q_cov": float(q_cov),
        "q_hits": int(q_hits),
        "chunks": int(n_chunks),
    }


def _diffseq(scores_desc: List[float]) -> List[Dict[str, Any]]:
    ds = []
    for i in range(len(scores_desc) - 1):
        s_i = float(scores_desc[i])
        s_n = float(scores_desc[i + 1])
        drop = s_i - s_n
        rel_drop = drop / max(1e-9, abs(s_i))
        ds.append({"i": i, "s_i": s_i, "s_next": s_n, "drop": float(drop), "rel_drop": float(rel_drop)})
    return ds


def _adaptive_kprime(scores_desc: List[float], eta: float = 0.35, eps: float = 1e-9) -> int:
    if not scores_desc:
        return 0
    if len(scores_desc) == 1:
        return 1

    ds = _diffseq(scores_desc)
    for i, d in enumerate(ds):
        if d["rel_drop"] >= eta and d["drop"] > eps:
            return min(len(scores_desc), i + 2) 
    return len(scores_desc)


def step5_compress(
    *,
    x_ins: str = "",
    x_que: str = "",
    q_graph: Dict[str, Any],
    recog_packs: Dict[str, Any],
    align_out_by_doc: Dict[str, Any],
    union_subgraphs: Dict[str, Any] = None,
    eta: float = 0.35,
    eps: float = 1e-9,
) -> Dict[str, Any]:
    doc_ids = list(recog_packs.keys())
    items = [_score_one_doc(did, q_graph, recog_packs, align_out_by_doc) for did in doc_ids]


    items.sort(key=lambda x: x["score"], reverse=True)
    scores_desc = [it["score"] for it in items]

    k_prime = _adaptive_kprime(scores_desc, eta=eta, eps=eps)
    kept_doc_ids = [it["doc_id"] for it in items[:k_prime]]

    ranked = [[it["doc_id"], it["score"], it["M_E"], it["M_R"]] for it in items]
    ds = _diffseq(scores_desc)

    return {
        "k_prime": int(k_prime),
        "kept_doc_ids": kept_doc_ids,
        "ranked": ranked,             
        "ranked_items": items,         
        "diffseq": ds,
        "eta": float(eta),
        "eps": float(eps),
    }
