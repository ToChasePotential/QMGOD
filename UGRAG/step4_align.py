
print("[LOADED] step4_align from:", __file__)

from typing import Dict, List, Any
import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def step4_align(
    *,
    q_graph: Dict[str, Any],
    doc_pack: Dict[str, Any],
    tau_E: float = 0.0,
    tau_R: float = 0.0,
    **kwargs,
) -> Dict[str, Any]:


    doc_E = doc_pack.get("E") or doc_pack.get("E_i", [])
    doc_R = doc_pack.get("R") or doc_pack.get("R_i", [])

    qE = (q_graph.get("low", {}).get("E", []) +
          q_graph.get("high", {}).get("E", []))
    qR = (q_graph.get("low", {}).get("R", []) +
          q_graph.get("high", {}).get("R", []))

    M_E: List[Dict[str, Any]] = []

    for qe in qE:
        qname = _norm(qe.get("name", ""))
        for de in doc_E:
            dname = _norm(de.get("name", ""))
            if not qname or not dname:
                continue
            if qname in dname or dname in qname:
                M_E.append({
                    "q_e": qe["id"],
                    "d_e": de["id"],
                    "score": 1.0,
                })

    M_R: List[Dict[str, Any]] = []

    matched_doc_entities = {m["d_e"] for m in M_E}

    for qr in qR:
        for dr in doc_R:
            if dr.get("head") in matched_doc_entities or \
               dr.get("tail") in matched_doc_entities:
                M_R.append({
                    "q_r": qr["id"],
                    "d_r": dr["id"],
                    "score": 1.0,
                })


    R_align: List[Dict[str, Any]] = []

    for m in M_E:
        R_align.append({
            "id": f"align_e_{m['q_e']}_{m['d_e']}",
            "head": m["q_e"],
            "tail": m["d_e"],
            "type": "align",
            "score": m["score"],
        })

    for m in M_R:
        R_align.append({
            "id": f"align_r_{m['q_r']}_{m['d_r']}",
            "head": m["q_r"],
            "tail": m["d_r"],
            "type": "align",
            "score": m["score"],
        })

    return {
        "M_E": M_E,
        "M_R": M_R,
        "R_align": R_align,
    }
