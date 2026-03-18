import json
import os
print("CWD:", os.getcwd())
print("FILE:", __file__)
from statistics import mean
from main import uniongraph_rag_pipeline  

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            yield json.loads(line)

def safe_len(x):
    try:
        return len(x)
    except Exception:
        return 0

def main():
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "toy_dataset.jsonl")
    print("USING data_path:", data_path)
    tau_E, tau_R = 0.55, 0.55        
    total_budget = 800

    stats = []
    for i, ex in enumerate(load_jsonl(data_path)):
        print("FIRST EXAMPLE:", ex)
        break
    import os

    print("EXISTS:", os.path.exists(data_path))
    print("SIZE:", os.path.getsize(data_path) if os.path.exists(data_path) else None)

    with open(data_path, "r", encoding="utf-8") as f:
        head = f.read(300)
    print("HEAD(300):", repr(head))

    for ex in load_jsonl(data_path):
        rid = ex.get("id", "NA")
        x_ins = ex.get("x_ins", "S_ins")
        x_que = ex["x_que"]
        x_doc = ex["x_doc"]

        out = uniongraph_rag_pipeline(
            x_ins=x_ins,
            x_doc=x_doc,
            x_que=x_que,
            llm_generate=llm_generate,     
        )
        union_outputs = out["Step4"]["union_outputs"]
        n_docs = len(union_outputs)

        mE = []
        mR = []
        align_edges = []
        for u in union_outputs:
            M_iE = u.get("M_iE", [])
            M_iR = u.get("M_iR", [])
            rels_with_align = u.get("Relations_doc_i_with_align", []) or []

            mE.append(safe_len(M_iE))
            mR.append(safe_len(M_iR))
            cnt_align = 0
            for r in rels_with_align:
                if getattr(r, "relation", None) == "align" or (isinstance(r, dict) and r.get("relation") == "align"):
                    cnt_align += 1
            align_edges.append(cnt_align)

        k_prime = out["Step5"].get("k_prime", None)

        G_global = out["Step6"].get("G_global", None)
        nE_global = safe_len(getattr(G_global, "entities", None)) or safe_len(getattr(G_global, "E", None))
        nR_global = safe_len(getattr(G_global, "relations", None)) or safe_len(getattr(G_global, "R", None))

        stats.append({
            "id": rid,
            "n_docs": n_docs,
            "avg_M_iE": mean(mE) if mE else 0.0,
            "avg_M_iR": mean(mR) if mR else 0.0,
            "avg_align_edges": mean(align_edges) if align_edges else 0.0,
            "k_prime": k_prime,
            "global_E": nE_global,
            "global_R": nR_global,
        })

        print(f"[{rid}] docs={n_docs} avg_M_iE={stats[-1]['avg_M_iE']:.2f} avg_M_iR={stats[-1]['avg_M_iR']:.2f} "
              f"align={stats[-1]['avg_align_edges']:.2f} k'={k_prime} global(E,R)=({nE_global},{nR_global})")

    if stats:
        print("\n=== SUMMARY ===")
        print("N =", len(stats))
        print("avg(avg_M_iE) =", mean(s["avg_M_iE"] for s in stats))
        print("avg(avg_M_iR) =", mean(s["avg_M_iR"] for s in stats))
        print("avg(global_E) =", mean(s["global_E"] for s in stats))
        print("avg(global_R) =", mean(s["global_R"] for s in stats))

if __name__ == "__main__":
    main()

