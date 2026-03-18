import json
from step3_recog import Recog, Step3Config

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    import argparse, os
    p=argparse.ArgumentParser()
    p.add_argument('--docs', default='docs.jsonl')
    args=p.parse_args()
    here=os.path.dirname(os.path.abspath(__file__))
    docs_path=args.docs
    if not os.path.isabs(docs_path):
        cand1=os.path.join(os.getcwd(), docs_path)
        cand2=os.path.join(here, docs_path)
        docs_path = cand1 if os.path.exists(cand1) else cand2

    cfg = Step3Config(cache_dir="recog_cache", enable_cache=True)

    for ex in iter_jsonl(docs_path):
        doc_id = ex["doc_id"]
        x_ins = ex["x_ins"]
        x_i_doc = ex["x_doc"]

        _ = Recog(x_ins=x_ins, x_i_doc=x_i_doc, doc_id=doc_id, cfg=cfg, llm_generate=None)
        print(f"[OK] ingested {doc_id}")

if __name__ == "__main__":
    main()
