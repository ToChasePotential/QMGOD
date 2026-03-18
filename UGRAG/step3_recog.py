
from __future__ import annotations

print("[LOADED] step3_recog from:", __file__)

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
import json


@dataclass
class Step3Config:
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    max_entities_per_chunk: int = 20


    max_chunk_chars: int = 600  # soft cap; if sentence is too long, split by length

class Recog:


    def __init__(
        self,
        *,
        x_ins: str,
        x_i_doc: str,
        doc_id: str,
        cfg: Optional[Step3Config] = None,
        llm_generate=None,
        **kwargs,
    ):
        self.x_ins = x_ins or ""
        self.x_i_doc = x_i_doc or ""
        self.doc_id = doc_id or "doc"
        self.cfg = cfg if cfg is not None else Step3Config()
        self.llm_generate = llm_generate

    def __call__(self) -> Dict[str, Any]:
        chunks = self._split_into_chunks(self.x_i_doc)
        entities, desc_map = self._extract_entities(chunks)


        relations = self._build_relations(chunks, entities)

 
        if not entities and chunks:
            entities, desc_map = self._synthesize_entities_from_chunk0(chunks)

        if not relations and entities:
            relations = self._cooccurrence_relations(entities)
            if relations:
              
                for r in relations:
                    desc_map[r["id"]] = [0]

        return {
            "source_chunks_i": chunks,
            "E_i": entities,
            "R_i": relations,
            "Description1_i": desc_map,
        }


    def _split_into_chunks(self, text: str) -> List[str]:
        if not text:
            return []


        sents = re.split(r"(?:[。！？!?]\s*|\.\s+)", text.strip())
        sents = [s.strip() for s in sents if s and s.strip()]

     
        chunks: List[str] = []
        for s in sents:
            if len(s) <= self.cfg.max_chunk_chars:
                chunks.append(s)
            else:
           
                step = self.cfg.max_chunk_chars
                for i in range(0, len(s), step):
                    part = s[i : i + step].strip()
                    if part:
                        chunks.append(part)
        return chunks

    def _extract_entities(self, chunks: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        entities: List[Dict[str, Any]] = []
        desc_map: Dict[str, List[int]] = {}
        seen = set()

        ent_id = 0

        for idx, chunk in enumerate(chunks):
        
            rule_words = self._rule_extract_words(chunk)


            llm_words: List[str] = []
            if self.llm_generate is not None:
                llm_words = self._llm_extract_words_safe(chunk)


            merged = []
            for w in rule_words + llm_words:
                w = (w or "").strip()
                if not w:
                    continue
                key = w.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(w)

            for w in merged[: self.cfg.max_entities_per_chunk]:
                eid = f"{self.doc_id}_e{ent_id}"
                entities.append(
                    {
                        "id": eid,
                        "name": w,
                        "type": "DocEntity",
                    }
                )
                desc_map[eid] = [idx]
                ent_id += 1

        return entities, desc_map

    def _rule_extract_words(self, chunk: str) -> List[str]:

        multi = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", chunk)

        single = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", chunk)

  
        words = []
        seen = set()
        for w in multi + single:
            w = w.strip()
            key = w.lower()
            if key in seen:
                continue
            seen.add(key)
            words.append(w)
        return words

    def _llm_extract_words_safe(self, chunk: str) -> List[str]:
    
        try:
            prompt = (
                "Extract key named entities from the following text. "
                "Return ONLY a JSON list of entity strings.\n\n"
                f"Text:\n{chunk}\n"
            )
            resp = self.llm_generate(prompt)
            if resp is None:
                return []
            s = str(resp).strip()

        
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [str(x).strip() for x in obj if str(x).strip()]
                if isinstance(obj, dict) and "entities" in obj and isinstance(obj["entities"], list):
                    return [str(x).strip() for x in obj["entities"] if str(x).strip()]
            except Exception:
                pass

    
            parts = re.split(r"[\n,;]+", s)
            out = []
            for p in parts:
                p = p.strip().strip('"').strip("'")
                if p:
                    out.append(p)
            return out[: self.cfg.max_entities_per_chunk]
        except Exception:
            return []

    def _build_relations(self, chunks: List[str], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        1) Pattern relations from chunks (capital_of, located_in)
        2) If none, fallback to co-occurrence
        """
        if not chunks or not entities:
            return []

   
        name2eid = {}
        for e in entities:
            name2eid[e["name"].lower()] = e["id"]

        rels: List[Dict[str, Any]] = []
        rid = 0


        for idx, sent in enumerate(chunks):
            s = sent.strip()
     
            s_norm = s.replace("’", "'")

            m = re.match(r"^(.+?)'s\s+capital(?:\s+city)?\s+is\s+(.+)$", s_norm, flags=re.I)
            if m:
                h_name = m.group(1).strip()
                t_name = m.group(2).strip()
                rid = self._append_relation_if_possible(
                    rels, name2eid, h_name, t_name,
                    rtype="capital_of", rid=rid
                )
                continue

            m = re.match(r"^(?:the\s+)?capital\s+of\s+(.+?)\s+is\s+(.+)$", s_norm, flags=re.I)
            if m:
                h_name = m.group(1).strip()
                t_name = m.group(2).strip()
                rid = self._append_relation_if_possible(
                    rels, name2eid, h_name, t_name,
                    rtype="capital_of", rid=rid
                )
                continue

            m = re.match(r"^(.+?)\s+is\s+(?:located\s+in|in)\s+(.+)$", s_norm, flags=re.I)
            if m:
                h_name = m.group(1).strip()
                t_name = m.group(2).strip()
                rid = self._append_relation_if_possible(
                    rels, name2eid, h_name, t_name,
                    rtype="located_in", rid=rid
                )
                continue

        if not rels:
            rels = self._cooccurrence_relations(entities)

        return rels

    def _append_relation_if_possible(
        self,
        rels: List[Dict[str, Any]],
        name2eid: Dict[str, str],
        head_name: str,
        tail_name: str,
        *,
        rtype: str,
        rid: int,
    ) -> int:
   
        h = name2eid.get(head_name.lower())
        t = name2eid.get(tail_name.lower())


        if h is None:
            for k, v in name2eid.items():
                if k in head_name.lower():
                    h = v
                    break
        if t is None:
            for k, v in name2eid.items():
                if k in tail_name.lower():
                    t = v
                    break

        if h is None or t is None or h == t:
            return rid

        rels.append(
            {
                "id": f"{self.doc_id}_r{rid}",
                "head": h,
                "tail": t,
                "type": rtype,
            }
        )
        return rid + 1

    def _cooccurrence_relations(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rels: List[Dict[str, Any]] = []
  
        for i in range(len(entities) - 1):
            rels.append(
                {
                    "id": f"{self.doc_id}_r{i}",
                    "head": entities[i]["id"],
                    "tail": entities[i + 1]["id"],
                    "type": "co_occurrence",
                }
            )
        return rels

    def _synthesize_entities_from_chunk0(self, chunks: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        desc_map: Dict[str, List[int]] = {}
   
        c0 = chunks[0]
        words = self._rule_extract_words(c0)
        words = words[:2] if words else ["EntityA", "EntityB"]

        ents = []
        for i, w in enumerate(words):
            eid = f"{self.doc_id}_e{i}"
            ents.append({"id": eid, "name": w, "type": "DocEntity"})
            desc_map[eid] = [0]
        return ents, desc_map

def step3_recog(
    *,
    x_ins: str,
    x_i_doc: str,
    doc_id: str,
    cfg=None,
    llm_generate=None,
    **kwargs,
) -> Dict[str, Any]:
    recog = Recog(
        x_ins=x_ins,
        x_i_doc=x_i_doc,
        doc_id=doc_id,
        cfg=cfg,
        llm_generate=llm_generate,
    )
    return recog()


if __name__ == "__main__":
    text = "France's capital city is Paris. France is located in Europe."
    out = step3_recog(x_ins="Use docs only.", x_i_doc=text, doc_id="demo")
    print("chunks:", out["source_chunks_i"])
    print("E_i:", out["E_i"])
    print("R_i:", out["R_i"])
    print("Description1_i:", out["Description1_i"])
