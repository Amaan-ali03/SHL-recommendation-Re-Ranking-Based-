"""
Recommender class with optional Cross-Encoder.
If the 'cross-encoder' package is not available, this falls back to a bi-encoder only pipeline.

Key improvements:
 - token_set_ratio for fuzzy matching
 - conservative fuzzy / intent weights to avoid double-counting semantic signal
 - lexical language + role boost to prefer explicit-language matches (e.g., "python")
 - normalize cross-encoder scores (min-max) and reduce CE weight to 0.6
 - tighter retrieval window by default (topk = max(40, k*10))
 - debug mode returns detailed candidate signals for inspection
"""
import json
import re
import os
from typing import List, Optional
import numpy as np
import httpx
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import importlib
CrossEncoder = None
CROSS_AVAILABLE = False
for _mod in ("cross_encoder", "sentence_transformers.cross_encoder"):
    try:
        _module = importlib.import_module(_mod)
        CrossEncoder = getattr(_module, "CrossEncoder", None)
        if CrossEncoder is not None:
            CROSS_AVAILABLE = True
            break
    except Exception:
        continue

INDEX_ITEMS = "index/items.json"
INDEX_EMBS = "index/embeddings.npy"

TYPE_MAP = {
    "A": "Ability & Aptitude", "B": "Biodata & Situational Judgement", "C": "Competencies",
    "D": "Development & 360", "E": "Assessment Exercises", "K": "Knowledge & Skills",
    "P": "Personality & Behavior", "S": "Simulations"
}


class Recommender:
    def __init__(self,
                 bi_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cross_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if not os.path.exists(INDEX_ITEMS) or not os.path.exists(INDEX_EMBS):
            raise FileNotFoundError("Index files missing. Run src.build_index first.")
        with open(INDEX_ITEMS, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.emb = np.load(INDEX_EMBS)
        # bi-encoder (always available)
        self.bi = SentenceTransformer(bi_model_name)
        # cross-encoder is optional
        self.reranker = None
        if CROSS_AVAILABLE:
            try:
                self.reranker = CrossEncoder(cross_model_name)
                print("Cross-encoder loaded.")
            except Exception as e:
                # if loading fails, continue without it
                print("Cross-encoder import ok but failed to load model:", e)
                self.reranker = None
        else:
            print("Cross-encoder NOT available; running in bi-encoder fallback mode.")

    def _fetch_url_text(self, url: str) -> str:
        try:
            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                r = client.get(url)
                html = r.text
        except Exception:
            return ""
        txt = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
        txt = re.sub(r"<style.*?</style>", " ", txt, flags=re.S | re.I)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt[:10000]

    def _guess_types(self, text: str) -> List[str]:
        t = (text or "").lower()
        wants = set()
        if re.search(r"\b(java(script)?|python|sql|react|node|c\+\+|c#|aws|docker|kubernetes|devops|engineer|developer|programming|data|machine|ml|ai|deep)\b", t):
            wants.add("K")
        if re.search(r"\b(collaborat|stakeholder|communication|teamwork|leadership|behaviour|behavior|personality)\b", t):
            wants.add("P")
        if re.search(r"\b(numerical|verbal|abstract|reasoning|aptitude|cognitive|ability)\b", t):
            wants.add("A")
        if re.search(r"\b(simulation|simulations|scenario)\b", t):
            wants.add("S")
        if re.search(r"\b(competenc|competency)\b", t):
            wants.add("C")
        if not wants:
            if re.search(r"\b(dev|engineer|developer|analyst|data)\b", t):
                wants.add("K")
            else:
                wants.add("P")
        return list(wants)

    def _balance(self, cands, wanted_types, k):
        """Balance results across wanted test types (simple round-robin)."""
        if not wanted_types or k <= 0:
            return cands[:k]
        out = []
        used_urls = set()
        buckets = {tt: [c for c in cands if c.get("test_type") == tt] for tt in wanted_types}
        done = False
        while len(out) < k and not done:
            done = True
            for tt in wanted_types:
                lst = buckets.get(tt, [])
                if not lst:
                    continue
                done = False
                while lst and len(out) < k:
                    cand = lst.pop(0)
                    if cand.get("url") in used_urls:
                        continue
                    out.append(cand)
                    used_urls.add(cand.get("url"))
                    break
        # fill remaining with highest pre-ranked ones
        for cand in cands:
            if len(out) >= k:
                break
            if cand.get("url") in used_urls:
                continue
            out.append(cand)
            used_urls.add(cand.get("url"))
        return out[:k]

    def recommend(self, query: str = None, jd_url: str = None, k: int = 10, debug: bool = False):
        """
        Recommend top-k assessments for the given query or job-description URL.

        Returns:
          - if debug == False: list of result dicts (same as before)
          - if debug == True: returns the candidate list (top retrieval window) with signal fields
        """
        if jd_url and not query:
            query = self._fetch_url_text(jd_url) or jd_url
        if not query:
            raise ValueError("Provide a query or jd_url")

        q_emb = self.bi.encode([query], normalize_embeddings=True)[0]

        # similarity with catalog embeddings (assumes self.emb is normalized)
        sims = (self.emb @ q_emb)

        # retrieval window: moderate but slightly tighter to reduce far-noise
        topk = min(len(sims), max(40, k * 10))
        idxs = np.argsort(-sims)[:topk]

        cands = []
        # Prepare query token set for lexical checks (keep tokens like 'c++' and 'c#')
        q_tokens = set(re.findall(r'(?:c\+\+|c#|[A-Za-z0-9_#+\-]+)', (query or "").lower()))

        lang_tokens = ['python', 'java', 'c++', 'c#', 'c', 'javascript', 'js', 'r', 'sql', 'react', 'node']
        role_tokens = ['developer', 'engineer', 'programmer']

        for i in idxs:
            it = dict(self.items[i])
            it["_sim"] = float(sims[i])  # cosine similarity (if embeddings normalized)

            # safer fuzzy/name matching using token_set_ratio
            name_and_desc = (it.get("name", "") + " " + (it.get("description") or "")).lower().strip()
            kw = fuzz.token_set_ratio(query.lower(), name_and_desc) / 100.0

            # adaptive fuzzy weight: conservative
            kw_weight = 0.06 if len(query.split()) <= 6 else 0.03

            # dynamic semantic intent boost: small to avoid double counting the same vector signal
            item_vec = self.emb[i]
            intent_sim = float(np.dot(q_emb, item_vec))
            intent_sim = max(-1.0, min(1.0, intent_sim))
            intent_sim = (intent_sim + 1.0) / 2.0  # map to [0,1]

            # ---------- lexical language / role boost ----------
            lex_boost = 0.0
            # boost candidates that explicitly mention the language token in name/desc/url/languages
            for tok in lang_tokens:
                if tok in q_tokens:
                    if tok in name_and_desc:
                        lex_boost += 0.10   # strong: language present in name/desc
                    elif tok in (it.get('url') or '').lower():
                        lex_boost += 0.06   # moderate: language in url
                    elif any(tok in (s or '').lower() for s in it.get('languages', [])):
                        lex_boost += 0.03   # small: language listed in languages field

            # role-awareness: prefer items whose name/desc include role tokens if query asks for them
            for rt in role_tokens:
                if rt in q_tokens:
                    if re.search(r'\b' + re.escape(rt) + r'\b', name_and_desc):
                        lex_boost += 0.03

            # combine into a pre_score (embedding sim + small fuzzy + tiny intent boost + lexical boost)
            it["_pre_score"] = it["_sim"] + (kw_weight * kw) + (0.04 * intent_sim) + float(lex_boost)

            # slight demotion for candidates that do NOT mention the language token when query is very short and language present
            if any(tok in q_tokens for tok in lang_tokens) and len(query.split()) <= 4:
                if not any(tok in name_and_desc for tok in lang_tokens):
                    it["_pre_score"] = it["_pre_score"] * 0.92  # small demotion to increase precision

            # save useful debugging fields
            it["_kw"] = float(kw)
            it["_kw_weight"] = float(kw_weight)
            it["_intent_sim"] = float(intent_sim)
            it["_lex_boost"] = float(lex_boost)

            # short text for cross-encoder reranker pairs
            it["_embed_text"] = (it.get("name", "") + " " + (it.get("description") or "")).strip()

            cands.append(it)

        if self.reranker is not None and len(cands) > 0:
            pairs = [(query, c["_embed_text"]) for c in cands]
            try:
                ce_scores = self.reranker.predict(pairs)
            except Exception as e:
                print("Cross-encoder prediction failed:", e)
                ce_scores = None

            if ce_scores is not None:
                ce_scores = np.asarray(ce_scores, dtype=float)
                smin, smax = float(np.min(ce_scores)), float(np.max(ce_scores))
                denom = (smax - smin) if (smax - smin) > 1e-8 else 1.0
                ce_norm = (ce_scores - smin) / denom

                for c, ce_n, raw_s in zip(cands, ce_norm.tolist(), ce_scores.tolist()):
                    c["_ce_raw"] = float(raw_s)
                    c["_ce_norm"] = float(ce_n)
                    bounded_pre = float(c["_pre_score"] / (1.0 + abs(c["_pre_score"])))
                    # reduce CE weight slightly to allow lexical boost to have effect
                    c["_final"] = 0.6 * c["_ce_norm"] + 0.4 * bounded_pre
                cands.sort(key=lambda x: x.get("_final", 0.0), reverse=True)
            else:
                cands.sort(key=lambda x: x.get("_pre_score", 0.0), reverse=True)
        else:
            cands.sort(key=lambda x: x.get("_pre_score", 0.0), reverse=True)

        # infer desired test types from the query and balance results
        wanted = self._guess_types(query)
        out_cands = self._balance(cands, wanted, k)

        # If debug, return the candidate info for inspection (full window)
        if debug:
            return cands

        # output format (same shape as before)
        results = []
        for r in out_cands:
            results.append({
                "assessment_name": r.get("name"),
                "url": r.get("url"),
                "test_type": r.get("test_type"),
                "assessment_length_min": r.get("assessment_length_min"),
                "languages": r.get("languages", [])
            })
        return results


if __name__ == "__main__":
    rec = Recommender()
    print("Loaded items:", len(rec.items))
    print(rec.recommend(query="Hiring a Java developer who collaborates with stakeholders", k=6))
