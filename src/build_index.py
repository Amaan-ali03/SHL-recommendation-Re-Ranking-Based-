"""
Build embedding index from catalog_individual.json
Run: python -m src.build_index
"""
import os, json, numpy as np
import re
from sentence_transformers import SentenceTransformer

IN_PATH = "src/index/catalog_individual.json"
OUT_DIR = "src/index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def _clean_text(s: str) -> str:
    if not s: return ""
    # remove that site-level boilerplate
    s = re.sub(r'Outdated browser detected.*?Latest browser options', ' ', s, flags=re.S|re.I)
    s = re.sub(r'Global Offices.*', ' ', s, flags=re.S|re.I)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

def load_items(path=IN_PATH):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    for x in items:
        name = x.get("name","") or ""
        desc = _clean_text(x.get("description") or "")
        tt = x.get("test_type") or ""
        tt_full = {
            "A":"Ability & Aptitude", "B":"Biodata & Situational Judgement",
            "C":"Competencies", "D":"Development & 360", "E":"Assessment Exercises",
            "K":"Knowledge & Skills", "P":"Personality & Behavior", "S":"Simulations"
        }.get(tt,"")
        # keep embed text concise
        embed_parts = [name]
        if desc and len(desc) < 800:
            embed_parts.append(desc)
        embed_parts.append(f"Test Type: {tt} {tt_full}")
        langs = []
        if x.get("languages"):
            # languages may be a noisy long blob (clean and keep first few)
            for l in x.get("languages")[:3]:
                if isinstance(l, str):
                    langs.append(l.split('\n')[0].strip())
        if langs:
            embed_parts.append("Languages: " + ", ".join(langs))
        x["_embed_text"] = "\n".join([p for p in embed_parts if p])
    return items


def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Missing {IN_PATH}. Run the crawler first: python -m src.crawl_shl_catalog")
    items = load_items()
    texts = [it["_embed_text"] for it in items]
    print("Loading model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    print("Encoding", len(texts), "items (this may take a minute)...")
    embs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), embs)
    with open(os.path.join(OUT_DIR, "items.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print("Saved embeddings:", os.path.join(OUT_DIR, "embeddings.npy"))
    print("Saved items:", os.path.join(OUT_DIR, "items.json"))
    print("Embedding shape:", embs.shape)

if __name__ == "__main__":
    main()
