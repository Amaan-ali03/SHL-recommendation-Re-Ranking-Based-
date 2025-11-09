"""
Normalized evaluation of baseline vs improved.
This version normalizes gold assessment labels (URLs or names) to match the catalog,
then computes Mean Recall@10 and writes submission files and eval_results.json.
"""
import json, os, re
import numpy as np
import pandas as pd
from collections import defaultdict
from src.config import DATASET_PATH
from src.recommender import Recommender
from sentence_transformers import SentenceTransformer

# helpers
def norm_url(u):
    if not isinstance(u, str): return u
    u=u.strip()
    if not u: return u
    u = u.split('#',1)[0].split('?',1)[0]
    u = u.replace("http://","https://").rstrip('/')
    u = u.lower()
    u = u.replace("/solutions/products/product-catalog", "/products/product-catalog")
    u = u.replace("/solutions/products", "/products")
    u = re.sub(r'//+', '/', u)
    if u.startswith('https:/') and not u.startswith('https://'):
        u = u.replace('https:/', 'https://', 1)
    return u

def norm_any(x):
    if not isinstance(x, str): return x
    x=x.strip()
    if x.startswith("http"):
        return norm_url(x)
    return x.lower().strip()

# load datasets
xls = pd.ExcelFile(DATASET_PATH)
train_df = pd.read_excel(xls, sheet_name="Train-Set")
test_df = pd.read_excel(xls, sheet_name="Test-Set")

# load items & index
items = json.load(open("src/index/items.json", "r", encoding="utf-8"))
embs = np.load("src/index/embeddings.npy")

# build name -> canonical url map from catalog
name_to_url = {}
url_set = set()
for it in items:
    url = norm_url(it.get("url",""))
    url_set.add(url)
    name_to_url[it.get("name","").lower()] = url

# function to map a gold label (may be url or name) to canonical catalog url (if possible)
def map_gold(g):
    gstr = str(g).strip()
    if gstr.startswith("http"):
        n = norm_url(gstr)
        # exact match
        if n in url_set:
            return n
        # last path slug match
        slug = n.rstrip('/').split('/')[-1]
        for u in url_set:
            if u.endswith('/' + slug):
                return u
        return n
    else:
        key = gstr.lower()
        if key in name_to_url:
            return name_to_url[key]
        # substring name match
        for nm,u in name_to_url.items():
            if key in nm or nm in key:
                return u
        return key

# build gold map Query -> list of mapped canonical urls (or names)
gold_map = defaultdict(list)
for _, r in train_df.iterrows():
    q = str(r["Query"]).strip()
    url_or_name = str(r["Assessment_url"]).strip()
    mapped = map_gold(url_or_name)
    gold_map[q].append(mapped)

# recommender instances
rec = Recommender()
baseline_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# load items list for index lookup if needed
def baseline_recommend(query, k=10):
    q_emb = baseline_encoder.encode([query], normalize_embeddings=True)[0]
    sims = (embs @ q_emb)
    idxs = np.argsort(-sims)[:k]
    out = []
    for i in idxs:
        it = items[i]
        out.append({"assessment_name": it.get("name"), "url": norm_url(it.get("url")), "test_type": it.get("test_type")})
    return out

def improved_recommend(query, k=10):
    preds = rec.recommend(query=query, k=k)
    # normalize their URLs if present
    out=[]
    for p in preds:
        u = p.get("url")
        out.append({"assessment_name": p.get("assessment_name"), "url": norm_url(u) if isinstance(u,str) else u, "test_type": p.get("test_type")})
    return out

def mean_recall_at_k(pred_map, gold_map, k=10):
    recalls=[]
    for q, golds in gold_map.items():
        # golds are already mapped by map_gold() above (canonicalized)
        preds = [u for u in pred_map.get(q, [])][:k]
        # if gold stored as non-url name (unlikely here), normalize via norm_any
        hits = 0
        for g in golds:
            if isinstance(g,str) and g.startswith("http"):
                if g in preds:
                    hits+=1
            else:
                # compare normalized text to assessment names (predictions include names)
                for p in preds:
                    if norm_any(g) == norm_any(p):
                        hits+=1
                        break
        recalls.append(hits / max(1, len(golds)))
    return float(np.mean(recalls)) if recalls else 0.0

# Evaluate
queries = list(gold_map.keys())
preds_base = {}
preds_imp = {}
print("Running normalized evaluation on", len(queries), "queries...")
for q in queries:
    preds_base[q] = [p["url"] for p in baseline_recommend(q, k=10)]
    preds_imp[q] = [p["url"] for p in improved_recommend(q, k=10)]

mr_base = mean_recall_at_k(preds_base, gold_map, k=10)
mr_imp = mean_recall_at_k(preds_imp, gold_map, k=10)

print("Mean Recall@10 (Baseline - normalized):", round(mr_base,4))
print("Mean Recall@10 (Improved - normalized):", round(mr_imp,4))

# save results per-query
eval_record = {"baseline_mr10": mr_base, "improved_mr10": mr_imp, "per_query": []}
for q in queries:
    eval_record["per_query"].append({
        "query": q,
        "gold_mapped": gold_map[q],
        "baseline_preds": preds_base[q],
        "improved_preds": preds_imp[q]
    })
with open("src/eval_results.json", "w", encoding="utf-8") as f:
    json.dump(eval_record, f, ensure_ascii=False, indent=2)

# write submission CSVs for Test-Set (normalized)
test_queries = [str(x).strip() for x in test_df["Query"].tolist()]
rows_base=[]; rows_imp=[]
for q in test_queries:
    for p in baseline_recommend(q, k=10):
        rows_base.append({"Query": q, "Assessment_url": p["url"]})
    for p in improved_recommend(q, k=10):
        rows_imp.append({"Query": q, "Assessment_url": p["url"]})

pd.DataFrame(rows_base, columns=["Query","Assessment_url"]).to_csv("src/submission_baseline.csv", index=False)
pd.DataFrame(rows_imp, columns=["Query","Assessment_url"]).to_csv("src/submission_improved.csv", index=False)

print("Wrote normalized submissions and src/eval_results.json")
