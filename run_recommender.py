from src.recommender import Recommender
rec = Recommender()
print("Loaded items:", len(rec.items))
print("Sample recommendations:")
for q in [
  "Hiring a Java developer who collaborates with stakeholders",
  "Looking for a customer service representative with strong communication and teamwork skills",
  "Need someone with analytical and numerical reasoning ability"
]:
    print("\\nQuery:", q)
    res = rec.recommend(query=q, k=6)
    for i,r in enumerate(res,1):
        print(f" {i}. {r['assessment_name'][:80]} -- {r.get('test_type')} -- {r.get('url')}")
