from src.recommender import Recommender

def main():
    r = Recommender()
    diag = r.recommend("python developer", k=10, debug=True)
    for i, c in enumerate(diag[:30]):
        print(
            f"{i+1}. {c.get('name')[:120]!s} | SIM: {round(c.get('_sim',0),4)} "
            f"| KW: {round(c.get('_kw',0),3)} | PRE: {round(c.get('_pre_score',0),4)} "
            f"| CE_norm: {c.get('_ce_norm', None)} | FINAL: {c.get('_final', None)} | URL: {c.get('url')}"
        )

if __name__ == "__main__":
    main()

