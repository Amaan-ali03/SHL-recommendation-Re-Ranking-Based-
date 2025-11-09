"""
Pagination-aware SHL crawler.
- Starts at /products/product-catalog/
- Extracts product links and pagination links, follows all pages found.
- Saves src/index/catalog_individual.json
Run: python -m src.crawl_shl_catalog
"""
import time, json, os, re
from urllib.parse import urljoin, urlparse, urlunparse
import requests
from bs4 import BeautifulSoup
from src.config import CATALOG_URL, BASE

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SHL-Catalog-Crawler/1.2)"}

def fetch(url):
    for _ in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.text
            else:
                print("HTTP", r.status_code, "for", url)
        except Exception as e:
            print("fetch error:", e, "for", url)
        time.sleep(1.0)
    return ""

def normalize_url(u, base=BASE):
    if not u:
        return None
    if u.startswith("http"):
        return u
    return urljoin(base, u)

def extract_product_links(html, base=BASE):
    soup = BeautifulSoup(html, "lxml")
    links=set()
    # find anchors that include '/product-catalog/view/'
    for a in soup.find_all("a", href=True):
        href = a['href']
        if "/product-catalog/view/" in href:
            links.add(normalize_url(href, base))
    # also regex fallback
    for m in re.finditer(r'https?://[^\s"\']+product-catalog/view/[^\s"\']+', html):
        links.add(m.group(0))
    return list(links)

def extract_pagination_links(html, base=BASE):
    soup = BeautifulSoup(html, "lxml")
    pages = set()
    # common patterns: anchors with '?_page=' or '?page=' or '/?page='
    for a in soup.find_all("a", href=True):
        href = a['href']
        if re.search(r'(\?|&)(page|_page)=\d+', href, re.I):
            pages.add(normalize_url(href, base))
        # anchors that look like page numbers (href contains '/products/product-catalog/?p=...')
        if "/products/product-catalog" in href and re.search(r'\d+', a.get_text() or ""):
            pages.add(normalize_url(href, base))
    # also look for rel="next"
    next_tag = soup.find("link", rel="next")
    if next_tag and next_tag.get("href"):
        pages.add(normalize_url(next_tag.get("href"), base))
    return list(pages)

def parse_detail_page(html, url):
    soup = BeautifulSoup(html, "lxml")

    # remove scripts/styles
    for s in soup(["script", "style", "noscript"]):
        s.decompose()

    # try meta description first
    meta_desc = None
    md = soup.find("meta", attrs={"name":"description"})
    if md and md.get("content"):
        meta_desc = md.get("content").strip()

    # try common product containers
    candidates = []
    for sel in ['article', 'main', 'div[class*="product"]', 'div[class*="description"]', 'section']:
        for tag in soup.select(sel):
            txt = tag.get_text(" ", strip=True)
            if txt and len(txt) > 80:
                candidates.append(txt)

    # fallback to longest text node
    if not candidates:
        whole = soup.get_text("\n", strip=True)
        # remove obvious UI boilerplate
        whole = re.sub(r'Outdated browser detected.*?Latest browser options', ' ', whole, flags=re.S|re.I)
        candidates = [whole]

    # pick the best candidate (longest meaningful)
    desc = max(candidates, key=lambda t: len(t)) if candidates else None
    if desc:
        # clean repetitive whitespace & common UI fragments
        desc = re.sub(r'\s{2,}', ' ', desc).strip()
        desc = re.sub(r'Outdated browser detected.*?Latest browser options', ' ', desc, flags=re.S|re.I)
        desc = re.sub(r'Global Offices.*', '', desc, flags=re.S|re.I)
        desc = desc.strip()

    # length
    whole = soup.get_text(" ", strip=True)
    m = re.search(r"(\d{1,3})\s*min", whole, re.I)
    length = int(m.group(1)) if m else None

    # test type (same as before)
    test_type = None
    m2 = re.search(r"Test Type[:\s]*([A-Z])", whole, re.I)
    if m2:
        test_type = m2.group(1).upper()

    # languages (try to find list-like)
    langs = []
    for hdr in soup.find_all(text=re.compile(r'Languages|Language', re.I)):
        parent = hdr.parent
        if parent:
            nxt = parent.find_next()
            if nxt:
                langs = [x.strip() for x in nxt.get_text(" ").split(",") if x.strip()]
                break

    return {
        "title": (soup.find(["h1","h2"]).get_text(" ",strip=True) if soup.find(["h1","h2"]) else None),
        "description": desc,
        "assessment_length_min": length,
        "test_type": test_type,
        "languages": langs
    }


def crawl(start_url=CATALOG_URL, max_pages=200):
    print("Starting pagination-aware crawl from:", start_url)
    seen_pages = set()
    to_visit = [start_url]
    product_urls = set()
    page_count=0

    while to_visit and page_count < max_pages:
        url = to_visit.pop(0)
        if url in seen_pages:
            continue
        print("Visiting catalog page:", url)
        seen_pages.add(url)
        page_count += 1
        html = fetch(url)
        if not html:
            print(" Empty html for", url)
            continue
        # extract product links
        found = extract_product_links(html)
        print(" Found product links:", len(found))
        for u in found:
            product_urls.add(u)
        # extract pagination links and enqueue new ones
        pagelinks = extract_pagination_links(html)
        new_pages = 0
        for p in pagelinks:
            if p not in seen_pages and p not in to_visit:
                to_visit.append(p); new_pages += 1
        print(" Found pagination links:", new_pages, "queue size:", len(to_visit))
        time.sleep(0.6)

    print("Finished scanning catalog pages. Unique product urls:", len(product_urls))
    # now fetch details for each product
    enriched=[]
    for i,u in enumerate(sorted(product_urls)):
        print(f"Fetching detail [{i+1}/{len(product_urls)}]:", u)
        html = fetch(u)
        if not html:
            print("  failed:", u); continue
        det = parse_detail_page(html, u)
        rec = {
            "name": det.get("title") or u.split("/")[-2] or u,
            "url": u,
            "description": det.get("description"),
            "assessment_length_min": det.get("assessment_length_min"),
            "test_type": det.get("test_type"),
            "languages": det.get("languages")
        }
        enriched.append(rec)
        if (i+1) % 20 == 0:
            print("  fetched", i+1, "details...")
        time.sleep(0.4)
    os.makedirs("src/index", exist_ok=True)
    with open("src/index/catalog_individual.json","w",encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    print("Saved", len(enriched), "items to src/index/catalog_individual.json")

if __name__ == "__main__":
    crawl()
