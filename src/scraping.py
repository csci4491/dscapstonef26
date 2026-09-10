import arxiv, requests, re
from pypdf import PdfReader
from io import BytesIO
from bs4 import BeautifulSoup
import pandas as pd

def fetch_papers(query, max_results=5):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    return list(client.results(search))

def get_full_text_html(arxiv_id):
    url = f"https://arxiv.org/html/{arxiv_id}v1"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text("\n")
    except Exception:
        return ""

def get_full_text_pdf(pdf_url):
    try:
        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()
        reader = PdfReader(BytesIO(resp.content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"PDF fetch failed for {pdf_url}: {e}")
        return ""

SECTION_PATTERNS = [
    (r'(?im)^\s*(?:\d+\.?\s*)?(?:abstract|summary)\s*$', "Abstract"),
    (r'(?im)^\s*(?:\d+\.?\s*)?(?:introduction|background)\s*$', "Introduction"),
    (r'(?im)^\s*(?:\d+\.?\s*)?(?:related work|literature review|prior work)\s*$', "Related Work"),
    (r'(?im)^\s*(?:\d+\.?\s*)?(?:methods?|methodology|materials? and methods?|study design|experimental setup)\s*$', "Methods"),
    (r'(?im)^\s*(?:\d+\.?\s*)?(?:results?|findings|experiments?)\s*$', "Results"),
    (r'(?im)^\s*(?:\d+\.?\s*)?(?:discussion)\s*$', "Discussion"),
    (r'(?im)^\s*(?:\d+\.?\s*)?(?:conclusions?)\s*$', "Conclusion"),
    (r'(?im)^\s*(?:\d+\.?\s*)?(?:references?|bibliography)\s*$', "References"),
]

def split_sections(full_text):
    positions = []
    for pattern, name in SECTION_PATTERNS:
        for m in re.finditer(pattern, full_text):
            positions.append((m.start(), name))
    positions.sort(key=lambda x: x[0])

    sections = {name: "" for _, name in SECTION_PATTERNS}

    seen = set()
    for i, (pos, name) in enumerate(positions):
        if name in seen:
            continue
        seen.add(name)
        end = positions[i + 1][0] if i + 1 < len(positions) else len(full_text)
        sections[name] = full_text[pos:end].strip()

    body_parts = [sections[k] for k in
                  ["Related Work", "Methods", "Results", "Discussion"]
                  if sections[k]]
    sections["Body"] = "\n\n".join(body_parts).strip()

    return sections

def build_dataframe(query, max_results=5):
    papers = fetch_papers(query, max_results)
    rows = []

    for p in papers:
        arxiv_id = p.entry_id.split("/")[-1]

        full_text = get_full_text_html(arxiv_id)
        if not full_text:
            full_text = get_full_text_pdf(p.pdf_url)

        sections = split_sections(full_text) if full_text else {
            "Abstract": "", "Introduction": "", "Related Work": "",
            "Methods": "", "Results": "", "Discussion": "",
            "Conclusion": "", "References": "", "Body": ""
        }

        rows.append({
            "id": arxiv_id,
            "title": p.title,
            "authors": ", ".join(a.name for a in p.authors),
            "published": p.published,
            "primary_category": p.primary_category,
            "pdf_url": p.pdf_url,
            "abstract": p.summary.replace("\n", " ").strip(),
            "introduction": sections["Introduction"],
            "related_work": sections["Related Work"],
            "methods": sections["Methods"],
            "results": sections["Results"],
            "discussion": sections["Discussion"],
            "conclusion": sections["Conclusion"],
            "body": sections["Body"],
            "references": sections["References"],
            "full_text": full_text,
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = build_dataframe("multiagent systems", max_results=10)

    pd.set_option("display.max_colwidth", 200)
    print(df[["title", "conclusion"]].head())

    df.to_csv("arxiv_papers.csv", index=False)
    print(df.head())