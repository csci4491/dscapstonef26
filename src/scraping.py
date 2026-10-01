import re
from io import BytesIO

import arxiv
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

def fetch_papers(query, max_results=5):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    return list(client.results(search))


def get_full_text_html(arxiv_id):
    """arxiv_id should be the base id WITHOUT version, e.g. '2401.12345'."""
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text("\n")
    except Exception as e:
        print(f"HTML fetch failed for {arxiv_id}: {e}")
        return ""


def get_full_text_pdf(pdf_url):
    try:
        resp = requests.get(pdf_url, timeout=60)
        resp.raise_for_status()
        reader = PdfReader(BytesIO(resp.content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"PDF fetch failed for {pdf_url}: {e}")
        return ""

SECTION_PATTERNS = [
    r"(?im)^\s*(?:\d+\.?\s*)?abstract\s*$",
    r"(?im)^\s*(?:\d+\.?\s*)?(?:introduction|background)\s*$",
    r"(?im)^\s*(?:\d+\.?\s*)?(?:related\s+work|literature\s+review)\s*$",
    r"(?im)^\s*(?:\d+\.?\s*)?(?:method(?:s|ology)?|approach|proposed\s+method)\s*$",
    r"(?im)^\s*(?:\d+\.?\s*)?(?:experiment(?:s|al)?(?:\s+setup)?|evaluation)\s*$",
    r"(?im)^\s*(?:\d+\.?\s*)?results?(?:\s+and\s+discussion)?\s*$",
    r"(?im)^\s*(?:\d+\.?\s*)?discussion\s*$",
    r"(?im)^\s*(?:\d+\.?\s*)?conclusion(?:s)?(?:\s+and\s+future\s+work)?\s*$",
    r"(?im)^\s*(?:\d+\.?\s*)?(?:references|bibliography)\s*$",
    r"(?im)^\s*(?:\d+\.?\s*)?(?:appendix|supplementary)\b.*$",
]

_SECTION_RE = [re.compile(p) for p in SECTION_PATTERNS]


def extract_sections(text):
    """Split paper text into a {section_key: body} dict."""
    if not text:
        return {}

    matches = []
    for pat in _SECTION_RE:
        for m in pat.finditer(text):
            matches.append((m.start(), m.group().strip()))

    if not matches:
        return {"full_text": text.strip()}

    matches.sort(key=lambda x: x[0])
    unique, last_end = [], -1
    for pos, title in matches:
        if pos > last_end:
            unique.append((pos, title))
            last_end = pos

    sections = {}
    for i, (pos, title) in enumerate(unique):
        end = unique[i + 1][0] if i + 1 < len(unique) else len(text)
        body = text[pos:end].strip()
        key = re.sub(r"^\d+\.?\s*", "", title, flags=re.I).strip().lower()
        key = re.sub(r"\s+", "_", key)
        if key and len(body) > 50:
            sections[key] = body
    return sections

HYPOTHESIS_SECTIONS = [
    r"^hypothes[ei]s(?:es)?$",
    r"^research_questions?$",
    r"^problem_statement$",
]
CONCLUSION_SECTIONS = [
    r"^conclusions?(?:_and_future_work)?$",
    r"^(?:summary|discussion|concluding_remarks)$",
]
ABSTRACT_SECTIONS = [r"^abstract$"]
INTRO_SECTIONS = [r"^(?:introduction|background)$"]


def pick_section(sections, patterns):
    """Match against normalized section keys (already lowercased, underscored)."""
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for key, body in sections.items():
            if rx.match(key):
                return key, body
    return None, ""

HYPOTHESIS_CUES = [
    r"\bwe\s+(?:hypothesi[sz]e|posit|conjecture|predict|expect)\b",
    r"\bour\s+(?:hypothes[ei]s|assumption|conjecture|prediction)\b",
    r"\bthe\s+(?:hypothes[ei]s|assumption)\s+(?:is|was|that)\b",
    r"\bwe\s+(?:aim|seek|set\s+out)\s+to\b",
    r"\bthis\s+(?:paper|work|study)\s+(?:investigates|examines|tests)\b",
    r"\bwe\s+(?:test|investigate|examine)\s+(?:whether|if)\b",
]
CONCLUSION_CUES = [
    r"\bwe\s+(?:conclude|find|show|demonstrate|report)\s+that\b",
    r"\bour\s+(?:results|findings|experiments)\s+(?:show|suggest|indicate|demonstrate)\b",
    r"\b(?:in|to)\s+conclusion\b",
    r"\bthese\s+results\s+(?:suggest|indicate|imply)\b",
    r"\boverall,?\s+we\b",
    r"\btaken\s+together\b",
    r"\bwe\s+(?:therefore|thus)\b",
]
HYP_RE = re.compile("|".join(HYPOTHESIS_CUES), re.I)
CONC_RE = re.compile("|".join(CONCLUSION_CUES), re.I)


def split_sentences(text):
    protected = re.sub(r"\b(e\.g|i\.e|et\s+al|vs|Fig|Eq|No|Dr|Mr|Ms|Prof)\.", r"\1<DOT>", text)
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def extract_by_cues(text, cue_re, max_sentences=8):
    sents = split_sentences(text)
    hits = []
    for i, s in enumerate(sents):
        if cue_re.search(s):
            start = i - 1 if i > 0 and len(sents[i - 1]) < 200 else i
            hits.append(" ".join(sents[start:i + 1]))
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out[:max_sentences]


def get_hypothesis(sections, full_text):
    key, body = pick_section(sections, HYPOTHESIS_SECTIONS)
    if body and len(body) > 100:
        return body, "section"

    source = ""
    for pats in (ABSTRACT_SECTIONS, INTRO_SECTIONS):
        _, b = pick_section(sections, pats)
        if b:
            source += b + "\n"
    if not source:
        source = full_text[:5000]

    hits = extract_by_cues(source, HYP_RE)
    return " ".join(hits), "cues"


def get_conclusion(sections, full_text):
key, body = pick_section(sections, CONCLUSION_SECTIONS)
    if body:
        return body.strip()
    return ""


def extract_hypothesis_and_conclusion(paper_row):
    sections = paper_row["sections"]
    full_text = "\n\n".join(sections.values()) if sections else paper_row.get("full_text", "")

    hyp, hsrc = get_hypothesis(sections, full_text)
    conc, csrc = get_conclusion(sections, full_text)

    return {
        "arxiv_id": paper_row["arxiv_id"],
        "title": paper_row["title"],
        "hypothesis": hyp.strip(),
        "conclusion": conc.strip(),
        "hypothesis_source": hsrc,
        "conclusion_source": csrc,
    }


def build_dataframe(query, max_results=5):
    papers = fetch_papers(query, max_results)
    rows = []

    for p in papers:
        versioned = p.entry_id.rsplit("/", 1)[-1]         
        base_id = re.sub(r"v\d+$", "", versioned)         

        full_text = get_full_text_html(base_id)
        source = "html"
        if len(full_text) < 500:
            full_text = get_full_text_pdf(p.pdf_url)
            source = "pdf"

        sections = extract_sections(full_text)

        rows.append({
            "arxiv_id": versioned,
            "title": p.title,
            "authors": ", ".join(a.name for a in p.authors),
            "published": p.published.isoformat(),
            "summary": p.summary,
            "primary_category": p.primary_category,
            "pdf_url": p.pdf_url,
            "source": source,
            "num_sections": len(sections),
            "sections": sections,
            "full_text": full_text,
        })
        print(f"[{source}] {versioned} — {p.title[:70]} ({len(sections)} sections)")

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = build_dataframe("artificial intelligence", max_results=200)

    if df.empty:
        print("No papers fetched.")
        raise SystemExit

    records = [extract_hypothesis_and_conclusion(row) for _, row in df.iterrows()]
    extracted = pd.DataFrame(records)

    extracted = extracted[extracted["hypothesis"].str.strip().astype(bool)].reset_index(drop=True)

    pd.set_option("display.max_colwidth", 200)
    print(extracted[["arxiv_id", "hypothesis_source"]].head())
    if not extracted.empty:
        print("HYPOTHESIS:", extracted.iloc[0]["hypothesis"][:400])
        print("\nCONCLUSION:", extracted.iloc[0]["conclusion"][:400])

    df.drop(columns=["sections", "full_text"]).to_csv("arxiv_papers.csv", index=False)
    extracted.to_csv("arxiv_extracted_final.csv", index=False)
