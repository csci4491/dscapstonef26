"""Preprocess scraped scientific-paper text into Hypothesis (X) and Conclusion (Y).

This script is intentionally conservative: it removes formatting/extraction noise while
preserving scientific semantics such as negation, case, percentages, and mathematical text.

Expected input:
- CSV containing a conclusion column.
- Preferably a hypothesis column.
- If hypothesis is absent, the script can extract a hypothesis/objective/proposed-method
  candidate from introduction and then abstract using cue phrases.

Outputs:
- cleaned XY CSV
- rejected rows CSV with reasons
- JSON preprocessing report
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd


HYPOTHESIS_CUES = [
    r"\bwe\s+hypothesi[sz]e\b",
    r"\bour\s+hypothesis\b",
    r"\bwe\s+expect\b",
    r"\bwe\s+predict\b",
    r"\bwe\s+conjecture\b",
    r"\bwe\s+investigate\b",
    r"\bwe\s+examine\b",
    r"\bwe\s+study\b",
    r"\bwe\s+ask\b",
    r"\bwe\s+test\b",
    r"\bwe\s+evaluate\b",
    r"\bwe\s+aim\b",
    r"\bour\s+aim\b",
    r"\bour\s+objective\b",
    r"\bthe\s+goal\s+of\s+this\s+(?:paper|work|study)\b",
    r"\bthe\s+objective\s+of\s+this\s+(?:paper|work|study)\b",
    r"\bthis\s+(?:paper|work|study)\s+(?:investigates|examines|studies|tests|evaluates|explores)\b",
    r"\bwe\s+propose\b",
    r"\bwe\s+introduce\b",
    r"\bwe\s+present\b",
    r"\bwe\s+develop\b",
]
HYPOTHESIS_CUE_RE = re.compile("|".join(HYPOTHESIS_CUES), flags=re.IGNORECASE)


SECTION_PREFIX_RE = re.compile(
    r"^\s*(?:section\s+)?(?:\d+(?:\.\d+)*\.?\s*)?"
    r"(?:conclusions?|concluding remarks|summary(?: and conclusions?)?|"
    r"discussion(?: and conclusions?)?|introduction|methods?|methodology|"
    r"experimental results?|results(?: and discussion)?|future work)\s*[:\-–—]?\s*",
    flags=re.IGNORECASE,
)

# PDF extraction often leaves these as stand-alone lines.
ARXIV_HEADER_RE = re.compile(r"\barXiv:\s*\S+[^\n]*", flags=re.IGNORECASE)
PAGE_NUMBER_LINE_RE = re.compile(r"(?m)^\s*(?:page\s+)?\d{1,4}\s*$", flags=re.IGNORECASE)

# [1], [2, 3], [4-7], [2–5, 8]
BRACKET_CITATION_RE = re.compile(r"\[(?:\s*\d+\s*(?:[-–—]\s*\d+)?\s*,?\s*)+\]")

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE)

# Conservative sentence splitter. Avoids requiring NLTK/spaCy for the first pipeline stage.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")


def is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "n/a", "na"}


def normalize_unicode(text: str) -> str:
    """Normalize Unicode and common PDF ligatures without changing meaning."""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "\u00a0": " ",  # non-breaking space
        "\u200b": "",   # zero-width space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_scientific_text(value: object, remove_section_heading: bool = False) -> str:
    """Clean extraction artifacts while preserving semantic scientific content.

    Intentionally retained:
    - capitalization
    - stopwords
    - negation (not/no/without)
    - numbers, percentages, metric values
    - mathematical variable text
    """
    if is_missing(value):
        return ""

    text = normalize_unicode(str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Repair words split across a PDF line break: "classifi-\ncation" -> "classification".
    # Only dehyphenate when both sides look alphabetic.
    text = re.sub(r"(?<=[A-Za-z])[-‐]\s*\n\s*(?=[A-Za-z])", "", text)

    # Remove common extraction/header noise.
    text = ARXIV_HEADER_RE.sub(" ", text)
    text = PAGE_NUMBER_LINE_RE.sub(" ", text)

    # Normalize citations rather than deleting the fact that a citation existed.
    text = BRACKET_CITATION_RE.sub(" [CITATION] ", text)

    # Normalize web/contact artifacts that are rarely semantically useful here.
    text = URL_RE.sub(" [URL] ", text)
    text = EMAIL_RE.sub(" [EMAIL] ", text)

    # Remove LaTeX wrappers while preserving their content where possible.
    text = re.sub(r"\\(?:textbf|textit|emph|mathrm|mathbf|mathit)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:label|ref|eqref|cite|citep|citet)\s*\{[^{}]*\}", " [CITATION] ", text)
    text = re.sub(r"\\(?:section|subsection|subsubsection)\*?\s*\{([^{}]*)\}", r" \1 ", text)

    # Remove math delimiters only; keep the expression itself.
    text = text.replace("\\(", " ").replace("\\)", " ")
    text = text.replace("\\[", " ").replace("\\]", " ")
    text = text.replace("$", "")

    if remove_section_heading:
        text = SECTION_PREFIX_RE.sub("", text, count=1)

    # Collapse whitespace after all line-aware operations.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def candidate_score(sentence: str) -> int:
    """Score likely hypothesis/objective/proposed-method sentences."""
    s = sentence.lower()
    score = 0
    if HYPOTHESIS_CUE_RE.search(sentence):
        score += 5
    if any(term in s for term in ["whether", "effect of", "impact of", "outperform", "improve", "increase", "decrease", "relationship between"]):
        score += 2
    if any(term in s for term in ["we propose", "we introduce", "our approach", "our method", "we develop"]):
        score += 2
    # Penalize sentences that mainly describe paper organization.
    if any(term in s for term in ["the rest of this paper", "paper is organized", "section ii", "section 2"]):
        score -= 5
    return score


def extract_hypothesis_candidate(
    introduction: object,
    abstract: object,
    context_before: int = 1,
    context_after: int = 1,
) -> Tuple[str, str, int]:
    """Extract a hypothesis/objective/proposed-method candidate.

    Search introduction first, then abstract. We return a local sentence window around
    the strongest cue sentence, plus source and score. If no credible cue is found,
    return empty rather than inventing a hypothesis.
    """
    for source_name, source_value in (("introduction", introduction), ("abstract", abstract)):
        cleaned = clean_scientific_text(source_value, remove_section_heading=(source_name == "introduction"))
        sentences = split_sentences(cleaned)
        if not sentences:
            continue

        scored = [(candidate_score(sentence), idx) for idx, sentence in enumerate(sentences)]
        best_score, best_idx = max(scored, default=(0, -1))
        if best_score < 5:
            continue

        start = max(0, best_idx - context_before)
        end = min(len(sentences), best_idx + context_after + 1)
        candidate = " ".join(sentences[start:end])
        return candidate, source_name, best_score

    return "", "not_found", 0


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def normalize_for_duplicate_check(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[citation\]|\[url\]|\[email\]", " ", text)
    text = re.sub(r"\W+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def build_xy_dataset(
    df: pd.DataFrame,
    hypothesis_col: str,
    conclusion_col: str,
    min_hypothesis_words: int,
    min_conclusion_words: int,
    allow_hypothesis_fallback: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if conclusion_col not in df.columns:
        raise ValueError(
            f"Required conclusion column '{conclusion_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    direct_hypothesis_available = hypothesis_col in df.columns
    records = []
    rejected = []

    for row_idx, row in df.iterrows():
        paper_id = row.get("id", row_idx)
        title = clean_scientific_text(row.get("title", ""))
        category = clean_scientific_text(row.get("primary_category", ""))

        hypothesis_source = hypothesis_col if direct_hypothesis_available else ""
        extraction_score: Optional[int] = None

        if direct_hypothesis_available and not is_missing(row.get(hypothesis_col)):
            x = clean_scientific_text(row.get(hypothesis_col), remove_section_heading=True)
            hypothesis_source = hypothesis_col
        elif allow_hypothesis_fallback:
            x, hypothesis_source, extraction_score = extract_hypothesis_candidate(
                row.get("introduction", ""), row.get("abstract", "")
            )
        else:
            x = ""

        y = clean_scientific_text(row.get(conclusion_col), remove_section_heading=True)

        x_words = word_count(x)
        y_words = word_count(y)

        reasons = []
        if not x:
            reasons.append("missing_hypothesis")
        elif x_words < min_hypothesis_words:
            reasons.append("hypothesis_too_short")

        if not y:
            reasons.append("missing_conclusion")
        elif y_words < min_conclusion_words:
            reasons.append("conclusion_too_short")

        if reasons:
            rejected.append(
                {
                    "source_row": int(row_idx),
                    "paper_id": paper_id,
                    "title": title,
                    "reason": ";".join(reasons),
                    "hypothesis_source": hypothesis_source or "not_found",
                    "hypothesis_words": x_words,
                    "conclusion_words": y_words,
                }
            )
            continue

        records.append(
            {
                "source_row": int(row_idx),
                "paper_id": paper_id,
                "title": title,
                "primary_category": category,
                "X_hypothesis": x,
                "Y_conclusion": y,
                "hypothesis_source": hypothesis_source,
                "hypothesis_extraction_score": extraction_score,
                "hypothesis_words": x_words,
                "conclusion_words": y_words,
            }
        )

    clean_df = pd.DataFrame(records)
    rejected_df = pd.DataFrame(rejected)

    duplicates_removed = 0
    if not clean_df.empty:
        clean_df["_x_norm"] = clean_df["X_hypothesis"].map(normalize_for_duplicate_check)
        clean_df["_y_norm"] = clean_df["Y_conclusion"].map(normalize_for_duplicate_check)
        before = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["_x_norm", "_y_norm"], keep="first").copy()
        duplicates_removed = before - len(clean_df)
        clean_df = clean_df.drop(columns=["_x_norm", "_y_norm"])

    report = {
        "input_rows": int(len(df)),
        "output_rows": int(len(clean_df)),
        "rejected_rows": int(len(rejected_df)),
        "duplicate_pairs_removed": int(duplicates_removed),
        "hypothesis_column_present": bool(direct_hypothesis_available),
        "hypothesis_fallback_enabled": bool(allow_hypothesis_fallback),
        "min_hypothesis_words": int(min_hypothesis_words),
        "min_conclusion_words": int(min_conclusion_words),
    }

    if not clean_df.empty:
        report.update(
            {
                "mean_hypothesis_words": round(float(clean_df["hypothesis_words"].mean()), 2),
                "mean_conclusion_words": round(float(clean_df["conclusion_words"].mean()), 2),
                "hypothesis_sources": {
                    str(k): int(v)
                    for k, v in clean_df["hypothesis_source"].value_counts(dropna=False).items()
                },
            }
        )

    if not rejected_df.empty:
        reason_counts = {}
        for reason_string in rejected_df["reason"]:
            for reason in reason_string.split(";"):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        report["rejection_reasons"] = reason_counts

    return clean_df, rejected_df, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean scraped paper text into Hypothesis (X) and Conclusion (Y)."
    )
    parser.add_argument("input_csv", type=Path, help="Path to scraped-paper CSV")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Preprocessing/output"), help="Directory for outputs"
    )
    parser.add_argument("--hypothesis-col", default="hypothesis", help="Hypothesis input column name")
    parser.add_argument("--conclusion-col", default="conclusion", help="Conclusion target column name")
    parser.add_argument("--min-hypothesis-words", type=int, default=8)
    parser.add_argument("--min-conclusion-words", type=int, default=15)
    parser.add_argument(
        "--extract-hypothesis-if-missing",
        action="store_true",
        help=(
            "Prototype-only fallback: derive a hypothesis/objective candidate from introduction/abstract "
            "when an explicit hypothesis field is absent. Leave this off for a strict ground-truth dataset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    clean_df, rejected_df, report = build_xy_dataset(
        df=df,
        hypothesis_col=args.hypothesis_col,
        conclusion_col=args.conclusion_col,
        min_hypothesis_words=args.min_hypothesis_words,
        min_conclusion_words=args.min_conclusion_words,
        allow_hypothesis_fallback=args.extract_hypothesis_if_missing,
    )

    clean_path = args.output_dir / "hypothesis_conclusion_clean.csv"
    rejected_path = args.output_dir / "hypothesis_conclusion_rejected.csv"
    report_path = args.output_dir / "preprocessing_report.json"

    clean_df.to_csv(clean_path, index=False)
    rejected_df.to_csv(rejected_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nClean dataset: {clean_path}")
    print(f"Rejected rows: {rejected_path}")
    print(f"Report:        {report_path}")


if __name__ == "__main__":
    main()
