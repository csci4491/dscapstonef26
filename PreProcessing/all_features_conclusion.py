"""
The target remains:
    Y_conclusion

The script preserves separate cleaned input features so later experiments can measure
how much additional paper context improves over a hypothesis-only baseline:
    X_hypothesis
    X_abstract
    X_introduction
    X_methods
    X_results
    X_discussion
    X_related_work

It also creates controlled combined inputs:
    X_hypothesis_methods
    X_hypothesis_methods_results

The combined columns intentionally do NOT include abstract or discussion by default.
Abstracts often summarize the final result and discussions often restate conclusions,
so including them automatically could make the task artificially easy (target leakage).
They are still preserved as separate features for explicit experiments.

This script reuses the conservative scientific-text cleaning and optional hypothesis
fallback from PP.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from PP import (
    clean_scientific_text,
    extract_hypothesis_candidate,
    is_missing,
    normalize_for_duplicate_check,
    word_count,
)


# Canonical feature -> likely source-column names across scraper versions.
FEATURE_ALIASES: dict[str, list[str]] = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "intro"],
    "methods": [
        "methods",
        "method",
        "methodology",
        "materials_and_methods",
        "materials and methods",
        "approach",
        "proposed_method",
    ],
    "results": ["results", "result", "experimental_results", "experimental results", "findings"],
    "discussion": ["discussion", "discussions"],
    "related_work": ["related_work", "related work", "background"],
}


def first_existing_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    """Return the first matching column name, using case-insensitive matching."""
    lower_to_original = {str(col).lower(): str(col) for col in df.columns}
    for alias in aliases:
        match = lower_to_original.get(alias.lower())
        if match is not None:
            return match
    return None


def find_paper_id(row: pd.Series, row_idx: int) -> object:
    """Use the best available paper identifier across known scraper schemas."""
    for key in ("arxiv_id", "id", "paper_id"):
        if key in row.index and not is_missing(row.get(key)):
            return row.get(key)
    return row_idx


def clean_optional_feature(value: object, min_words: int) -> str:
    """Clean a non-required feature; discard tiny/noisy fragments without rejecting the row."""
    cleaned = clean_scientific_text(value, remove_section_heading=True)
    if cleaned and word_count(cleaned) < min_words:
        return ""
    return cleaned


def combine_features(*parts: str) -> str:
    """Combine non-empty sections with explicit markers for downstream models."""
    labels = [
        "HYPOTHESIS",
        "METHODS",
        "RESULTS",
        "SECTION_4",
        "SECTION_5",
    ]
    blocks: list[str] = []
    for idx, part in enumerate(parts):
        if part:
            label = labels[idx] if idx < len(labels) else f"SECTION_{idx + 1}"
            blocks.append(f"[{label}] {part}")
    return "\n\n".join(blocks)


def build_multifeature_dataset(
    df: pd.DataFrame,
    hypothesis_col: str,
    conclusion_col: str,
    min_hypothesis_words: int,
    min_conclusion_words: int,
    min_optional_words: int,
    allow_hypothesis_fallback: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Create one clean row per valid hypothesis-conclusion pair plus optional X features."""

    if conclusion_col not in df.columns:
        raise ValueError(
            f"Required conclusion column '{conclusion_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    direct_hypothesis_available = hypothesis_col in df.columns
    resolved_feature_columns = {
        feature: first_existing_column(df, aliases)
        for feature, aliases in FEATURE_ALIASES.items()
    }

    records: list[dict] = []
    rejected: list[dict] = []

    for row_idx, row in df.iterrows():
        paper_id = find_paper_id(row, int(row_idx))
        title = clean_scientific_text(row.get("title", ""))
        category = clean_scientific_text(row.get("primary_category", row.get("category", "")))

        # --- Core X: hypothesis ---
        hypothesis_source = hypothesis_col if direct_hypothesis_available else ""
        extraction_score: Optional[int] = None

        if direct_hypothesis_available and not is_missing(row.get(hypothesis_col)):
            x_hypothesis = clean_scientific_text(
                row.get(hypothesis_col), remove_section_heading=True
            )
            hypothesis_source = hypothesis_col
        elif allow_hypothesis_fallback:
            intro_col = resolved_feature_columns.get("introduction")
            abstract_col = resolved_feature_columns.get("abstract")
            x_hypothesis, fallback_source, extraction_score = extract_hypothesis_candidate(
                row.get(intro_col, "") if intro_col else "",
                row.get(abstract_col, "") if abstract_col else "",
            )
            hypothesis_source = f"extracted_from_{fallback_source}" if x_hypothesis else "not_found"
        else:
            x_hypothesis = ""

        # --- Target Y: conclusion ---
        y_conclusion = clean_scientific_text(
            row.get(conclusion_col), remove_section_heading=True
        )

        hypothesis_words = word_count(x_hypothesis)
        conclusion_words = word_count(y_conclusion)

        reasons: list[str] = []
        if not x_hypothesis:
            reasons.append("missing_hypothesis")
        elif hypothesis_words < min_hypothesis_words:
            reasons.append("hypothesis_too_short")

        if not y_conclusion:
            reasons.append("missing_conclusion")
        elif conclusion_words < min_conclusion_words:
            reasons.append("conclusion_too_short")

        if reasons:
            rejected.append(
                {
                    "source_row": int(row_idx),
                    "paper_id": paper_id,
                    "title": title,
                    "reason": ";".join(reasons),
                    "hypothesis_source": hypothesis_source or "not_found",
                    "hypothesis_words": hypothesis_words,
                    "conclusion_words": conclusion_words,
                }
            )
            continue

        # --- Optional X features ---
        optional: dict[str, str] = {}
        optional_words: dict[str, int] = {}
        for feature, source_col in resolved_feature_columns.items():
            cleaned = ""
            if source_col is not None:
                cleaned = clean_optional_feature(row.get(source_col, ""), min_optional_words)
            optional[feature] = cleaned
            optional_words[feature] = word_count(cleaned)

        # Controlled combined inputs. These support ablation-style experiments:
        # H only vs H+M vs H+M+R.
        x_hypothesis_methods = combine_features(
            x_hypothesis,
            optional["methods"],
        )
        x_hypothesis_methods_results = combine_features(
            x_hypothesis,
            optional["methods"],
            optional["results"],
        )

        available_optional = [
            name for name, text in optional.items() if text
        ]

        records.append(
            {
                "source_row": int(row_idx),
                "paper_id": paper_id,
                "title": title,
                "primary_category": category,
                "X_hypothesis": x_hypothesis,
                "X_abstract": optional["abstract"],
                "X_introduction": optional["introduction"],
                "X_methods": optional["methods"],
                "X_results": optional["results"],
                "X_discussion": optional["discussion"],
                "X_related_work": optional["related_work"],
                "X_hypothesis_methods": x_hypothesis_methods,
                "X_hypothesis_methods_results": x_hypothesis_methods_results,
                "Y_conclusion": y_conclusion,
                "hypothesis_source": hypothesis_source,
                "hypothesis_extraction_score": extraction_score,
                "hypothesis_words": hypothesis_words,
                "abstract_words": optional_words["abstract"],
                "introduction_words": optional_words["introduction"],
                "methods_words": optional_words["methods"],
                "results_words": optional_words["results"],
                "discussion_words": optional_words["discussion"],
                "related_work_words": optional_words["related_work"],
                "conclusion_words": conclusion_words,
                "available_optional_features": ",".join(available_optional),
                "optional_feature_count": len(available_optional),
            }
        )

    clean_df = pd.DataFrame(records)
    rejected_df = pd.DataFrame(rejected)

    # Deduplicate based on the core scientific pair, not on optional context.
    duplicates_removed = 0
    if not clean_df.empty:
        clean_df["_x_norm"] = clean_df["X_hypothesis"].map(normalize_for_duplicate_check)
        clean_df["_y_norm"] = clean_df["Y_conclusion"].map(normalize_for_duplicate_check)
        before = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["_x_norm", "_y_norm"], keep="first").copy()
        duplicates_removed = before - len(clean_df)
        clean_df = clean_df.drop(columns=["_x_norm", "_y_norm"])

    report: dict = {
        "input_rows": int(len(df)),
        "output_rows": int(len(clean_df)),
        "rejected_rows": int(len(rejected_df)),
        "duplicate_pairs_removed": int(duplicates_removed),
        "hypothesis_column_present": bool(direct_hypothesis_available),
        "hypothesis_fallback_enabled": bool(allow_hypothesis_fallback),
        "min_hypothesis_words": int(min_hypothesis_words),
        "min_conclusion_words": int(min_conclusion_words),
        "min_optional_words": int(min_optional_words),
        "resolved_feature_columns": resolved_feature_columns,
    }

    if not clean_df.empty:
        report["mean_word_counts"] = {
            "hypothesis": round(float(clean_df["hypothesis_words"].mean()), 2),
            "abstract": round(float(clean_df["abstract_words"].mean()), 2),
            "introduction": round(float(clean_df["introduction_words"].mean()), 2),
            "methods": round(float(clean_df["methods_words"].mean()), 2),
            "results": round(float(clean_df["results_words"].mean()), 2),
            "discussion": round(float(clean_df["discussion_words"].mean()), 2),
            "related_work": round(float(clean_df["related_work_words"].mean()), 2),
            "conclusion": round(float(clean_df["conclusion_words"].mean()), 2),
        }
        report["feature_coverage"] = {
            feature: {
                "rows_present": int((clean_df[f"{feature}_words"] > 0).sum()),
                "percentage": round(float((clean_df[f"{feature}_words"] > 0).mean() * 100), 2),
            }
            for feature in ["abstract", "introduction", "methods", "results", "discussion", "related_work"]
        }
        report["hypothesis_sources"] = {
            str(k): int(v)
            for k, v in clean_df["hypothesis_source"].value_counts(dropna=False).items()
        }

    if not rejected_df.empty:
        reason_counts: dict[str, int] = {}
        for reason_string in rejected_df["reason"]:
            for reason in reason_string.split(";"):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        report["rejection_reasons"] = reason_counts

    return clean_df, rejected_df, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean scientific-paper CSV data into hypothesis/conclusion pairs plus "
            "additional model input features such as methods and results."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Path to scraped-paper CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("PreProcessing/output_multifeature"),
        help="Directory for output CSVs and report",
    )
    parser.add_argument("--hypothesis-col", default="hypothesis")
    parser.add_argument("--conclusion-col", default="conclusion")
    parser.add_argument("--min-hypothesis-words", type=int, default=8)
    parser.add_argument("--min-conclusion-words", type=int, default=15)
    parser.add_argument(
        "--min-optional-words",
        type=int,
        default=5,
        help="Optional sections shorter than this are blanked but do not cause row rejection",
    )
    parser.add_argument(
        "--extract-hypothesis-if-missing",
        action="store_true",
        help=(
            "If hypothesis is absent, derive a hypothesis/objective candidate from "
            "introduction and then abstract using the existing fallback extractor."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    clean_df, rejected_df, report = build_multifeature_dataset(
        df=df,
        hypothesis_col=args.hypothesis_col,
        conclusion_col=args.conclusion_col,
        min_hypothesis_words=args.min_hypothesis_words,
        min_conclusion_words=args.min_conclusion_words,
        min_optional_words=args.min_optional_words,
        allow_hypothesis_fallback=args.extract_hypothesis_if_missing,
    )

    clean_path = args.output_dir / "multifeature_clean.csv"
    rejected_path = args.output_dir / "multifeature_rejected.csv"
    report_path = args.output_dir / "multifeature_report.json"

    clean_df.to_csv(clean_path, index=False)
    rejected_df.to_csv(rejected_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nClean dataset: {clean_path}")
    print(f"Rejected rows: {rejected_path}")
    print(f"Report:        {report_path}")


if __name__ == "__main__":
    main()
