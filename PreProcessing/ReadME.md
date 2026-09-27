Preprocessing

First preprocessing stage for the hypothesis-to-conclusion project.

Goal

Convert scraped scientific-paper CSV data into a model-ready paired dataset:

X_hypothesis: cleaned hypothesis / research objective / proposed-method text

Y_conclusion: cleaned conclusion text

The cleaner is intentionally conservative because the downstream models are semantic models. It preserves case, stopwords, negation, numbers, percentages, and mathematical text while removing extraction noise.

Expected CSV columns

Preferred:

id

title

primary_category

hypothesis

conclusion

The current scraper sample does not yet contain a hypothesis column. By default, such rows are rejected so that the model dataset does not silently treat an approximate extraction as ground truth. For prototyping only, the optional --extract-hypothesis-if-missing flag can search introduction and abstract for hypothesis/objective/proposed-method cues such as we hypothesize, we investigate, we propose, or we introduce.

A real conclusion section is required. The script intentionally does not use the abstract as a substitute conclusion target.

Run

From the repository root:

python Preprocessing/PP.py arxiv_papers.csv

Explicit output directory:

python Preprocessing/PP.py arxiv_papers.csv \
  --output-dir Preprocessing/output

For a prototype run on a file that does not yet have a dedicated hypothesis column:

python Preprocessing/preprocess_hypothesis_conclusion.py arxiv_papers.csv \
  --extract-hypothesis-if-missing

Do not use fallback-extracted hypotheses as a human-verified gold test set.

Outputs

Preprocessing/output/hypothesis_conclusion_clean.csv

Model-ready accepted rows.

Preprocessing/output/hypothesis_conclusion_rejected.csv

Rows rejected because of missing/short hypotheses or conclusions. Keep this file for auditing rather than silently discarding papers.

Preprocessing/output/preprocessing_report.json

Counts and basic quality statistics for the run.

Current cleaning operations

Missing-value normalization.

Unicode normalization and PDF ligature repair.

PDF line-break dehyphenation.

arXiv header and isolated page-number removal.

Citation normalization to [CITATION].

URL/email normalization.

Light LaTeX-wrapper cleanup while retaining useful text.

Section-heading removal from extracted hypothesis/conclusion sections.

Whitespace normalization.

Minimum-length validation.

Exact normalized X/Y pair deduplication.

Provenance fields and rejection logging.

Intentionally NOT done

lowercasing all text

stopword removal

stemming/lemmatization

deletion of negation

deletion of experimental numbers/percentages

abstract-as-conclusion substitution

Those transformations can damage the scientific relationship we want BGE-M3 and Qwen to learn.