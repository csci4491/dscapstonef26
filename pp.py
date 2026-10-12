"""Convenience root wrapper for PreProcessing.PP.

Allows importing directly as `import pp` or `from pp import build_xy_dataset`.
"""

from PreProcessing.PP import (
    build_xy_dataset,
    clean_scientific_text,
    extract_hypothesis_candidate,
    normalize_unicode,
    main,
)

__all__ = [
    "build_xy_dataset",
    "clean_scientific_text",
    "extract_hypothesis_candidate",
    "normalize_unicode",
    "main",
]

if __name__ == "__main__":
    main()
