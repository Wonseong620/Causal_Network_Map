from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
LOOKUP_FILE = BASE_DIR / "icio_sectors_with_isic_rev4_descriptions.csv"
DEFAULT_INPUT_FILES = [
    BASE_DIR / "iran_ar_v3.csv",
    BASE_DIR / "iran_ch_v3.csv",
    BASE_DIR / "iran_pe_v3.csv",
]

TEXT_COLUMNS = [
    "economic_subtopic",
    "title",
    "fact_eng",
    "opinion_eng",
    "fact",
    "opinion",
    "content",
]


def clean_whitespace(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_text(value: object) -> str:
    return clean_whitespace(value).replace("_", " ")


def load_lookup() -> pd.DataFrame:
    lookup = pd.read_csv(LOOKUP_FILE, dtype=str, encoding="utf-8-sig")
    for col in lookup.columns:
        lookup[col] = lookup[col].map(clean_whitespace)

    lookup["ISIC Rev.4"] = lookup["ISIC Rev.4"].str.replace(
        r"\s*,\s*", ", ", regex=True
    )
    lookup["V1"] = lookup["V1"].astype(int)
    lookup = lookup.sort_values("V1").reset_index(drop=True)

    expected = list(range(1, 51))
    actual = lookup["V1"].tolist()
    if actual != expected:
        raise ValueError(f"Lookup V1 must be 1..50 in order, got {actual}")

    lookup["sector_text"] = (
        lookup["Code"]
        + " "
        + lookup["Industry"]
        + " "
        + lookup["ISIC Rev.4"]
        + " "
        + lookup["ISIC Rev. 4 description"]
    ).map(clean_text)
    return lookup


def build_article_text(df: pd.DataFrame) -> pd.Series:
    existing = [col for col in TEXT_COLUMNS if col in df.columns]
    return (
        df[existing]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(clean_text)
    )


def output_path(path: Path) -> Path:
    return path.with_name(path.stem + "_with_io_matrix.csv")


def classify_file(path: Path, lookup: pd.DataFrame) -> None:
    print(f"Reading {path.name}")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    article_text = build_article_text(df)

    corpus = lookup["sector_text"].tolist() + article_text.tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=100_000,
    )
    matrix = vectorizer.fit_transform(corpus)
    sector_matrix = matrix[: len(lookup)]
    article_matrix = matrix[len(lookup) :]

    similarities = cosine_similarity(article_matrix, sector_matrix)
    row_sums = similarities.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] == 0

    probabilities = np.divide(
        similarities,
        row_sums,
        out=np.zeros_like(similarities),
        where=row_sums != 0,
    )
    probabilities[zero_rows, :] = 1 / len(lookup)

    prob_cols = [f"IO_V1_{v1}" for v1 in lookup["V1"].tolist()]
    prob_df = pd.DataFrame(probabilities, columns=prob_cols).round(8)
    out = pd.concat([df, prob_df], axis=1)

    outpath = output_path(path)
    print(f"Writing {outpath.name}")
    out.to_csv(outpath, index=False, encoding="utf-8-sig")
    print(
        f"{path.name}: rows={len(out):,}, "
        f"zero_similarity_uniform_rows={int(zero_rows.sum()):,}"
    )


def main() -> None:
    lookup = load_lookup()
    if len(sys.argv) > 1:
        input_files = [BASE_DIR / arg for arg in sys.argv[1:]]
    else:
        input_files = DEFAULT_INPUT_FILES

    for path in input_files:
        classify_file(path, lookup)


if __name__ == "__main__":
    main()
