from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
LOOKUP_FILE = BASE_DIR / "icio_sectors_with_isic_rev4_descriptions.csv"
INPUT_FILES = [
    BASE_DIR / "iran_ar_v3.csv",
    BASE_DIR / "iran_ch_v3.csv",
    BASE_DIR / "iran_en_v3.csv",
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

IO_COLUMNS = [
    "IO_V1",
    "IO_Code",
    "IO_Industry",
    "IO_ISIC_Rev4",
    "IO_ISIC_Rev4_description",
    "IO_match_confidence",
]


def clean_whitespace(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(value: object) -> str:
    return clean_whitespace(value).replace("_", " ")


def load_lookup() -> pd.DataFrame:
    lookup = pd.read_csv(LOOKUP_FILE, dtype=str, encoding="utf-8-sig")
    for col in lookup.columns:
        lookup[col] = lookup[col].map(clean_whitespace)

    lookup["ISIC Rev.4"] = lookup["ISIC Rev.4"].str.replace(
        r"\s*,\s*", ", ", regex=True
    )

    lookup["sector_text"] = (
        lookup["Code"]
        + " "
        + lookup["Industry"]
        + " "
        + lookup["ISIC Rev.4"]
        + " "
        + lookup["ISIC Rev. 4 description"]
    ).map(clean_whitespace)
    lookup["V1"] = lookup["V1"].astype(int)
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


def classify_file(path: Path, lookup: pd.DataFrame) -> Path:
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
    best_index = similarities.argmax(axis=1)
    best_score = similarities.max(axis=1)
    selected = lookup.iloc[best_index].reset_index(drop=True)

    df["IO_V1"] = selected["V1"].astype(str)
    df["IO_Code"] = selected["Code"]
    df["IO_Industry"] = selected["Industry"]
    df["IO_ISIC_Rev4"] = selected["ISIC Rev.4"]
    df["IO_ISIC_Rev4_description"] = selected["ISIC Rev. 4 description"]
    df["IO_match_confidence"] = [f"{score:.6f}" for score in best_score]

    zero_score = best_score == 0
    if zero_score.any():
        df.loc[zero_score, IO_COLUMNS] = ""

    outpath = path.with_name(path.stem + "_with_io.csv")
    print(f"Writing {outpath.name}")
    df.to_csv(outpath, index=False, encoding="utf-8-sig")
    print(
        f"{path.name}: rows={len(df):,}, classified={(~zero_score).sum():,}, "
        f"unclassified={zero_score.sum():,}"
    )
    return outpath


def main() -> None:
    lookup = load_lookup()
    for path in INPUT_FILES:
        classify_file(path, lookup)


if __name__ == "__main__":
    main()
