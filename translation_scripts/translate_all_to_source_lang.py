"""Language identification and translation for all CSVs in the current folder.

For each CSV:
  1. Identify the language of the text column (german / french / other).
  2. Take the dominant language (german or french) of the file.
  3. Translate every text not in that language with the OpenAI API.
  4. Identify the language again on the final texts and print any text that is
     still not in the dominant language, so it can be checked manually.

Usage (run from the folder containing the CSVs; needs LLM_API_KEY, and
LLM_BASE_URL for a server other than OpenAI):
    python path/to/translation_scripts/translate_all_to_source_lang.py --column text

Results are written to the 'output' subfolder under the same file names,
with the text column replaced by the final text (original or translation).
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from openai import OpenAI

DetectorFactory.seed = 0  # makes langdetect deterministic

TARGET_LANGUAGES = {"de": "german", "fr": "french"}
MODEL = "gpt-6-sol"


def make_client() -> OpenAI:
    """OpenAI client with the same settings as the pipeline (LLM_API_KEY, LLM_BASE_URL)."""
    return OpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("LLM_API_KEY", "EMPTY"),
    )


def identify_language(text) -> str:
    """Return 'german', 'french' or 'other' for a single text."""
    if not isinstance(text, str) or not text.strip():
        return "other"
    try:
        code = detect(text)
    except LangDetectException:
        return "other"
    return TARGET_LANGUAGES.get(code, "other")


def find_dominant_language(languages: pd.Series) -> str | None:
    """Return the more frequent of 'german' / 'french', or None if neither occurs."""
    counts = languages[languages.isin(TARGET_LANGUAGES.values())].value_counts()
    return counts.idxmax() if not counts.empty else None


def translate_text(client: OpenAI, text: str, target_language: str) -> str:
    """Translate a single text into the target language."""
    prompt = (
        f"Translate the following text to {target_language}. "
        f"Return only the translation.\n\n{text}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def translate_non_dominant(
    df: pd.DataFrame, column: str, dominant: str, client: OpenAI
) -> pd.Series:
    """Return the text column with all non-dominant, non-empty texts translated."""
    has_text = df[column].apply(lambda t: isinstance(t, str) and bool(t.strip()))
    needs_translation = has_text & (df["language"] != dominant)

    final = df[column].copy()
    for idx in df.index[needs_translation]:
        final[idx] = translate_text(client, df.at[idx, column], dominant)
    return final


def find_remaining_rows(df: pd.DataFrame, column: str, dominant: str) -> pd.DataFrame:
    """Return rows whose non-empty final text is still not in the dominant language."""
    has_text = df[column].apply(lambda t: isinstance(t, str) and bool(t.strip()))
    return df[has_text & (df["final_language"] != dominant)]


def report_remaining_rows(remaining: pd.DataFrame, column: str, dominant: str):
    """Print the rows that need a manual check (row numbers as in a spreadsheet)."""
    if remaining.empty:
        print(f"  All final texts are in {dominant}.")
        return
    print(f"  CHECK MANUALLY: {len(remaining)} text(s) not in {dominant}:")
    for idx, row in remaining.iterrows():
        print(f"    row {idx + 2} ({row['final_language']}): {row[column]}")


def process_file(input_path: Path, output_path: Path, column: str, client: OpenAI):
    """Run language id, translation and a second language id on one CSV."""
    df = pd.read_csv(input_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")

    df["language"] = df[column].apply(identify_language)
    dominant = find_dominant_language(df["language"])
    if dominant is None:
        print(f"  No german or french text found, skipping {input_path.name}")
        return

    print(f"  Dominant language: {dominant}")
    df[column] = translate_non_dominant(df, column, dominant, client)
    df["final_language"] = df[column].apply(identify_language)

    report_remaining_rows(find_remaining_rows(df, column, dominant), column, dominant)
    df = df.drop(columns=["language", "final_language"])
    df.to_csv(output_path, index=False)


def process_folder(input_dir: Path, output_dir: Path, column: str, client: OpenAI):
    """Process every .csv in input_dir (not recursive) and write to output_dir."""
    output_dir.mkdir(exist_ok=True)
    for input_path in sorted(input_dir.glob("*.csv")):
        print(f"Processing {input_path.name}")
        process_file(input_path, output_dir / input_path.name, column, client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--column", required=True, help="name of the text column")
    parser.add_argument("--output-dir", default="output", help="output subfolder")
    args = parser.parse_args()

    process_folder(Path("."), Path(args.output_dir), args.column, make_client())


if __name__ == "__main__":
    main()