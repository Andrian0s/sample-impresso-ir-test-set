"""Translate the query column of a merged CSV to English and flag non-English results.

Edit the settings below, then run from the repository root (needs LLM_API_KEY,
and LLM_BASE_URL for a server other than OpenAI):
    python translation_scripts/translate_to_english.py

The translated copy is written to OUTPUT_DIR under the same file name, with the
query column replaced by its English translation. Any translated text that is
not detected as English is printed for a manual check.
"""

import os
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from openai import OpenAI

DetectorFactory.seed = 0  # makes langdetect deterministic

# --- Settings (edit between runs) ---
INPUT_FILE = Path("final_dataset/multilingual/Impresso_IR_test_set_FR.csv")
OUTPUT_DIR = Path("final_dataset/crosslingual/")
QUERY_COLUMN = "query"
# ------------------------------------

MODEL = "gpt-6-luna"


def make_client() -> OpenAI:
    """OpenAI client with the same settings as the pipeline (LLM_API_KEY, LLM_BASE_URL)."""
    return OpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("LLM_API_KEY", "EMPTY"),
    )


def translate_text(client: OpenAI, text: str, target_language: str) -> str:
    """Translate a single text into the target language."""
    prompt = (
        f"Translate the following text to {target_language}. "
        f"Return only the translation.\n\n{text}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        reasoning_effort="none",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

def has_text(value) -> bool:
    """True for non-empty strings."""
    return isinstance(value, str) and bool(value.strip())


def detect_language_code(text: str) -> str:
    """Return the detected language code (e.g. 'en'), or 'unknown'."""
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def translate_column(df: pd.DataFrame, column: str, client: OpenAI) -> pd.Series:
    """Return the column with every non-empty text translated to English."""
    translated = df[column].copy()
    for idx in df.index[df[column].apply(has_text)]:
        translated[idx] = translate_text(client, df.at[idx, column], "English")
    return translated


def find_non_english(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return rows whose non-empty text is not detected as English, with the code."""
    mask = df[column].apply(has_text)
    codes = df.loc[mask, column].apply(detect_language_code)
    flagged = df.loc[codes[codes != "en"].index].copy()
    flagged["detected"] = codes[codes != "en"]
    return flagged


def report_non_english(flagged: pd.DataFrame, column: str):
    """Print rows that need a manual check (row numbers as in a spreadsheet)."""
    if flagged.empty:
        print("All translated texts are detected as English.")
        return
    print(f"CHECK MANUALLY: {len(flagged)} text(s) not detected as English:")
    for idx, row in flagged.iterrows():
        print(f"  row {idx + 2} ({row['detected']}): {row[column]}")


def main():
    df = pd.read_csv(INPUT_FILE)
    if QUERY_COLUMN not in df.columns:
        raise ValueError(f"Column '{QUERY_COLUMN}' not found. Available: {list(df.columns)}")

    df[QUERY_COLUMN] = translate_column(df, QUERY_COLUMN, make_client())
    report_non_english(find_non_english(df, QUERY_COLUMN), QUERY_COLUMN)

    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / INPUT_FILE.name, index=False)


if __name__ == "__main__":
    main()
