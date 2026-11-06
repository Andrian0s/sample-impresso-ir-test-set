"""Check that the pipeline can run before starting it.

Usage:
    python precheck.py

Checks:
    1. The LLM server accepts the API key and the configured models.
    2. The embedding model loads and encodes text in this environment.
    3. Every CSV in input/ can be loaded by the pipeline.

Exits with code 1 if any check fails.
"""

import sys
import time

import numpy as np
import openai
import pandas as pd

import config
import llm
from pipeline import load_csv


# --- 1. LLM ---

def check_llm():
    if config.API_KEY == "EMPTY" and "api.openai.com" in config.BASE_URL:
        return [("FAIL", "LLM_API_KEY is not set.")]

    # One test call per (model, temperature) pair used by the pipeline.
    pairs = dict.fromkeys([
        (config.GEN_MODEL, config.GEN_TEMPERATURE),
        (config.JUDGE_MODEL, config.JUDGE_TEMPERATURE),
    ])
    results = []
    for model, temperature in pairs:
        label = f"{model} (temperature {temperature}, reasoning_effort {config.REASONING_EFFORT})"
        try:
            answer = llm.chat("Reply with the word OK.", model, temperature)
            results.append(("PASS", f"{label} answered: {answer[:40]!r}"))
        except openai.AuthenticationError:
            return [("FAIL", f"API key rejected by {config.BASE_URL}")]
        except openai.APIConnectionError:
            return [("FAIL", f"Cannot connect to {config.BASE_URL}")]
        except openai.NotFoundError:
            results.append(("FAIL", f"Model '{model}' not found on {config.BASE_URL}"))
        except Exception as e:
            results.append(("FAIL", f"{label}: {type(e).__name__}: {e}"))
    return results


# --- 2. Embedding ---

def check_embedding():
    try:
        start = time.time()
        model = llm.get_embedder()
        vecs = llm.embed(["A short test sentence.", "Ein kurzer Testsatz."])
        seconds = time.time() - start
    except Exception as e:
        return [("FAIL", f"Could not load or run '{config.EMBED_MODEL}': {type(e).__name__}: {e}")]

    if not np.isfinite(vecs).all():
        return [("FAIL", f"'{config.EMBED_MODEL}' returned invalid values (NaN or inf).")]

    results = [("PASS", f"'{config.EMBED_MODEL}' runs on {model.device}, dimension {vecs.shape[1]}, "
                        f"max {model.max_seq_length} tokens ({seconds:.1f}s incl. loading)")]
    if model.device.type == "cpu":
        results.append(("WARN", "Running on CPU. Embedding large corpora will be slow."))
    return results


# --- 3. Input files ---

def check_inputs():
    if not config.INPUT_DIR.is_dir():
        return [("FAIL", f"Folder not found: {config.INPUT_DIR}")]

    files = sorted(p for p in config.INPUT_DIR.iterdir() if p.is_file() and not p.name.startswith("."))
    csvs = [p for p in files if p.suffix.lower() == ".csv"]
    results = [("WARN", f"{p.name}: not a .csv file, not checked") for p in files if p not in csvs]
    if not csvs:
        return results + [("FAIL", f"No CSV files in {config.INPUT_DIR}")]

    for path in csvs:
        try:
            df, records = load_csv(path)
        except Exception as e:
            results.append(("FAIL", f"{path.name}: {e}"))
            continue
        results += _input_findings(path.name, df, records)
    return results


def _input_findings(name, df, records):
    n = len(records)
    if n == 0:
        return [("FAIL", f"{name}: no rows with text")]

    id_note = f"ids from '{config.ID_COLUMN}'" if config.ID_COLUMN in df else "no id column, row numbers used as ids"
    findings = [("PASS", f"{name}: {n} documents, {id_note}")]

    empty = len(df) - n
    if empty:
        findings.append(("WARN", f"{name}: {empty} rows have no text and will be skipped"))
    if n <= config.TOP_K:
        findings.append(("WARN", f"{name}: only {n} documents, so queries get fewer than {config.TOP_K} neighbors"))
    long_docs = sum(len(r["text"]) > config.MAX_DOC_CHARS for r in records)
    if long_docs:
        findings.append(("WARN", f"{name}: {long_docs} documents longer than {config.MAX_DOC_CHARS} "
                                 "characters will be truncated in prompts"))
    duplicates = pd.Series([r["text"] for r in records]).duplicated().sum()
    if duplicates:
        findings.append(("WARN", f"{name}: {duplicates} duplicate texts; their queries will always match "
                                 "another document"))
    return findings


# --- Main ---

CHECKS = [
    ("1. LLM server and API key", check_llm),
    ("2. Embedding model", check_embedding),
    ("3. Input files", check_inputs),
]


def main():
    failed = 0
    for title, check in CHECKS:
        print(f"\n{title}")
        for status, message in check():
            print(f"  [{status}] {message}")
            failed += status == "FAIL"
    print(f"\n{'All checks passed.' if not failed else f'{failed} check(s) failed.'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
