"""Build an IR test set from a CSV of documents.

Usage:
    python pipeline.py input/example.csv
    python pipeline.py input/example.csv --start-from 3 --resume in_process/<run>_process2.jsonl
    python pipeline.py input/example.csv --start-from 4 --resume in_process/<run>_process3.jsonl

Intermediate results: in_process/<run>_process<N>.jsonl for steps 1-3,
                      in_process/<run>_process<4|5>_round<R>.jsonl for the rewrite rounds
Final results:        output/<run>_queries.csv   (input rows with a valid final query + `query`)
                      output/<run>_failed.csv    (discarded rows + reason)
                      output/<run>_rewrites.json (every rewritten query, before and after)
                      output/<run>_report.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
import report
import steps

STEPS = {
    1: steps.step1_generate,
    2: steps.step2_validate,
    3: steps.step3_retrieve,
}

NEW_COLUMNS = ["query", "original_query", "rewritten_query", "fail_reason"]


# --- I/O ---

def load_csv(path):
    """Return the input DataFrame and one record per row with text."""
    df = pd.read_csv(path)
    if config.TEXT_COLUMN not in df:
        raise ValueError(f"Input has no '{config.TEXT_COLUMN}' column.")
    clash = set(NEW_COLUMNS) & set(df.columns)
    if clash:
        raise ValueError(f"Input already has columns the pipeline would add: {sorted(clash)}")

    keys = _doc_keys(df)
    if keys.duplicated().any():
        raise ValueError(f"Column '{config.ID_COLUMN}' contains duplicate ids.")
    has_text = df[config.TEXT_COLUMN].notna()
    records = [
        {"doc_id": k, "text": str(t)}
        for k, t in zip(keys[has_text], df.loc[has_text, config.TEXT_COLUMN])
    ]
    return df, records


def _doc_keys(df):
    """Document ids as strings: the ID column if present, otherwise the row index."""
    keys = df[config.ID_COLUMN] if config.ID_COLUMN in df else df.index.to_series()
    return keys.astype(str)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
    
def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
def export_queries_csv(df, records, path):
    """Final dataset: input rows whose query passed every check, plus `query`."""
    final = {r["doc_id"]: r["final_query"] for r in records if steps.final_status(r) == "ok"}
    keys = _doc_keys(df)
    out = df[keys.isin(final.keys())].copy()
    out["query"] = keys.map(final)
    out.to_csv(path, index=False)


def export_failed_csv(df, records, path):
    """Discarded rows: input columns plus the queries tried and the reason."""
    by_id = {r["doc_id"]: r for r in records}
    rows = {}
    for key in _doc_keys(df):
        r = by_id.get(key)
        if r is None:
            rows[key] = {"original_query": None, "rewritten_query": None, "fail_reason": "no_text"}
        elif steps.final_status(r) != "ok":
            rewrites = r.get("rewrites") or []
            rows[key] = {
                "original_query": r.get("query"),  # for failed validation: the last attempt
                "rewritten_query": rewrites[-1]["query"] if rewrites else None,
                "fail_reason": steps.final_status(r),
            }
    keys = _doc_keys(df)
    out = df[keys.isin(rows.keys())].copy()
    for col in ["original_query", "rewritten_query", "fail_reason"]:
        out[col] = keys.map({k: v[col] for k, v in rows.items()})
    out.to_csv(path, index=False)


def export_rewrites_json(records, path):
    """Every query that went through rewriting: neighbors before, each rewrite
    with its checks, and the final outcome."""
    entries = [{
        "doc_id": r["doc_id"],
        "original_query": r["query"],
        "neighbors_before": r["neighbors"],
        "rewrites": r["rewrites"],
        "final_query": r["final_query"] if steps.final_status(r) == "ok" else None,
        "status": steps.final_status(r),
    } for r in records if r.get("rewrites")]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", help=f"CSV with a '{config.TEXT_COLUMN}' column")
    parser.add_argument("--start-from", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--resume", help="process<N>.jsonl to continue from (needed with --start-from > 1)")
    args = parser.parse_args()
    if args.start_from > 1 and not args.resume:
        parser.error("--resume is required when --start-from is above 1")

    df, records = load_csv(args.input_csv)
    if args.resume:
        records = load_jsonl(args.resume)

    config.PROCESS_DIR.mkdir(exist_ok=True)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    run = f"{Path(args.input_csv).stem}_{datetime.now():%Y%m%d_%H%M%S}"

    def save(name):
        save_jsonl(records, config.PROCESS_DIR / f"{run}_{name}.jsonl")

    # Steps 1-3
    for n in range(args.start_from, 4):
        records = STEPS[n](records)
        save(f"process{n}")
        if n == 2:
            failed = sum(not r["valid"] for r in records)
            print(f"{failed} of {len(records)} documents failed after {config.MAX_ATTEMPTS} attempts.")
            if failed and config.STOP_ON_FAILURE:
                print("Stopping after step 2 (STOP_ON_FAILURE = True).")
                return finish(df, records, run)

    # Steps 4-5: rewrite rounds, only for queries that are still flagged
    records = steps.init_rewrites(records)
    skip_rewrite = args.start_from == 5  # resuming from a process4_round<N> file
    for round_no in range(steps.rounds_done(records) + 1, config.MAX_REWRITE_ROUNDS + 1):
        if not skip_rewrite:
            if not steps.to_fix(records):
                break
            records = steps.step4_rewrite(records, round_no)
            save(f"process4_round{round_no}")
        skip_rewrite = False
        records = steps.step5_recheck(records, round_no)
        save(f"process5_round{round_no}")

    finish(df, records, run)


def finish(df, records, run):
    """Write the report, the final dataset, the discarded rows and the rewrite log."""
    out = config.OUTPUT_DIR
    (out / f"{run}_report.md").write_text(report.build(records, run), encoding="utf-8")
    export_queries_csv(df, records, out / f"{run}_queries.csv")
    export_failed_csv(df, records, out / f"{run}_failed.csv")
    export_rewrites_json(records, out / f"{run}_rewrites.json")
    print(f"Done. Results in {out}/:")
    print(f"  {run}_queries.csv   final dataset")
    print(f"  {run}_failed.csv    discarded rows and reasons")
    print(f"  {run}_rewrites.json rewritten queries, before and after")
    print(f"  {run}_report.md")


if __name__ == "__main__":
    main()