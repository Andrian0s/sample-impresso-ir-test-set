"""Build the markdown report from the records of a run."""

from collections import Counter

import config
import steps

MAX_EXAMPLES = 5
MAX_FAILED_LISTED = 10


def build(records, run_id):
    lines = [f"# IR test set report (run {run_id})", ""] + _settings()
    if records and "valid" in records[0]:
        lines += _validation(records)
    valid = [r for r in records if r.get("valid")]
    if valid and "flagged" in valid[0]:
        lines += _retrieval(valid)
    if valid and "rewrites" in valid[0]:
        lines += _rewrite(valid)
    lines += _final(records)
    return "\n".join(lines)


# --- Sections ---

def _settings():
    return [
        "## Settings",
        "",
        f"- Generation model: `{config.GEN_MODEL}` (temperature {config.GEN_TEMPERATURE})",
        f"- Judge model: `{config.JUDGE_MODEL}` (temperature {config.JUDGE_TEMPERATURE})",
        f"- Reasoning effort: {config.REASONING_EFFORT}",
        f"- Embedding model: `{config.EMBED_MODEL}`",
        f"- Max attempts: {config.MAX_ATTEMPTS}, top-k: {config.TOP_K}",
        "",
    ]


def _validation(records):
    valid = [r for r in records if r["valid"]]
    failed = [r for r in records if not r["valid"]]
    by_attempt = Counter(len(r["attempts"]) for r in valid)

    lines = [
        "## Steps 1-2: query generation and validation",
        "",
        f"Documents: {len(records)}. Valid queries: {len(valid)} ({_pct(len(valid), len(records))}). "
        f"Failed after {config.MAX_ATTEMPTS} attempts (discarded): {len(failed)}.",
        "",
        "| Attempts needed | Documents |",
        "|---|---|",
    ]
    lines += [f"| {n} | {by_attempt.get(n, 0)} |" for n in range(1, config.MAX_ATTEMPTS + 1)]

    if failed:
        shown = failed[:MAX_FAILED_LISTED]
        lines += ["", f"Failed documents ({len(shown)} of {len(failed)} shown, last judge reason):", ""]
        lines += [f"- `{r['doc_id']}`: {r['attempts'][-1]['reason']}" for r in shown]
    return lines + [""]


def _retrieval(valid):
    metrics = _metrics(valid, "")
    lines = [
        "## Step 3: retrieval and cross-relevance",
        "",
        f"Each valid query was checked against its top {config.TOP_K} retrieved documents "
        "(excluding its own document).",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in metrics.items()]
    return lines + [""]


def _rewrite(valid):
    before, after = _metrics(valid, ""), _metrics(valid, "_after")
    rounds_run = max((len(r["rewrites"]) for r in valid), default=0)
    changed = [r for r in valid if r["rewritten"]]
    still_flagged = sum(r["flagged_after"] for r in valid)

    lines = [
        "## Steps 4-5: rewriting",
        "",
        f"{config.MAX_REWRITE_ROUNDS} rewrite round(s) for queries that match another document. "
        "Queries that still match another document afterwards, or whose rewrite no longer matches "
        "its own document, are discarded.",
        "",
        f"Before rewriting, {sum(r['flagged'] for r in valid)} queries matched another document.",
        "",
        "| Round | Rewritten | Discarded (not relevant to own document) | Resolved | Still flagged |",
        "|---|---|---|---|---|",
    ]
    for round_no in range(1, rounds_run + 1):
        attempts = [a for r in valid for a in r["rewrites"] if a["round"] == round_no]
        discarded = sum(not a["own_relevant"] for a in attempts)
        resolved = sum(a["own_relevant"] and not a["flagged"] for a in attempts)
        lines.append(f"| {round_no} | {len(attempts)} | {discarded} | {resolved} | "
                     f"{len(attempts) - resolved} |")

    lines += [
        "",
        f"{len(changed)} queries were changed. {still_flagged} queries were discarded after "
        f"{rounds_run} round(s).",
        "",
        "| Metric | Before | After |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {before[k]} | {after[k]} |" for k in before]
    lines.append("")

    if changed:
        lines += ["### Examples", ""]
        for r in changed[:MAX_EXAMPLES]:
            lines += [
                f"**{r['doc_id']}** ({len(r['rewrites'])} rewrite attempt(s))",
                f"- Before: {r['query']} ({_n_relevant(r, '')} relevant neighbors)",
                f"- After: {r['final_query']} ({_n_relevant(r, '_after')} relevant neighbors)",
                "",
            ]
    return lines

def _final(records):
    counts = Counter(steps.final_status(r) for r in records)
    lines = [
        "## Final dataset",
        "",
        f"Kept: {counts['ok']} of {len(records)} documents with text ({_pct(counts['ok'], len(records))}).",
        "",
        "| Discard reason | Documents |",
        "|---|---|",
    ]
    for reason in ["failed_validation", "rewrite_not_relevant", "still_overlapping"]:
        lines.append(f"| {reason} | {counts.get(reason, 0)} |")
    return lines + ["", "Rows without text are also listed in the _failed.csv file (reason `no_text`).", ""]

# --- Helpers ---

def _metrics(valid, suffix):
    n = len(valid)
    flagged = sum(r["flagged" + suffix] for r in valid)
    ranks = [r["own_rank" + suffix] for r in valid]
    k = config.TOP_K
    return {
        "Queries with at least one relevant neighbor": f"{flagged} ({_pct(flagged, n)})",
        "Mean relevant neighbors per query": f"{sum(_n_relevant(r, suffix) for r in valid) / n:.2f}",
        "Own document at rank 1 (Hit@1)": _pct(sum(x == 1 for x in ranks), n),
        f"Own document in top {k} (Hit@{k})": _pct(sum(x <= k for x in ranks), n),
        "MRR of own document": f"{sum(1 / x for x in ranks) / n:.3f}",
    }


def _n_relevant(record, suffix):
    return sum(nb["relevant"] for nb in record["neighbors" + suffix])


def _pct(part, total):
    return f"{100 * part / total:.1f}%" if total else "n/a"