"""Pipeline steps. Each step takes the list of records, updates it and returns it."""

from tqdm import tqdm

import config
import llm


def final_status(r):
    """Outcome of one document: "ok" if its query goes into the final dataset,
    otherwise the reason it is discarded."""
    if not r.get("valid"):
        return "failed_validation"
    if not r.get("flagged_after", False):
        return "ok"
    if r.get("rewrites") and not r["rewrites"][-1]["own_relevant"]:
        return "rewrite_not_relevant"
    return "still_overlapping"

def step1_generate(records):
    """Generate one query per document."""
    for r in tqdm(records, desc="Step 1: generate"):
        r["query"] = llm.generate_query(r["text"])
    return records


def step2_validate(records):
    """Judge each (query, document) pair. Regenerate failed queries up to MAX_ATTEMPTS."""
    for r in tqdm(records, desc="Step 2: validate"):
        r["attempts"] = []
        for attempt in range(config.MAX_ATTEMPTS):
            if attempt > 0:
                r["query"] = llm.generate_query(r["text"])
            relevant, reason = llm.judge(r["query"], r["text"])
            r["attempts"].append({"query": r["query"], "relevant": relevant, "reason": reason})
            if relevant:
                break
        r["valid"] = relevant
    return records


def step3_retrieve(records):
    """Retrieve the top-k other documents per query and judge their relevance."""
    active = _valid(records)
    items = [(r, r["query"]) for r in active]
    for r, result in zip(active, _retrieve_and_judge(records, items, "Step 3: retrieve + judge")):
        r["neighbors"], r["flagged"], r["own_rank"] = result["neighbors"], result["flagged"], result["own_rank"]
    return records


# --- Steps 4-5: rewrite rounds ---
# The current version of each query is kept in final_query / neighbors_after /
# flagged_after / own_rank_after. It always matches its own document.
# Every rewrite attempt is stored in r["rewrites"].

def init_rewrites(records):
    """Start the rewrite state from the step 3 results (only once per record)."""
    for r in _valid(records):
        if "rewrites" not in r:
            r["final_query"] = r["query"]
            r["neighbors_after"] = r["neighbors"]
            r["flagged_after"] = r["flagged"]
            r["own_rank_after"] = r["own_rank"]
            r["rewrites"] = []
            r["rewritten"] = False
    return records


def to_fix(records):
    """Queries whose current version still matches another document."""
    return [r for r in _valid(records) if r["flagged_after"]]


def rounds_done(records):
    return max((len(r.get("rewrites", [])) for r in records), default=0)


def step4_rewrite(records, round_no):
    """Rewrite each still-flagged query, starting from its current version."""
    by_id = {r["doc_id"]: r for r in records}
    for r in tqdm(to_fix(records), desc=f"Step 4: rewrite (round {round_no})"):
        others = [by_id[n["doc_id"]]["text"] for n in r["neighbors_after"]]
        r["pending"] = llm.rewrite_query(r["final_query"], r["text"], others)
    return records


def step5_recheck(records, round_no):
    """Check new rewrites against their own document. Rewrites that pass are
    retrieved and judged again; the others are discarded and retried next round."""
    pending = [r for r in records if "pending" in r]
    passed = []
    for r in tqdm(pending, desc=f"Step 5: re-check own document (round {round_no})"):
        query = r.pop("pending")
        relevant, reason = llm.judge(query, r["text"])
        r["rewrites"].append({"round": round_no, "query": query, "own_relevant": relevant, "own_reason": reason})
        if relevant:
            passed.append(r)

    items = [(r, r["rewrites"][-1]["query"]) for r in passed]
    results = _retrieve_and_judge(records, items, f"Step 5: retrieve + judge (round {round_no})")
    for r, result in zip(passed, results):
        r["rewrites"][-1].update(result)
        r["final_query"] = r["rewrites"][-1]["query"]
        r["neighbors_after"] = result["neighbors"]
        r["flagged_after"] = result["flagged"]
        r["own_rank_after"] = result["own_rank"]
        r["rewritten"] = True
    return records


# --- Helpers ---

def _valid(records):
    return [r for r in records if r.get("valid")]


_doc_vecs = None  # document embeddings, computed once per run


def _document_vectors(records):
    """Embed all documents on first use and reuse the result (documents never change)."""
    global _doc_vecs
    if _doc_vecs is None:
        _doc_vecs = llm.embed([r["text"] for r in records])
    return _doc_vecs


def _retrieve_and_judge(records, items, desc):
    """For each (record, query) in items: rank all documents, skip the record's own
    document and judge the top-k others. Returns one result dict per item."""
    if not items:
        return []
    doc_vecs = _document_vectors(records)
    query_vecs = llm.embed([query for _, query in items])
    scores = query_vecs @ doc_vecs.T
    position = {r["doc_id"]: i for i, r in enumerate(records)}

    results = []
    for (r, query), row in tqdm(zip(items, scores), total=len(items), desc=desc):
        own = position[r["doc_id"]]
        ranking = row.argsort()[::-1]
        neighbors = []
        for idx in ranking:
            if len(neighbors) == config.TOP_K:
                break
            if idx == own:
                continue
            other = records[idx]
            relevant, reason = llm.judge(query, other["text"])
            neighbors.append({
                "doc_id": other["doc_id"],
                "score": round(float(row[idx]), 4),
                "relevant": relevant,
                "reason": reason,
            })
        results.append({
            "neighbors": neighbors,
            "flagged": any(n["relevant"] for n in neighbors),
            "own_rank": int((ranking == own).nonzero()[0][0]) + 1,
        })
    return results