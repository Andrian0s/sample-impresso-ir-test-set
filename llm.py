"""LLM and embedding helpers. Prompts are read from .txt files in PROMPTS_DIR."""

import json
import re
from pathlib import Path

from openai import OpenAI
from sentence_transformers import SentenceTransformer

import config

_client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)
_embedder = None
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


# --- Prompts ---

def load_prompt(name):
    """Read prompts/<name>.txt."""
    return (Path(config.PROMPTS_DIR) / f"{name}.txt").read_text(encoding="utf-8")


def render(template, **values):
    """Fill {{ placeholder }} fields. Raises KeyError if a value is missing."""
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


def truncate(text):
    return text[: config.MAX_DOC_CHARS]


# --- LLM calls ---

def chat(prompt, model, temperature):
    options = {"reasoning_effort": config.REASONING_EFFORT} if config.REASONING_EFFORT else {}
    response = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        **options,
    )
    return response.choices[0].message.content.strip()


def generate_query(document):
    prompt = render(load_prompt("generate_query"), document=truncate(document))
    return _clean_query(chat(prompt, config.GEN_MODEL, config.GEN_TEMPERATURE))


def judge(query, document):
    """Return (relevant, reason)."""
    prompt = render(load_prompt("judge_relevance"), query=query, document=truncate(document))
    return _parse_judgment(chat(prompt, config.JUDGE_MODEL, config.JUDGE_TEMPERATURE))


def rewrite_query(query, target, others):
    others_text = "\n\n".join(f"[{i}]\n{truncate(t)}" for i, t in enumerate(others, 1))
    prompt = render(
        load_prompt("rewrite_query"),
        query=query,
        target=truncate(target),
        others=others_text,
    )
    return _clean_query(chat(prompt, config.GEN_MODEL, config.GEN_TEMPERATURE))


def _clean_query(raw):
    """Keep the first non-empty line and drop surrounding quotes."""
    line = next((l for l in raw.splitlines() if l.strip()), "")
    return line.strip().strip("\"'").strip()


def _parse_judgment(raw):
    """Parse {"relevant": ..., "reason": ...}. Unparseable output counts as not relevant."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0))
        relevant = data["relevant"]
        if isinstance(relevant, str):
            relevant = relevant.strip().lower() == "true"
        return bool(relevant), str(data.get("reason", ""))
    except (AttributeError, json.JSONDecodeError, KeyError):
        return False, f"Could not parse judge output: {raw[:200]}"


# --- Embeddings ---

def get_embedder():
    """Load the embedding model once and reuse it."""
    global _embedder
    if _embedder is None:
        # gte-multilingual-base ships its own model code, which needs trust_remote_code.
        _embedder = SentenceTransformer(config.EMBED_MODEL, trust_remote_code=True)
    return _embedder


def embed(texts):
    """Return normalised embeddings, so a dot product equals cosine similarity."""
    return get_embedder().encode(
        [text[: config.EMBED_MAX_CHARS] for text in texts],
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )