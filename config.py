"""Pipeline settings."""

import os
from pathlib import Path

# LLM (any OpenAI-compatible server). Local servers such as vLLM accept any key.
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
GEN_MODEL = "gpt-6-luna"       # used for generating and rewriting queries
JUDGE_MODEL = "gpt-6-luna"     # used for relevance judgments
# GPT-6 models reason by default and only accept temperature with reasoning off.
# Set to None for servers or models that don't know this parameter (e.g. vLLM).
REASONING_EFFORT = "none"
GEN_TEMPERATURE = None          # > 0 so retries produce different queries
JUDGE_TEMPERATURE = None

# Embeddings (SentenceTransformers)
EMBED_MODEL = "Alibaba-NLP/gte-multilingual-base"

EMBED_MAX_CHARS = 6000

EMBED_BATCH_SIZE = 8

MAX_REWRITE_ROUNDS = 1

# Input data
TEXT_COLUMN = "text"
ID_COLUMN = "id"               # falls back to the row index if the column is missing

# Pipeline
MAX_ATTEMPTS = 2               # total generate + judge attempts per document
TOP_K = 3                      # other documents to check per query
MAX_DOC_CHARS = 32000           # documents are truncated to this length in prompts
STOP_ON_FAILURE = False        # True: stop after step 2 if any document failed

# Paths (relative to this folder, so the script can be run from anywhere)
BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "prompts"
INPUT_DIR = BASE_DIR / "input"         # checked by precheck.py
PROCESS_DIR = BASE_DIR / "in_process"   # intermediate results per step
OUTPUT_DIR = BASE_DIR / "output"        # final CSV and report
