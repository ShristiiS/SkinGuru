import json
import re

import httpx

from config import (
    OPENAI_API_URL,
    OPENAI_TIMEOUT_SECONDS,
    require_openai_config,
)
from tracing import record_http, record_llm_exchange, record_llm_usage, traced

# Verbatim from the n8n node / audit appendix. Do not edit.
INCI_SYSTEM_PROMPT = """You are an expert cosmetic chemist with encyclopedic knowledge of INCI (International Nomenclature of Cosmetic Ingredients) standards used globally.

You will receive a list of cosmetic ingredient names exactly as a seller has written them on a product label. There is NO standardization in how sellers write these — they can be written in any possible incorrect, informal, or non-standard way imaginable.

YOUR ONLY JOB:
Convert every single input entry into its correct standard INCI name.

Apply your full cosmetic chemistry expertise to handle ANY issue you encounter, including but not limited to spelling errors, spacing errors, marketing words, percentages, combined ingredients, common names, non-English words, typos, punctuation, brackets, prefixes, suffixes, or anything else that makes the name non-standard.

STRICT RULES:
1. Every input entry must produce exactly one output object — never skip or drop an input
2. If one entry contains multiple ingredients — split into separate INCI names in the output array
3. Output INCI names must always be in UPPERCASE
4. If you are uncertain about the correct INCI name — clean it up as best as possible and output your best guess
5. Never add explanations, comments, or notes
6. If an ingredient name is already a correct standard INCI name — return it exactly as is
7. Return ONLY valid JSON — no markdown, no preamble, no extra text whatsoever
8. CRITICAL: Never add ANY word that is not present in the original input. This includes plant parts like FLOWER, FRUIT, LEAF, BARK, ROOT, SEED, STEM, OIL, JUICE, or any other word. Your job is ONLY to remove noise and fix errors — never to add missing information.
9. If an input line is NOT an ingredient at all (marketing text, product claims, descriptions, skin concern names like "acne marks") — return an empty array [] for that entry

INPUT will be provided as a list, one per line.
OUTPUT must be a JSON array of objects, one per input line, in the same order as input. Each object must have exactly two fields:
- "input": the exact original input line
- "output": an array of corrected INCI names (usually 1, but multiple if split, or empty [] if not an ingredient)"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, count=1, flags=re.I)
    text = re.sub(r"^```\s*", "", text, count=1)
    text = re.sub(r"\s*```$", "", text, count=1)
    return text.strip()


def _fallback_raw_names(items: list[dict]) -> list[dict]:
    return [
        {**item, "normalized_name": item["ingredient_name"]}
        for item in items
    ]


@traced("llm_inci_normalizer")
def llm_inci_normalizer(items: list[dict]) -> tuple[list[dict], bool]:
    """Node 14 — LLM INCI Normalizer.

    Feeds raw ingredient_name, never normalized_name (Port Decision #2).
    Returns (rows, used_parse_fallback).
    """
    if not items:
        return [], False

    # Port Decision #2: raw pre-regex ingredient_name, not normalized_name.
    ingredient_names = [item["ingredient_name"] for item in items]
    api_key = require_openai_config()
    response = httpx.post(
        OPENAI_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-mini",
            "temperature": 0,
            "max_tokens": 3000,
            "messages": [
                {"role": "system", "content": INCI_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(ingredient_names)},
            ],
        },
        timeout=OPENAI_TIMEOUT_SECONDS,
    )
    record_http(response.status_code, url=OPENAI_API_URL)
    response.raise_for_status()
    payload = response.json()
    record_llm_usage(payload.get("model") or "gpt-4o-mini", payload.get("usage") or {})
    content = payload["choices"][0]["message"]["content"] or ""
    record_llm_exchange({"content": content})

    try:
        parsed = json.loads(_strip_fences(content))
    except (json.JSONDecodeError, TypeError):
        return _fallback_raw_names(items), True

    rows = []
    display_order = 1
    for index, item in enumerate(items):
        entry = parsed[index]
        output_names = entry["output"]
        if not output_names:
            continue
        for name in output_names:
            rows.append(
                {
                    "product_id": item["product_id"],
                    "ingredient_name": name,
                    "normalized_name": name,
                    "original_name": item["original_name"],
                    "display_order": display_order,
                }
            )
            display_order += 1
    return rows, False
