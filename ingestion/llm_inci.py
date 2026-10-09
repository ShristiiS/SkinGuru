import json
import re

from config import OPENAI_TIMEOUT_SECONDS
from ingestion.product_record import record_inci_dropped, record_rerun
from precompute.llm_call import format_llm_feedback, run_llm_call
from tracing import traced

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

INCI_MAX_RUNS = 3
ALL_INCI_DROPPED_REASON = "all INCI lines dropped as non-ingredients"
INCI_SCHEMA_NAME = "inci_lines"
INCI_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "output": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["input", "output"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, count=1, flags=re.I)
    text = re.sub(r"^```\s*", "", text, count=1)
    text = re.sub(r"\s*```$", "", text, count=1)
    return text.strip()


def _output_is_list_of_strings(value) -> bool:
    return isinstance(value, list) and all(isinstance(name, str) for name in value)


def collect_inci_problems(content: str, ingredient_names: list[str]):
    """Return (parsed_or_None, problem strings with exact values)."""
    try:
        parsed = json.loads(_strip_fences(content))
    except (json.JSONDecodeError, TypeError):
        return None, ["INCI reply is not valid JSON"]
    if not isinstance(parsed, list):
        return None, ["INCI reply is not a JSON array"]

    problems = []
    if len(parsed) != len(ingredient_names):
        problems.append(
            f"expected {len(ingredient_names)} objects, got {len(parsed)}"
        )

    for index, line in enumerate(ingredient_names):
        if index >= len(parsed):
            problems.append(f"line {index + 1} missing object for {line!r}")
            continue
        entry = parsed[index]
        if not isinstance(entry, dict):
            problems.append(f"line {index + 1} is not an object: {entry!r}")
            continue
        actual_input = entry.get("input")
        if actual_input != line:
            problems.append(
                f"line {index + 1} input {actual_input!r} != {line!r}"
            )
        output = entry.get("output")
        if not _output_is_list_of_strings(output):
            problems.append(
                f"line {index + 1} output is not a list of strings: {output!r}"
            )
    return parsed, problems


def inci_feedback_extra(ingredient_names) -> list:
    return [
        " Return the complete JSON array again with exactly "
        f"{len(ingredient_names)} objects, one per input line, in the same "
        "order. Each object must have \"input\" equal to that line and "
        '"output" a list of strings (empty [] if the line is not an ingredient).'
    ]


def format_inci_feedback(problems, ingredient_names) -> str:
    return format_llm_feedback(problems, inci_feedback_extra(ingredient_names))


def unwrap_inci_reply(content: str) -> str:
    """Turn {\"lines\": [...]} into the JSON array collect_inci_problems expects."""
    try:
        parsed = json.loads(_strip_fences(content))
    except (json.JSONDecodeError, TypeError):
        return content
    if isinstance(parsed, dict) and isinstance(parsed.get("lines"), list):
        return json.dumps(parsed["lines"])
    return content


def _rows_from_parsed(items: list[dict], parsed: list) -> tuple[list[dict], list[str]]:
    rows = []
    dropped = []
    display_order = 1
    for index, item in enumerate(items):
        output_names = parsed[index]["output"]
        if not output_names:
            dropped.append(item["ingredient_name"])
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
    return rows, dropped


def _inci_check(ingredient_names: list[str]):
    def check(text: str):
        _parsed, problems = collect_inci_problems(
            unwrap_inci_reply(text), ingredient_names
        )
        return problems

    return check


@traced("llm_inci_normalizer")
def llm_inci_normalizer(items: list[dict]) -> list[dict]:
    """Node 14 — LLM INCI Normalizer.

    Feeds raw ingredient_name, never normalized_name (Port Decision #2).
    """
    if not items:
        return []

    # Port Decision #2: raw pre-regex ingredient_name, not normalized_name.
    ingredient_names = [item["ingredient_name"] for item in items]
    user_content = "\n".join(ingredient_names)

    result = run_llm_call(
        api="chat",
        model="gpt-4o-mini",
        timeout_seconds=OPENAI_TIMEOUT_SECONDS,
        messages=[
            {"role": "system", "content": INCI_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=3000,
        temperature=0,
        schema_name=INCI_SCHEMA_NAME,
        schema=INCI_SCHEMA,
        check=_inci_check(ingredient_names),
        step="inci_normalizer",
        feedback_extra=inci_feedback_extra(ingredient_names),
        on_quality_failure=lambda run, problems: record_rerun(
            "inci_normalizer", run, "\n".join(problems)
        ),
    )
    parsed, problems = collect_inci_problems(
        unwrap_inci_reply(result.text), ingredient_names
    )
    if problems:
        raise ValueError("\n".join(problems))
    rows, dropped = _rows_from_parsed(items, parsed)
    if dropped:
        record_inci_dropped(dropped)
    if not rows:
        raise RuntimeError(ALL_INCI_DROPPED_REASON)
    return rows
