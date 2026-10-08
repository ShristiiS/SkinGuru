import json
import re

import httpx

from config import (
    OPENAI_API_URL,
    OPENAI_TIMEOUT_SECONDS,
    require_openai_config,
)
from ingestion.product_record import record_inci_dropped, record_rerun
from precompute.call_retry import call_with_retry
from tracing import record_http, record_llm_exchange, record_llm_usage, trace_step, traced

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


def format_inci_feedback(problems, ingredient_names) -> str:
    lines = ["Your reply has these problems:"]
    for problem in problems:
        lines.append(f" - {problem}")
    lines.append(
        " Return the complete JSON array again with exactly "
        f"{len(ingredient_names)} objects, one per input line, in the same "
        "order. Each object must have \"input\" equal to that line and "
        '"output" a list of strings (empty [] if the line is not an ingredient).'
    )
    return "\n".join(lines)


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


@traced("llm_inci_normalizer")
def llm_inci_normalizer(items: list[dict]) -> list[dict]:
    """Node 14 — LLM INCI Normalizer.

    Feeds raw ingredient_name, never normalized_name (Port Decision #2).
    """
    if not items:
        return []

    # Port Decision #2: raw pre-regex ingredient_name, not normalized_name.
    ingredient_names = [item["ingredient_name"] for item in items]
    api_key = require_openai_config()
    user_content = "\n".join(ingredient_names)
    previous_text = None
    feedback = None
    last_error: BaseException | None = None

    for run in range(1, INCI_MAX_RUNS + 1):
        try:
            with trace_step("inci_normalizer_run") as fields:
                fields["debug_input"] = {"run": run}
                if feedback:
                    fields["debug_input"]["problems"] = feedback
                messages = [
                    {"role": "system", "content": INCI_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                if previous_text is not None and feedback is not None:
                    messages.append(
                        {"role": "assistant", "content": previous_text}
                    )
                    messages.append({"role": "user", "content": feedback})

                def _post_inci():
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
                            "messages": messages,
                        },
                        timeout=OPENAI_TIMEOUT_SECONDS,
                    )
                    record_http(response.status_code, url=OPENAI_API_URL)
                    response.raise_for_status()
                    return response

                payload = call_with_retry(_post_inci).json()
                record_llm_usage(
                    payload.get("model") or "gpt-4o-mini",
                    payload.get("usage") or {},
                )
                content = payload["choices"][0]["message"]["content"] or ""
                record_llm_exchange({"content": content})
                previous_text = content

                parsed, problems = collect_inci_problems(
                    content, ingredient_names
                )
                fields["debug_output"] = {
                    "content": content,
                    "problems": problems,
                }
                if problems:
                    feedback = format_inci_feedback(problems, ingredient_names)
                    fields["debug_output"]["feedback"] = feedback
                    raise ValueError("\n".join(problems))

                rows, dropped = _rows_from_parsed(items, parsed)
                fields["debug_output"]["dropped"] = dropped
                if dropped:
                    record_inci_dropped(dropped)
                if not rows:
                    raise RuntimeError(ALL_INCI_DROPPED_REASON)
                return rows
        except RuntimeError as exc:
            if str(exc) == ALL_INCI_DROPPED_REASON:
                raise
            last_error = exc
            record_rerun("inci_normalizer", run, str(exc))
            if run == INCI_MAX_RUNS:
                raise
        except Exception as exc:
            last_error = exc
            record_rerun("inci_normalizer", run, str(exc))
            if run == INCI_MAX_RUNS:
                raise
    raise last_error
