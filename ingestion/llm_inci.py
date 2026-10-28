import json
import re
import string

from clients.supabase import get_ingredients_by_canonical_names
from config import OPENAI_TIMEOUT_SECONDS
from ingestion.inci_catalog import (
    ensure_inci_ingredient_catalog,
    near_match_key,
    pick_near_match,
)
from ingestion.product_record import record_inci_dropped, record_rerun, record_warning
from precompute.call_retry import call_with_retry
from precompute.llm_call import QUALITY_MAX_RUNS, format_llm_feedback, run_llm_call
from tracing import trace_step, traced
from tracing.step import record_warnings

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


_PUNCT_CLASS = re.escape(string.punctuation)
_INPUT_COMPARE_PUNCT = re.compile(f"[{_PUNCT_CLASS}]+")
_RULE8_PUNCT = re.compile(f"[{_PUNCT_CLASS}]+")
_WHITESPACE = re.compile(r"\s+")
_CONCENTRATION = re.compile(r"\d+(?:\.\d+)?\s*%")
_EMPTY_BRACKETS = re.compile(r"\(\)")
_EDGE_PUNCT = frozenset(".,;:")
_RULE8_WORDS = (
    "FLOWER",
    "FRUIT",
    "LEAF",
    "BARK",
    "ROOT",
    "SEED",
    "STEM",
    "OIL",
    "JUICE",
)
_COMBINED_SPLIT = re.compile(r" AND | & | \+ ", re.I)
CANONICAL_LOOKUP_BATCH = 50
INCI_LOOKUP_WARNING = "canonical_name lookup failed"
INCI_TABLE_UNKNOWN = (
    " is not in the ingredient table. Give the official INCI name if one "
    "exists. If it is a genuinely new ingredient, repeat the same name exactly."
)


def _input_compare_key(value) -> str:
    if not isinstance(value, str):
        return ""
    text = _INPUT_COMPARE_PUNCT.sub("", value.lower())
    return _WHITESPACE.sub(" ", text).strip()


def _rule8_words(value: str) -> set:
    text = _RULE8_PUNCT.sub(" ", (value or "").upper())
    text = _WHITESPACE.sub(" ", text).strip()
    if not text:
        return set()
    return set(text.split(" "))


def _output_name_problems(line_no: int, name: str, original_line: str) -> list:
    problems = []
    if name.strip() == "":
        problems.append(f"line {line_no}: '{name}' is empty")
        return problems
    if name != name.strip():
        problems.append(
            f"line {line_no}: '{name}' has spaces at the start or end"
        )
    if "  " in name:
        problems.append(
            f"line {line_no}: '{name}' contains two or more spaces in a row"
        )
    if name[:1] in _EDGE_PUNCT or name[-1:] in _EDGE_PUNCT:
        problems.append(
            f"line {line_no}: '{name}' starts or ends with punctuation"
        )
    if _CONCENTRATION.search(name) or _EMPTY_BRACKETS.search(name):
        problems.append(
            f"line {line_no}: '{name}' contains a concentration. "
            "Return only the ingredient name, without the percentage."
        )
    if name != name.upper():
        problems.append(f"line {line_no}: '{name}' must be in UPPERCASE.")
    allowed = _rule8_words(original_line)
    for word in _RULE8_WORDS:
        if word in _rule8_words(name) and word not in allowed:
            problems.append(
                f"line {line_no}: '{name}' adds the word '{word}', which is "
                "not in the original input. Never add words."
            )
    return problems


def _is_combined_name(name: str) -> bool:
    return bool(_COMBINED_SPLIT.search(name))


def _unique_lookup_keys(names) -> list[str]:
    seen = set()
    keys = []
    for name in names:
        if not isinstance(name, str):
            continue
        key = name.upper().strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _combined_output_names(parsed, ingredient_names: list[str]) -> list[str]:
    names = []
    if not isinstance(parsed, list):
        return names
    for index, _line in enumerate(ingredient_names):
        if index >= len(parsed) or not isinstance(parsed[index], dict):
            continue
        output = parsed[index].get("output")
        if not _output_is_list_of_strings(output):
            continue
        for name in output:
            if _is_combined_name(name):
                names.append(name)
    return names


def _preview_parsed(content: str):
    try:
        parsed = json.loads(_strip_fences(content))
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed


class CanonicalNameLookup:
    """Cache exact canonical_name hits across INCI quality runs."""

    def __init__(self):
        self.known = set()
        self.looked_up = set()
        self.failed = False

    def fetch(self, names) -> set | None:
        if self.failed:
            return None
        pending = [key for key in _unique_lookup_keys(names) if key not in self.looked_up]
        if pending:
            try:
                for start in range(0, len(pending), CANONICAL_LOOKUP_BATCH):
                    chunk = pending[start : start + CANONICAL_LOOKUP_BATCH]
                    rows = call_with_retry(get_ingredients_by_canonical_names, chunk)
                    for row in rows or []:
                        canonical = row.get("canonical_name")
                        if isinstance(canonical, str) and canonical:
                            self.known.add(canonical)
                self.looked_up.update(pending)
            except Exception as exc:
                self.failed = True
                message = f"{INCI_LOOKUP_WARNING}: {exc}"
                record_warning(message)
                record_warnings([message])
                return None
        return self.known


class InciTableResolver:
    """After A–F: exact/near-match table names, or ask once about unknowns."""

    def __init__(self):
        self.asked = set()
        self.runs = 0
        self.traces: list[str] = []

    def _lookup(self, name: str, exact, near):
        if name in exact:
            return "exact", name
        matches = near.get(near_match_key(name)) or []
        if not matches:
            return "unknown", name
        picked = pick_near_match(matches)
        return "near", picked["canonical_name"]

    def _walk_output_names(self, parsed):
        names = []
        if not isinstance(parsed, list):
            return names
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            output = entry.get("output")
            if not _output_is_list_of_strings(output):
                continue
            for name in output:
                names.append((entry, name))
        return names

    def check(self, parsed) -> list[str]:
        self.runs += 1
        catalog = ensure_inci_ingredient_catalog()
        if catalog is None:
            return []
        exact, near = catalog
        problems = []
        for _entry, name in self._walk_output_names(parsed):
            kind, _resolved = self._lookup(name, exact, near)
            if kind != "unknown":
                continue
            if name in self.asked:
                continue
            self.asked.add(name)
            problems.append(f"{name}{INCI_TABLE_UNKNOWN}")
        if problems and self.runs >= QUALITY_MAX_RUNS:
            return []
        return problems

    def apply(self, parsed) -> None:
        catalog = ensure_inci_ingredient_catalog()
        if catalog is None or not isinstance(parsed, list):
            return
        exact, near = catalog
        traces = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            output = entry.get("output")
            if not _output_is_list_of_strings(output):
                continue
            resolved = []
            for name in output:
                kind, table_name = self._lookup(name, exact, near)
                if kind == "near" and table_name != name:
                    traces.append(f"{name} → {table_name} (near-match)")
                    resolved.append(table_name)
                else:
                    resolved.append(name)
            entry["output"] = resolved
        if traces:
            self.traces.extend(traces)
            with trace_step("inci_catalog_near_match") as fields:
                fields["debug_output"] = traces


def collect_inci_problems(
    content: str, ingredient_names: list[str], canonical_names=None
):
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
        line_no = index + 1
        if index >= len(parsed):
            problems.append(f"line {line_no} missing object for {line!r}")
            continue
        entry = parsed[index]
        if not isinstance(entry, dict):
            problems.append(f"line {line_no} is not an object: {entry!r}")
            continue
        actual_input = entry.get("input")
        if _input_compare_key(actual_input) != _input_compare_key(line):
            problems.append(
                f"line {line_no} input {actual_input!r} != {line!r}"
            )
        output = entry.get("output")
        if not _output_is_list_of_strings(output):
            problems.append(
                f"line {line_no} output is not a list of strings: {output!r}"
            )
            continue
        for name in output:
            problems.extend(_output_name_problems(line_no, name, line))
            if canonical_names is not None and _is_combined_name(name):
                if name.upper().strip() not in canonical_names:
                    problems.append(
                        f"line {line_no}: '{name}' contains more than one "
                        "ingredient. Split it into separate INCI names and "
                        "remove 'AND' / '&' / '+'."
                    )
        if canonical_names is not None:
            original_key = line.upper().strip()
            if original_key in canonical_names and output != [original_key]:
                note = (
                    f"line {line_no}: '{line}' is an exact table name; "
                    f"AI wrote {output}, used {original_key}"
                )
                entry["output"] = [original_key]
                with trace_step("inci_rule_f") as fields:
                    fields["debug_output"] = note
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


def _inci_check(
    ingredient_names: list[str],
    lookup: CanonicalNameLookup,
    resolver: InciTableResolver,
):
    def check(text: str):
        unwrapped = unwrap_inci_reply(text)
        preview = _preview_parsed(unwrapped)
        to_fetch = list(ingredient_names)
        to_fetch.extend(_combined_output_names(preview, ingredient_names))
        canonical = lookup.fetch(to_fetch)
        parsed, problems = collect_inci_problems(
            unwrapped, ingredient_names, canonical_names=canonical
        )
        if problems:
            return problems
        return resolver.check(parsed)

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
    lookup = CanonicalNameLookup()
    resolver = InciTableResolver()

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
        check=_inci_check(ingredient_names, lookup, resolver),
        step="inci_normalizer",
        feedback_extra=inci_feedback_extra(ingredient_names),
        on_quality_failure=lambda run, problems: record_rerun(
            "inci_normalizer", run, "\n".join(problems)
        ),
    )
    canonical = None if lookup.failed else lookup.known
    parsed, problems = collect_inci_problems(
        unwrap_inci_reply(result.text),
        ingredient_names,
        canonical_names=canonical,
    )
    if problems:
        raise ValueError("\n".join(problems))
    resolver.apply(parsed)
    rows, dropped = _rows_from_parsed(items, parsed)
    if dropped:
        record_inci_dropped(dropped)
    if not rows:
        raise RuntimeError(ALL_INCI_DROPPED_REASON)
    return rows
