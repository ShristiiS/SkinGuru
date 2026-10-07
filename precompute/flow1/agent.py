from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from clients.supabase import (
    calculate_formulation_score,
    calculate_product_concern_scores,
    calculate_product_safety_flags,
    count_botanicals,
    get_active_ingredients_for_interactions,
    get_ingredient_irritation_flags,
    get_predefined_interactions_for_interaction_builder,
    get_product_ingredients_with_functions,
    save_llm_interaction,
    store_product_concern_scores,
    store_product_formulation_score,
    store_product_safety_flags,
)
from precompute.call_retry import call_with_retry
from precompute.flow1.llm import chat_completions_turn, responses_turn
from tracing import record_prompt_values, trace_step
from tracing.context import current_span_fields
from tracing.debug import to_jsonable

MAX_AGENT_MODEL_CALLS = 10
FORMULATION_STORE_NUDGE = (
    "You have not called store_formulation_score. Call it now with the JSON as instructed."
)

CONCERN_CALCULATE_DESCRIPTION = (
    "Calculates concern scores for all 15 concerns for a product. "
    "Input: product_id (number)"
)
CONCERN_STORE_DESCRIPTION = (
    "Stores concern scores for all 15 concerns in product_concern_scores "
    "table. Input: data as JSON string — array of 15 concern objects with "
    "product_id, concern_key, concern_score, bonus_score, concern_reasoning, "
    "concern_contributing_ingredients, bonus_contributing_ingredients, synergy_score, synergy_pairs, "
    "synergy_reasoning, bonus_reasoning, full_explanation"
)


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    parameters: tuple
    run: Callable


@dataclass(frozen=True)
class AgentRunResult:
    text: str
    tool_log: list
    max_calls_hit: bool


class ToolBlock(Exception):
    """Python blocked the tool; ok=false without parsing the result text."""


def _openai_tools(tools: list[AgentTool]) -> list:
    out = []
    for tool in tools:
        properties = {}
        required = []
        for spec in tool.parameters:
            properties[spec["name"]] = {
                "type": spec["type"],
                "description": spec["description"],
            }
            if spec.get("required", True):
                required.append(spec["name"])
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return out


def _responses_tools(tools: list[AgentTool]) -> list:
    out = []
    for tool in tools:
        properties = {}
        required = []
        for spec in tool.parameters:
            properties[spec["name"]] = {
                "type": spec["type"],
                "description": spec["description"],
            }
            if spec.get("required", True):
                required.append(spec["name"])
        out.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        )
    return out


def _assistant_message(message: dict) -> dict:
    out = {"role": message.get("role") or "assistant"}
    if "content" in message:
        out["content"] = message.get("content")
    elif message.get("tool_calls"):
        out["content"] = None
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    return out


def _tool_error_text(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        body = exc.response.text
        if body:
            return body
        return f"HTTP {exc.response.status_code}"
    return str(exc)


def _run_one_tool(tool_call: dict, tools: dict[str, AgentTool]) -> tuple[str, bool]:
    function = tool_call.get("function") or {}
    name = function.get("name") or ""
    raw_args = function.get("arguments")
    if raw_args is None:
        raw_args = ""
    if not isinstance(raw_args, str):
        raw_args = json.dumps(raw_args)

    result = None
    ok = False
    parsed_args = raw_args
    with trace_step(name or "unknown_tool") as fields:
        try:
            try:
                args = json.loads(raw_args)
                parsed_args = args
            except Exception as exc:
                result = str(exc)
                return result, False
            if not isinstance(args, dict):
                result = "JSON.parse of the model's string did not produce an object"
                return result, False
            tool = tools.get(name)
            if tool is None:
                result = f"Unknown tool: {name}"
                return result, False
            for spec in tool.parameters:
                if spec["name"] not in args:
                    if not spec.get("required", True):
                        continue
                    result = f"Required parameter '{spec['name']}' is missing"
                    return result, False
            try:
                result = tool.run(args)
                ok = True
            except ToolBlock as exc:
                result = str(exc)
                ok = False
            except Exception as exc:
                result = _tool_error_text(exc)
                ok = False
            return result, ok
        finally:
            try:
                fields["debug_input"] = {
                    "name": name,
                    "arguments": to_jsonable(parsed_args),
                }
                fields["debug_output"] = to_jsonable(result)
            except Exception:
                pass


def run_tools_agent(
    system: str,
    user: str,
    model: str,
    timeout_seconds: float,
    tools: list[AgentTool],
    max_completion_tokens=None,
    max_calls=MAX_AGENT_MODEL_CALLS,
    nudge_if_tool_missing=None,
    nudge_missing_message=None,
    tool_log=None,
) -> AgentRunResult:
    """n8n Tools Agent v3: up to 10 model calls; tool errors go back to the model."""
    openai_tools = _openai_tools(tools)
    by_name = {tool.name: tool for tool in tools}
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    agent_log = {"model_calls": [], "stop_reason": None}
    parent_fields = current_span_fields()
    if parent_fields is not None:
        parent_fields["agent_debug"] = agent_log
    last_text = ""
    if tool_log is None:
        tool_log = []
    nudged = False
    for call_number in range(1, max_calls + 1):
        with trace_step("tools_agent_llm"):
            message = chat_completions_turn(
                messages,
                model,
                timeout_seconds,
                max_completion_tokens=max_completion_tokens,
                tools=openai_tools,
            )
        turn = {
            "n": call_number,
            "reply": message.get("content") or "",
            "tool_calls": [],
        }
        agent_log["model_calls"].append(turn)
        messages.append(_assistant_message(message))
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            last_text = message.get("content") or ""
            if (
                nudge_if_tool_missing
                and nudge_missing_message
                and not nudged
                and call_number < max_calls
                and not any(
                    isinstance(entry, dict)
                    and entry.get("name") == nudge_if_tool_missing
                    for entry in tool_log
                )
            ):
                nudged = True
                messages.append(
                    {"role": "user", "content": nudge_missing_message}
                )
                continue
            agent_log["stop_reason"] = "no more tool calls"
            break
        for tool_call in tool_calls:
            result, ok = _run_one_tool(tool_call, by_name)
            function = tool_call.get("function") or {}
            arguments = function.get("arguments")
            parsed_arguments = arguments
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except Exception:
                    parsed_arguments = arguments
            turn["tool_calls"].append(
                {
                    "name": function.get("name"),
                    "arguments": to_jsonable(parsed_arguments),
                    "result": to_jsonable(result),
                    "ok": ok,
                }
            )
            tool_log.append(
                {
                    "name": function.get("name"),
                    "arguments": to_jsonable(parsed_arguments),
                    "result": result,
                    "ok": ok,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id") or "",
                    "content": result,
                }
            )
    else:
        agent_log["stop_reason"] = f"{max_calls}-call limit"
    max_calls_hit = agent_log["stop_reason"] == f"{max_calls}-call limit"
    return AgentRunResult(
        text=last_text, tool_log=tool_log, max_calls_hit=max_calls_hit
    )


def run_tools_agent_responses(
    system: str,
    user: str,
    model: str,
    timeout_seconds: float,
    tools: list[AgentTool],
    max_output_tokens=None,
    max_calls=MAX_AGENT_MODEL_CALLS,
) -> AgentRunResult:
    """Interaction Builder only: Responses API tool loop."""
    openai_tools = _responses_tools(tools)
    by_name = {tool.name: tool for tool in tools}
    input_items = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    agent_log = {"model_calls": [], "stop_reason": None}
    parent_fields = current_span_fields()
    if parent_fields is not None:
        parent_fields["agent_debug"] = agent_log
    last_text = ""
    tool_log = []
    for call_number in range(1, max_calls + 1):
        with trace_step("tools_agent_llm"):
            message = responses_turn(
                input_items,
                model,
                timeout_seconds,
                tools=openai_tools,
                max_output_tokens=max_output_tokens,
            )
        turn = {
            "n": call_number,
            "reply": message.get("content") or "",
            "tool_calls": [],
        }
        agent_log["model_calls"].append(turn)
        for item in message.get("output") or []:
            input_items.append(item)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            last_text = message.get("content") or ""
            agent_log["stop_reason"] = "no more tool calls"
            break
        for tool_call in tool_calls:
            result, ok = _run_one_tool(tool_call, by_name)
            function = tool_call.get("function") or {}
            arguments = function.get("arguments")
            parsed_arguments = arguments
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except Exception:
                    parsed_arguments = arguments
            turn["tool_calls"].append(
                {
                    "name": function.get("name"),
                    "arguments": to_jsonable(parsed_arguments),
                    "result": to_jsonable(result),
                    "ok": ok,
                }
            )
            tool_log.append(
                {
                    "name": function.get("name"),
                    "arguments": to_jsonable(parsed_arguments),
                    "result": result,
                    "ok": ok,
                }
            )
            output = result if isinstance(result, str) else str(result)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.get("id") or "",
                    "output": output,
                }
            )
    else:
        agent_log["stop_reason"] = f"{max_calls}-call limit"
    max_calls_hit = agent_log["stop_reason"] == f"{max_calls}-call limit"
    return AgentRunResult(
        text=last_text, tool_log=tool_log, max_calls_hit=max_calls_hit
    )


def _run_calculate_concern_scores(args: dict) -> str:
    return call_with_retry(calculate_product_concern_scores, args["product_id"])


def _run_store_concern_results(args: dict) -> str:
    data = args["data"]
    if not isinstance(data, str):
        raise ToolBlock("Required parameter 'data' must be a string")
    return call_with_retry(store_product_concern_scores, data)


def concern_agent_tools(tool_log, product_id, synergy):
    from precompute.flow1.concern_validate import concern_store_precheck

    def store(args: dict) -> str:
        data = args["data"]
        if not isinstance(data, str):
            raise ToolBlock("Required parameter 'data' must be a string")
        blocked = concern_store_precheck(data, tool_log, product_id, synergy)
        if blocked:
            raise ToolBlock(blocked)
        return call_with_retry(store_product_concern_scores, data)

    return (
        AgentTool(
            name="calculate_concern_scores",
            description=CONCERN_CALCULATE_DESCRIPTION,
            parameters=(
                {
                    "name": "product_id",
                    "type": "number",
                    "description": "0",
                },
            ),
            run=_run_calculate_concern_scores,
        ),
        AgentTool(
            name="store_concern_results",
            description=CONCERN_STORE_DESCRIPTION,
            parameters=(
                {
                    "name": "data",
                    "type": "string",
                    "description": "[]",
                },
            ),
            run=store,
        ),
    )


CONCERN_AGENT_TOOLS = concern_agent_tools([], None, None)


def run_concern_agent(node22: dict, node23, synergy_text: str, synergy=None) -> str:
    from precompute.flow1.prompts import (
        CONCERN_AGENT_SYSTEM,
        render_concern_agent_user,
        strip_leading_equals,
    )

    system = strip_leading_equals(CONCERN_AGENT_SYSTEM)
    user = render_concern_agent_user(
        node22["product_id"], node23, synergy_text
    )
    record_prompt_values(
        {
            "product_id": node22["product_id"],
            "concern_data": node23,
            "synergy_text": synergy_text,
        }
    )
    tool_log = []
    return run_tools_agent(
        system,
        user,
        "gpt-5.4-mini",
        600,
        list(concern_agent_tools(tool_log, node22["product_id"], synergy)),
        max_completion_tokens=32768,
        tool_log=tool_log,
    )


def _parse_model_json(value):
    """JS JSON.parse of a $fromAI string parameter."""
    return json.loads(value)


SAFETY_CALC_RESULT_KEYS = (
    "has_pregnancy_unsafe",
    "pregnancy_unsafe_ingredients",
    "has_lactation_unsafe",
    "lactation_unsafe_ingredients",
    "has_eczema_trigger",
    "eczema_trigger_ingredients",
    "has_rosacea_trigger",
    "rosacea_trigger_ingredients",
    "has_psoriasis_trigger",
    "psoriasis_trigger_ingredients",
    "has_fungal_acne",
    "fungal_acne_ingredients",
    "comedogenicity_ingredients",
    "worst_comedogenicity_rating",
    "is_carcinogenic",
    "is_mutagenic",
    "dermal_severity",
    "has_ghs_toxic",
    "ghs_toxic_ingredients",
    "has_banned",
    "banned_ingredients",
    "has_fragrance",
    "fragrance_ingredients",
    "has_eu26_allergen",
    "eu26_allergen_ingredients",
    "has_preservative",
    "preservative_ingredients",
    "has_sun_sensitivity",
    "sun_sensitivity_ingredients",
    "has_high_irritation",
    "high_irritation_ingredients",
    "has_medium_irritation",
    "medium_irritation_ingredients",
    "avoid_for_skin_types",
    "interaction_deduction",
    "interaction_status",
    "interaction_reason",
    "interaction_worst_case",
    "irritation_synergy_offset",
    "irritation_synergy_status",
    "interaction_synergy_offset",
    "interaction_synergy_status",
)
_REQUIRED_INTERACTION_RESULT_KEYS = (
    "interactions",
    "irritation_synergies",
    "interaction_synergies",
)
_STORE_BEFORE_CALCULATE_ERROR = (
    "ERROR: call calculate_and_store_safety_flags successfully before "
    "store_safety_flags"
)


def _build_safety_store_payload(
    product_id, interaction_result, calc_result, args
) -> dict:
    payload = {"product_id": product_id}
    for key in SAFETY_CALC_RESULT_KEYS:
        payload[key] = calc_result[key]
    payload["interaction_details"] = interaction_result["interactions"]
    payload["irritation_synergy_pairs"] = interaction_result[
        "irritation_synergies"
    ]
    payload["interaction_synergy_pairs"] = interaction_result[
        "interaction_synergies"
    ]
    payload["safety_reasoning"] = args["safety_reasoning"]
    payload["full_explanation"] = args["full_explanation"]
    payload["irritation_synergy_reasoning"] = args[
        "irritation_synergy_reasoning"
    ]
    payload["interaction_synergy_reasoning"] = args[
        "interaction_synergy_reasoning"
    ]
    return payload


def build_safety_agent_tools(product_id, interaction_result) -> list:
    state = {"calc_result": None}

    def run_calculate(_args: dict) -> str:
        raw = call_with_retry(
            calculate_product_safety_flags, product_id, interaction_result
        )
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return raw
        if isinstance(parsed, dict):
            state["calc_result"] = parsed
        return raw

    def run_store(args: dict) -> str:
        calc_result = state["calc_result"]
        if not isinstance(calc_result, dict):
            raise ToolBlock(_STORE_BEFORE_CALCULATE_ERROR)
        missing = [
            key for key in SAFETY_CALC_RESULT_KEYS if key not in calc_result
        ]
        if missing:
            raise ToolBlock(
                "ERROR: calculate_and_store_safety_flags result is missing "
                f"fields: {missing}"
            )
        payload = _build_safety_store_payload(
            product_id, interaction_result, calc_result, args
        )
        return call_with_retry(store_product_safety_flags, payload)

    return [
        AgentTool(
            name="calculate_and_store_safety_flags",
            description=(
                "Calculates safety flags for this product using its "
                "interaction_result, which the system supplies automatically. "
                "Returns all calculated safety flags. Input: product_id "
                "(number)"
            ),
            parameters=(
                {
                    "name": "product_id",
                    "type": "number",
                    "description": (
                        "The product_id you received in the user message"
                    ),
                },
            ),
            run=run_calculate,
        ),
        AgentTool(
            name="store_safety_flags",
            description=(
                "Stores this product's safety flags. The system adds "
                "product_id, all calculated flag values and the interaction "
                "arrays automatically. Input: safety_reasoning (string), "
                "full_explanation (string), irritation_synergy_reasoning "
                "(string), interaction_synergy_reasoning (string)"
            ),
            parameters=(
                {
                    "name": "safety_reasoning",
                    "type": "string",
                    "description": "Step 2 safety_reasoning text",
                },
                {
                    "name": "full_explanation",
                    "type": "string",
                    "description": "Step 2 full_explanation text",
                },
                {
                    "name": "irritation_synergy_reasoning",
                    "type": "string",
                    "description": "Step 2 irritation_synergy_reasoning text",
                },
                {
                    "name": "interaction_synergy_reasoning",
                    "type": "string",
                    "description": (
                        "Step 2 interaction_synergy_reasoning text"
                    ),
                },
            ),
            run=run_store,
        ),
    ]


def _run_get_product_ingredients_with_functions(args: dict) -> str:
    return call_with_retry(
        get_product_ingredients_with_functions, args["product_id"]
    )


def _run_count_botanicals(args: dict) -> str:
    return call_with_retry(count_botanicals, args["product_id"])


def _run_calculate_formulation_score(args: dict) -> str:
    return call_with_retry(
        calculate_formulation_score,
        args["product_id"],
        args["product_type"],
        args["botanical_count"],
    )


def _run_store_formulation_score(args: dict) -> str:
    parsed = _parse_model_json(args["data"])
    return call_with_retry(store_product_formulation_score, parsed)


def formulation_agent_tools(tool_log, node22):
    from precompute.flow1.formulation_validate import formulation_store_precheck

    def store(args: dict) -> str:
        data = args["data"]
        blocked = formulation_store_precheck(data, tool_log, node22)
        if blocked:
            raise ToolBlock(blocked)
        parsed = _parse_model_json(data)
        return call_with_retry(store_product_formulation_score, parsed)

    return (
        AgentTool(
            name="get_product_ingredients_with_functions",
            description=(
                "Fetches all ingredients with their function categories for a "
                "product. Input: product_id (number)"
            ),
            parameters=(
                {
                    "name": "product_id",
                    "type": "number",
                    "description": "0",
                },
            ),
            run=_run_get_product_ingredients_with_functions,
        ),
        AgentTool(
            name="count_botanicals",
            description=(
                "Counts botanical ingredients in a product. Input: product_id "
                "(number)"
            ),
            parameters=(
                {
                    "name": "product_id",
                    "type": "number",
                    "description": "0",
                },
            ),
            run=_run_count_botanicals,
        ),
        AgentTool(
            name="calculate_formulation_score",
            description=(
                "Calculates formulation score across 4 components. Input: "
                "product_id (number), product_type (string), botanical_count "
                "(number)"
            ),
            parameters=(
                {
                    "name": "product_id",
                    "type": "number",
                    "description": "0",
                },
                {
                    "name": "product_type",
                    "type": "string",
                    "description": "",
                },
                {
                    "name": "botanical_count",
                    "type": "number",
                    "description": "0",
                },
            ),
            run=_run_calculate_formulation_score,
        ),
        AgentTool(
            name="store_formulation_score",
            description=(
                "Stores formulation score. Input: data as a valid JSON string. "
                "IMPORTANT: escape all double quotes inside string values, no "
                "trailing commas, no comments. Validate JSON before passing."
            ),
            parameters=(
                {
                    "name": "data",
                    "type": "string",
                    "description": "{}",
                },
            ),
            run=store,
        ),
    )


FORMULATION_AGENT_TOOLS = formulation_agent_tools([], {})


def _run_get_active_ingredients_for_interactions(args: dict) -> str:
    return call_with_retry(
        get_active_ingredients_for_interactions, args["product_id"]
    )


def _run_get_predefined_interactions_for_interaction_builder(
    args: dict,
) -> str:
    return call_with_retry(
        get_predefined_interactions_for_interaction_builder,
        args["product_id"],
    )


def _run_get_ingredient_irritation_flags(args: dict) -> str:
    return call_with_retry(
        get_ingredient_irritation_flags, args["product_id"]
    )


def _run_save_llm_interaction(args: dict) -> str:
    severity = args.get("severity")
    if severity in (None, ""):
        severity = None
    return call_with_retry(
        save_llm_interaction,
        args["ingredient_a"],
        args["ingredient_b"],
        args["raw_type"],
        severity,
        args["reason"],
        args.get("is_irritation_related"),
    )


INTERACTION_BUILDER_AGENT_TOOLS = (
    AgentTool(
        name="get_active_ingredients_for_interactions",
        description=(
            "Gets only active ingredients for this product that need "
            "interaction checking. Returns list of ingredient names. "
            "Input: product_id (integer)"
        ),
        parameters=(
            {
                "name": "product_id",
                "type": "number",
                "description": "0",
            },
        ),
        run=_run_get_active_ingredients_for_interactions,
    ),
    AgentTool(
        name="get_predefined_interactions",
        description=(
            "Checks predefined ingredient interactions database for known "
            "interaction pairs found in this product. Call this BEFORE "
            "searching web. Returns all known pairs with interaction_type, "
            "interaction_severity and reason. Input: product_id (integer)"
        ),
        parameters=(
            {
                "name": "product_id",
                "type": "number",
                "description": "0",
            },
        ),
        run=_run_get_predefined_interactions_for_interaction_builder,
    ),
    AgentTool(
        name="get_ingredient_irritation_flags",
        description=(
            "Gets which of this product's ingredients have high or medium "
            "irritation potential, sourced from the ingredient database. "
            "Call this before building irritation_synergies or "
            "interaction_synergies in Step 5. Returns list of "
            "ingredient_name and irritation_potential. Input: product_id "
            "(integer)"
        ),
        parameters=(
            {
                "name": "product_id",
                "type": "number",
                "description": "0",
            },
        ),
        run=_run_get_ingredient_irritation_flags,
    ),
    AgentTool(
        name="save_llm_interaction",
        description=(
            "Saves a newly LLM-classified ingredient interaction pair to "
            "the predefined interactions database, so future runs can "
            "reuse it instead of re-classifying. Call this once for each "
            "pair you classified yourself (source = \"llm\") that was NOT "
            "already found via get_predefined_interactions. Do not call "
            "this for pairs that came from the database already. Input: "
            "ingredient_a (string), ingredient_b (string), raw_type "
            "(string), severity (string, synergistic only), reason "
            "(string), is_irritation_related (boolean, synergistic only)."
        ),
        parameters=(
            {
                "name": "ingredient_a",
                "type": "string",
                "description": "",
            },
            {
                "name": "ingredient_b",
                "type": "string",
                "description": "",
            },
            {
                "name": "raw_type",
                "type": "string",
                "description": "",
            },
            {
                "name": "severity",
                "type": "string",
                "description": "",
                "required": False,
            },
            {
                "name": "reason",
                "type": "string",
                "description": "",
            },
            {
                "name": "is_irritation_related",
                "type": "boolean",
                "description": "",
                "required": False,
            },
        ),
        run=_run_save_llm_interaction,
    ),
)


def run_interaction_builder_agent(
    product_id, model="gpt-5.6-luna", timeout=300.0
):
    from precompute.flow1.prompts import (
        INTERACTION_BUILDER_AGENT_SYSTEM,
        render_interaction_builder_agent_user,
        strip_leading_equals,
    )

    record_prompt_values({"product_id": product_id})
    return run_tools_agent_responses(
        strip_leading_equals(INTERACTION_BUILDER_AGENT_SYSTEM),
        render_interaction_builder_agent_user(product_id),
        model,
        timeout,
        list(INTERACTION_BUILDER_AGENT_TOOLS),
        max_output_tokens=32768,
        max_calls=50,
    )


def run_safety_agent(
    node22: dict, interaction_result, model="gpt-5.4-mini", timeout=300.0
) -> str:
    from precompute.flow1.prompts import (
        SAFETY_AGENT_SYSTEM,
        render_safety_agent_user,
        strip_leading_equals,
    )

    missing = [
        key
        for key in _REQUIRED_INTERACTION_RESULT_KEYS
        if key not in interaction_result
    ]
    if missing:
        raise KeyError(missing[0])

    record_prompt_values({"product_id": node22["product_id"]})
    return run_tools_agent(
        strip_leading_equals(SAFETY_AGENT_SYSTEM),
        render_safety_agent_user(node22["product_id"], interaction_result),
        model,
        timeout,
        build_safety_agent_tools(node22["product_id"], interaction_result),
        max_completion_tokens=32768,
    )


def run_formulation_agent(node22: dict) -> str:
    from precompute.flow1.prompts import (
        FORMULATION_AGENT_SYSTEM,
        render_formulation_agent_user,
        strip_leading_equals,
    )

    record_prompt_values(
        {
            "product_id": node22["product_id"],
            "formulation_product_type": node22.get("formulation_product_type"),
        }
    )
    tool_log = []
    return run_tools_agent(
        strip_leading_equals(FORMULATION_AGENT_SYSTEM),
        render_formulation_agent_user(
            node22["product_id"],
            node22.get("formulation_product_type"),
        ),
        "gpt-5.4-nano",
        300,
        list(formulation_agent_tools(tool_log, node22)),
        max_completion_tokens=32768,
        nudge_if_tool_missing="store_formulation_score",
        nudge_missing_message=FORMULATION_STORE_NUDGE,
        tool_log=tool_log,
    )
