from precompute.concentration.helpers import compact_json, js_number_str

LLM1_SYSTEM = (
    "You are a cosmetic formulation expert. You always respond with valid JSON only. "
    "No explanation, no markdown, no code blocks."
)

LLM1_USER_TEMPLATE = """You are a cosmetic formulation expert.
INGREDIENT LIST (in label order):
{{ JSON.stringify($json.ingredients) }}
TASK: Detect the product type.
Classify this formula as exactly one of: CLEANSER, SUNSCREEN, EMULSION, GEL, WATER_SERUM, ANHYDROUS

DETECTION RULES (apply in this order):
- CLEANSER: 2 or more surfactants in top 10 ingredients
IMPORTANT: Emulsifiers like polyglyceryl esters, glyceryl stearate, glyceryl distearate are NOT surfactants. Do not count them as surfactants for the CLEANSER rule. Surfactants for CLEANSER detection are cleansing agents like sodium lauryl sulfate, cocamidopropyl betaine, sodium cocoyl isethionate, decyl glucoside etc.
- SUNSCREEN: 1 or more UV filters in top 15 ingredients
- EMULSION: water or aqua first + 2 or more emollients in top 15 + 1 or more emulsifier in top 20
- GEL: water or aqua first + 1 or more gelling agents in top 15 (carbomer, xanthan gum, hydroxyethylcellulose, hydroxypropyl methylcellulose, HPMC, gelatin, agar, carrageenan, sodium polyacrylate, acrylates copolymer) + 0 or 1 emollients + no emulsifier
- WATER_SERUM: water or aqua first + 0 emulsifiers + 0 gelling agents + 2 or more humectants in top 10
- ANHYDROUS: no water or aqua + 3 or more emollients
Default to WATER_SERUM if nothing matches.

SANITY REVIEW after applying rules:
If you detected WATER_SERUM but the formula has ALL of these:
- water or aqua first
- at least one clear emollient or oil
- at least one solubilizer or emulsifier
- at least one rheology builder typical of gel-cream systems
- overall pattern looks like a moisturiser or gel-cream rather than purely watery serum
Then UPGRADE from WATER_SERUM to EMULSION.

If you detected GEL but the formula has ALL of these:
- water or aqua first
- gelling agent present
- at least one clear emollient or oil
- at least one solubilizer or emulsifier
- overall pattern looks like a gel-cream or gel-moisturiser rather than a pure gel
Then UPGRADE from GEL to EMULSION.

Return ONLY this exact JSON. No explanation, no markdown, no code blocks:
{
  "product_type": "EMULSION"
}"""

_LLM1_INGREDIENTS_PLACEHOLDER = "{{ JSON.stringify($json.ingredients) }}"


def render_llm1_user(ingredients) -> str:
    return LLM1_USER_TEMPLATE.replace(
        _LLM1_INGREDIENTS_PLACEHOLDER,
        compact_json(ingredients),
    )


LLM2_SYSTEM = (
    "You are a cosmetic formulation expert. Think step by step. "
    "At the end of your reasoning return only the JSON on the last line."
)

LLM2_USER_TEMPLATE = """You are a cosmetic formulation expert. Your only task is to find the 1% marker position in this ingredient list.

PRODUCT TYPE: {{ $json.product_type }}

INGREDIENT LIST (in label order):
{{ JSON.stringify($json.ingredients) }}

THE 1% MARKER is the position where the formula transitions from above-1% ingredients to at-or-below-1% ingredients.

FOLLOW THESE STEPS IN EXACT ORDER:

STEP 1 — CLASSIFY EVERY INGREDIENT as one of three types and write it out explicitly:

BULK-USE (comfortable above 1%):
- water, aqua and water-based solvents
- glycerin, propanediol, butylene glycol, pentylene glycol and other polyols
- alcohol denat
- emollient esters — any ester used as skin softener or spreading agent
- plant-derived oils and oil fractions
- fatty alcohols
- silicones
- UV filters
- starches used as main texture base such as corn starch or tapioca starch
- NOTE: cellulose derivatives, microcrystalline cellulose, and synthetic polymer thickeners are LOW-DOSE not bulk-use
- plant extracts appearing in position 1-5 that serve as the formula base
- any ingredient whose primary function is emollient, solvent, humectant or occlusive

BORDERLINE (can be above or below 1% depending on formula):
- some emulsifiers
- some vitamin derivatives
- some texture builders
- some light emollients
- arginine in some formulas
- 1,2-hexanediol
- sodium ascorbyl phosphate
- glyceryl stearate
- hydroxyethylcellulose in some formulas

LOW-DOSE (typically below 1%):
- preservatives
- chelators
- carbomer
- xanthan gum
- gellan gum
- fragrance and essential oils
- most extracts
- tocopherol
- sodium hyaluronate in many formulas
- glutathione
- EDTA
- potent claim actives
- citric acid
- potassium sorbate
- sodium benzoate
- phenoxyethanol
- chlorphenesin
- ethylhexylglycerin
- parfum
- microcrystalline cellulose
- cellulose derivatives
- synthetic polymer thickeners

This classification is not the final concentration. It is only used to locate the 1% boundary.

STEP 2 — SCAN FROM TOP TO BOTTOM:
Find the point where bulk-use ingredients stop dominating and low-dose or borderline ingredients begin to dominate in a SUSTAINED way.

STEP 3 — APPLY CONSECUTIVE CLUSTER RULE:
A good 1% boundary appears when you enter a stretch of SEVERAL CONSECUTIVE ingredients that are mostly preservatives, chelators, gums, pH adjusters, fragrance, or extracts typically used below 1%.
Do NOT choose a boundary based on one single ingredient alone.
The consecutive cluster rule ALWAYS takes priority.
Never move the marker earlier based on one single low-dose ingredient alone.

CANDIDATE ANCHOR INGREDIENTS that signal the boundary zone (clues only, not definitive):
phenoxyethanol, chlorphenesin, ethylhexylglycerin, EDTA, carbomer, xanthan gum, gellan gum, citric acid, parfum, sodium benzoate, potassium sorbate

STEP 4 — APPLY DESCENDING ORDER CONSISTENCY RULE:
If you place the marker at position X then ALL ingredients before position X must be estimable at or above 1% in descending order.
If any pre-marker ingredient would need to be estimated below 1% to make sense the marker is too late — move it earlier.
You must NEVER apply a low-dose treatment to a pre-marker ingredient if it breaks descending order.

CRITICAL CONSEQUENCE:
If a pre-marker ingredient gets estimated below a later pre-marker ingredient the logic is broken. The boundary must be moved.

NO PRE-MARKER LOW-DOSE OVERRIDE:
If an ingredient is truly low-dose by nature but appears high in the list one of two things must happen:
- Move the 1% marker earlier
- Assign it a band that still preserves descending order and above-1% plausibility
Example: If Sodium Hyaluronate is at position 4 and marker is at position 9 then Sodium Hyaluronate cannot be estimated at 0.65% if positions 5 to 8 are still pre-marker. That breaks descending order.

ADDITIONAL RULE for EMULSION formulas:
Any ingredient whose primary function is emollient, oil, or spreading agent appearing in positions 3-8 is typically used at 1-5%. Do not place the marker before these ingredients.

STEP 5 — FINAL BOUNDARY DECISION:
Choose the EARLIEST position that satisfies BOTH:
- Transition-zone logic: sustained cluster of low-dose ingredients begins here
- Descending-order consistency: all pre-marker ingredients can plausibly be at or above 1% in descending order

IMPORTANT: Before giving your final answer write out your reasoning explicitly:
- List every ingredient with its classification (bulk/borderline/low-dose)
- Identify where the sustained low-dose cluster begins
- Confirm all pre-marker ingredients can be above 1% in descending order
- Then state your final marker position

OUTPUT: After your reasoning return this exact JSON on the last line. No markdown, no code blocks:
{"marker_position": 10, "marker_ingredient": "XANTHAN GUM"}"""

_LLM2_PRODUCT_TYPE_PLACEHOLDER = "{{ $json.product_type }}"
_LLM2_INGREDIENTS_PLACEHOLDER = "{{ JSON.stringify($json.ingredients) }}"


def render_llm2_user(product_type, ingredients) -> str:
    return LLM2_USER_TEMPLATE.replace(
        _LLM2_PRODUCT_TYPE_PLACEHOLDER,
        str(product_type),
    ).replace(
        _LLM2_INGREDIENTS_PLACEHOLDER,
        compact_json(ingredients),
    )


LLM3_SYSTEM = (
    "You are a cosmetic formulation expert. Return valid JSON array only. "
    "No explanation, no markdown, no code blocks."
)

LLM3_USER_TEMPLATE = """You are a cosmetic formulation expert.

PRODUCT TYPE: {{ $json.product_type }}
MARKER POSITION: {{ $json.marker_position }}
MARKER INGREDIENT: {{ $json.marker_ingredient }}

INGREDIENT LIST:
{{ JSON.stringify($json.ingredients) }}

YOUR TASK: Classify every ingredient and assign it to a zone.

STEP 1 — CLASSIFY every ingredient as one of three types:

BULK-USE (comfortable above 1%):
- water, aqua and water-based solvents
- glycerin, propanediol, butylene glycol, pentylene glycol and other polyols
- alcohol denat
- emollient esters — any ester used as skin softener or spreading agent
- plant-derived oils and oil fractions
- fatty alcohols
- silicones
- UV filters
- starches used as main texture base such as corn starch or tapioca starch
- plant extracts appearing in position 1-5 that serve as the formula base
- any ingredient whose primary function is emollient, solvent, humectant or occlusive

BORDERLINE (can be above or below 1% depending on formula):
- some emulsifiers
- some vitamin derivatives
- some texture builders
- some light emollients
- arginine in some formulas
- 1,2-hexanediol
- sodium ascorbyl phosphate
- glyceryl stearate
- hydroxyethylcellulose in some formulas

LOW-DOSE (typically below 1%):
- preservatives
- chelators
- carbomer
- xanthan gum
- gellan gum
- fragrance and essential oils
- most extracts
- tocopherol
- sodium hyaluronate in many formulas
- glutathione
- EDTA
- potent claim actives
- citric acid
- potassium sorbate
- sodium benzoate
- phenoxyethanol
- chlorphenesin
- ethylhexylglycerin
- parfum
- microcrystalline cellulose
- cellulose derivatives
- synthetic polymer thickeners

This classification is not the final concentration. It is only used to assign zones correctly.

STEP 2 — ASSIGN ZONE to every ingredient:

ZONE A (pre-marker):
- display_order is LESS THAN marker_position
- Order matters here — concentrations must descend
- All must be plausibly at or above 1%

ZONE B (marker zone):
- The ingredient AT marker_position OR one position before or after marker_position if those ingredients are also borderline or transitioning
- Around 0.5-1.5%
- Can be slightly above or below 1%
- Descending order becomes less reliable here
- Typically 1-3 ingredients around the boundary point

ZONE C (post-marker):
- display_order is GREATER THAN marker_position
- At or below 1%
- Exact order no longer reliably maps to concentration
- Ingredients may be arranged for claim or marketing reasons
- A lower position ingredient can still be at or near 1% if it is borderline
- Example: 1,2-hexanediol can be ~1% even if glutathione or arginine above it are lower because all are in the Zone C section where order is flexible

CRITICAL RULES:
- Zone assignment is based purely on display_order vs marker_position
- Do NOT override zone based on ingredient classification alone
- A bulk ingredient before marker stays Zone A even if you think it should be lower
- A low-dose ingredient after marker stays Zone C
- Never assign a pre-marker ingredient to Zone C just because it is low-dose by nature

Return ONLY a valid JSON array. No explanation, no markdown, no code blocks. Every ingredient must appear exactly once:
[
  {"ingredient_name": "AQUA/WATER/EAU", "display_order": 1, "classification": "bulk", "zone": "A"},
  {"ingredient_name": "XANTHAN GUM", "display_order": 10, "classification": "low-dose", "zone": "B"}
]"""

_LLM3_PRODUCT_TYPE_PLACEHOLDER = "{{ $json.product_type }}"
_LLM3_MARKER_POSITION_PLACEHOLDER = "{{ $json.marker_position }}"
_LLM3_MARKER_INGREDIENT_PLACEHOLDER = "{{ $json.marker_ingredient }}"
_LLM3_INGREDIENTS_PLACEHOLDER = "{{ JSON.stringify($json.ingredients) }}"


def render_llm3_user(
    product_type, marker_position, marker_ingredient, ingredients
) -> str:
    return (
        LLM3_USER_TEMPLATE.replace(
            _LLM3_PRODUCT_TYPE_PLACEHOLDER,
            str(product_type),
        )
        .replace(
            _LLM3_MARKER_POSITION_PLACEHOLDER,
            js_number_str(marker_position),
        )
        .replace(
            _LLM3_MARKER_INGREDIENT_PLACEHOLDER,
            str(marker_ingredient),
        )
        .replace(
            _LLM3_INGREDIENTS_PLACEHOLDER,
            compact_json(ingredients),
        )
    )


LLM4_SYSTEM = (
    "You are a cosmetic formulation expert. Return valid JSON array only. "
    "No explanation, no markdown, no code blocks."
)

LLM4_USER_TEMPLATE = """You are a cosmetic formulation expert.

PRODUCT TYPE: {{ $json.product_type }}
MARKER POSITION: {{ $json.marker_position }}
MARKER INGREDIENT: {{ $json.marker_ingredient }}

CLASSIFIED INGREDIENTS:
{{ JSON.stringify($json.classified_ingredients) }}

YOUR TASK: Build a realistic concentration band [L, U] for every ingredient.
A band is a lower bound L and upper bound U representing the realistic concentration range for that ingredient in this formula.

--- ZONE A INGREDIENTS (pre-marker) ---

CRITICAL CONSTRAINT BEFORE BUILDING ANY ZONE A BAND:
CRITICAL: IGNORE the classification field entirely when building Zone A bands.
Zone A status OVERRIDES classification. If an ingredient is Zone A, it gets
a wide band that supports >1% REGARDLESS of whether it is classified as
low-dose, borderline or bulk-use. Classification only matters for Zone B and Zone C.

First estimate the approximate total of ALL non-base Zone A ingredients combined.
This total MUST be between 20-60% to leave room for the base ingredient.
If your bands would produce a total exceeding 60%, reduce all Zone A bands proportionally before proceeding.
This check is MANDATORY before assigning any individual band.

EXTRACT-HEAVY ZONE A RULE:
If 3 or more consecutive Zone A ingredients are botanical extracts, treat them
as a shared botanical phase. Cap each extract band upper limit at 15% max.
Ensure the sum of ALL extract midpoints in Zone A does not exceed 40%.

- Band must include territory above 1%
- Band must be compatible with product type
- Band must be compatible with ingredient behavior
- Band must preserve descending order across ALL Zone A ingredients — band of ingredient at position N must be lower than band of ingredient at position N-1
- If a pre-marker ingredient is low-dose by nature but appears in Zone A, its band must still be adjusted upward to preserve descending order above 1%

IMPORTANT: All Zone A non-base ingredient bands must be realistic. The sum of ALL non-base ingredients must leave room for the base ingredient. If bands are too high, reduce them. Typical non-base Zone A total should be between 20-60%.

PRODUCT TYPE ADJUSTMENTS FOR ZONE A:
- EMULSION: water 50-80%, humectants like glycerin and butylene glycol 2-8%, emollients 1-8%, emulsifiers 0.5-3%, light oils and silicones 1-5%
- WATER_SERUM: humectants and solvents 2-15%, oils and emollients 0.5-3%, water 60-85%, emulsifiers usually low, polymers low but common
- GEL: gelling agents 0.3-2%, humectants 2-10%, extracts variable
- CLEANSER: surfactants 5-20%, humectants 1-5%, leave-on active logic weaker
- SUNSCREEN: UV filters 2-20%, emollients and solvents 2-10%, very high UV filter concentrations possible
- ANHYDROUS: oils 5-30%, waxes 2-15%, water-soluble ingredients unlikely

--- ZONE B INGREDIENTS (marker zone) ---
- Use a narrow band around 1%
- Typical bands: [0.3, 1.5] or [0.5, 1.2] or [0.6, 1.0]
- Adapt based on ingredient type
- Descending order becomes less reliable here

--- ZONE C INGREDIENTS (post-marker) ---
- Band must stay at or below 1% unless ingredient is a special case that can sit at exactly ~1%
- Exact list order is weak evidence — do NOT force descending order here
- Typical use behavior matters more than position
- Special case: borderline ingredients like 1,2-hexanediol can sit at exactly ~1% even if lower positioned ingredients are lower because Zone C order is flexible
- A borderline ingredient in Zone C can be up to 1%
- Preservatives: [0.1, 1.0]
- Chelators: [0.01, 0.1]
- pH adjusters: [0.01, 0.2]
- Fragrance: [0.01, 0.3]
- Gums and polymers: [0.1, 0.8]
- Most extracts: [0.01, 0.5]
- Borderline ingredients in Zone C: [0.1, 1.0]

--- NEIGHBOR NARROWING ---
- Zone A: neighbors are strong constraints — band of ingredient at position N must be below band of ingredient at position N-1
- Zone C: neighbors are weak constraints — order may be flexible, do not force descending order

--- OUTPUT FORMAT ---
Return ONLY a valid JSON array. No explanation, no markdown, no code blocks. Every ingredient must appear exactly once with band_L and band_U as numbers:
[
  {"ingredient_name": "AQUA/WATER/EAU", "display_order": 1, "zone": "A", "band_L": 50.0, "band_U": 80.0},
  {"ingredient_name": "XANTHAN GUM", "display_order": 10, "zone": "B", "band_L": 0.3, "band_U": 1.5}
]"""

_LLM4_PRODUCT_TYPE_PLACEHOLDER = "{{ $json.product_type }}"
_LLM4_MARKER_POSITION_PLACEHOLDER = "{{ $json.marker_position }}"
_LLM4_MARKER_INGREDIENT_PLACEHOLDER = "{{ $json.marker_ingredient }}"
_LLM4_CLASSIFIED_PLACEHOLDER = "{{ JSON.stringify($json.classified_ingredients) }}"


def render_llm4_user(
    product_type, marker_position, marker_ingredient, classified_ingredients
) -> str:
    return (
        LLM4_USER_TEMPLATE.replace(
            _LLM4_PRODUCT_TYPE_PLACEHOLDER,
            str(product_type),
        )
        .replace(
            _LLM4_MARKER_POSITION_PLACEHOLDER,
            js_number_str(marker_position),
        )
        .replace(
            _LLM4_MARKER_INGREDIENT_PLACEHOLDER,
            str(marker_ingredient),
        )
        .replace(
            _LLM4_CLASSIFIED_PLACEHOLDER,
            compact_json(classified_ingredients),
        )
    )
