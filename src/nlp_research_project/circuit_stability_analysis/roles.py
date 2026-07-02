from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .inventory import TokenRow, load_inventory

ROLE_COLUMNS = [
    "wave",
    "prompt_id",
    "label",
    "trajectory_name",
    "generated_index",
    "region",
    "span_type",
    "token_group",
    "fine_role",
    "tags",
    "confidence",
    "role_notes",
    "analysis_group",
    "span_id",
    "char_start",
    "char_end",
    "rule_ids",
    "review_priority",
    "target_token_text",
]

STOP = "<end_of_turn>"

# Prefer an explicit final-answer marker over earlier answer-summary paragraphs.
FINAL_ANSWER_RE = re.compile(r"(?i)(?:\*\*\s*)?final\s+answer\s*:(?:\*\*)?")
FALLBACK_ANSWER_RE = re.compile(r"(?i)(?:\*\*\s*)?answer\s*:(?:\*\*)?|answer\s+is\s+")

OPENING_LINE_RE = re.compile(
    r"(?ix)^\s*(?:"
    r"let['’]s\b|here['’]s\b|okay\b|ok\b|sure\b|to\s+solve\b|"
    r"we\s+need\b|i\s+need\b|i['’]ll\b|first,?\s+let['’]s\b"
    r")"
)
LIST_LINE_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
LIST_MARKER_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]?|[-*•])\s*$")
STEP_HEADER_RE = re.compile(r"^\s*(?:\d+[.)]\s*)?(?:\*\*)?[^\n]{0,90}?:\*\*")
MATH_LINE_RE = re.compile(
    r"\d\s*(?:[+\-*/=×÷]|\\times|\\div|times|plus|minus|divided|equals?)"
    r"|(?:[+\-*/=×÷]|\\times|\\div|=)\s*[$\\]?\d"
    r"|\$\s*\d"
    r"|\\frac\b",
    re.I,
)
WORK_CUE_RE = re.compile(
    r"(?i)^\s*(?:in\s+\d+|now\b|next\b|then\b|therefore\b|thus\b|so\b|"
    r"finally\b|the\s+total\b|total\b)"
)
CONCLUSION_CUE_RE = re.compile(
    r"(?i)\b(?:total|therefore|thus|so,?|difference|remaining|left|final|answer|"
    r"made|cost|earnings?|income|profit|loss)\b"
)

NUMBER_RE = re.compile(
    r"^\s*[$+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?(?:%)?\s*$"
)
NUMBER_PART_RE = re.compile(r"^\s*\d+\s*$")
OPERATOR_RE = re.compile(
    r"^\s*(?:[+\-*/=×÷]|\\times|\\div|times|plus|minus|divided|equals?)\s*$",
    re.I,
)
PUNCT_RE = re.compile(r"^\s*[,.:;!?(){}\[\]]+\s*$")
UNIT_WORD_RE = re.compile(
    r"^\s*(?:hours?|hrs?|miles?|days?|weeks?|months?|years?|quarts?|gallons?|"
    r"cars?|chickens?|shirts?|suits?|pants?|dollars?|cents?|percent|%)\s*$",
    re.I,
)
OPERATION_WORD_RE = re.compile(
    r"(?i)calculate|total|difference|left|remaining|rate|sum|multiply|divide|"
    r"subtract|add|earn|cost|made|saved|lost"
)
LATEX_COMMAND_PART_RE = re.compile(r"^\s*(?:frac|text|times|div|cdot|left|right)\s*$")


@dataclass(frozen=True)
class LineSpan:
    start: int
    end: int
    text: str
    index: int


@dataclass(frozen=True)
class AnswerMarker:
    start: int
    end: int
    pattern: str


@dataclass(frozen=True)
class RegionPlan:
    answer: AnswerMarker | None
    intro_end: int
    work_start: int | None
    final_calc_start: int | None
    final_calc_end: int | None
    stop_start: int | None


def _line_spans(text: str) -> list[LineSpan]:
    spans: list[LineSpan] = []
    start = 0
    for index, line in enumerate(text.splitlines(keepends=True)):
        end = start + len(line)
        spans.append(LineSpan(start=start, end=end, text=line, index=index))
        start = end
    if not spans or start < len(text):
        spans.append(
            LineSpan(start=start, end=len(text), text=text[start:], index=len(spans))
        )
    return spans


def _find_answer_marker(text: str) -> AnswerMarker | None:
    final_matches = list(FINAL_ANSWER_RE.finditer(text))
    if final_matches:
        match = final_matches[-1]
        return AnswerMarker(match.start(), match.end(), "final_answer")
    fallback_matches = list(FALLBACK_ANSWER_RE.finditer(text))
    if fallback_matches:
        # Only use generic Answer markers when no explicit final-answer marker
        # exists.  Take the last one so earlier explanatory "Answer" paragraphs do
        # not steal the final answer span.
        match = fallback_matches[-1]
        return AnswerMarker(match.start(), match.end(), "generic_answer")
    return None


def _stop_start(rows: list[TokenRow]) -> int | None:
    for row in rows:
        if row.text == STOP or STOP in row.text:
            return row.char_start
    return None


def _is_blank_line(line: LineSpan) -> bool:
    return line.text.strip() == ""


def _line_for_char(lines: list[LineSpan], char_start: int) -> LineSpan:
    for line in lines:
        if line.start <= char_start < line.end:
            return line
    return lines[-1]


def _nonblank_lines_between(
    lines: list[LineSpan], start: int, end: int
) -> list[LineSpan]:
    return [
        line
        for line in lines
        if line.end > start and line.start < end and line.text.strip()
    ]


def _line_has_math(line: str) -> bool:
    return bool(MATH_LINE_RE.search(line))


def _line_has_conclusion_or_quantity(line: str) -> bool:
    return bool(re.search(r"\d", line) and CONCLUSION_CUE_RE.search(line))


def _first_nonblank_line_after(
    lines: list[LineSpan], start: int, end: int
) -> LineSpan | None:
    for line in lines:
        if line.end <= start or line.start >= end:
            continue
        if line.text.strip():
            return line
    return None


def _intro_end(lines: list[LineSpan], end: int) -> int:
    first = _first_nonblank_line_after(lines, 0, end)
    if first is None:
        return 0
    if not OPENING_LINE_RE.search(first.text):
        return 0
    intro_end = first.end
    for line in lines[first.index + 1 :]:
        if line.start >= end:
            break
        if _is_blank_line(line):
            intro_end = line.end
            continue
        break
    return intro_end


def _final_calculation_bounds(
    lines: list[LineSpan], *, answer_start: int | None, stop_start: int | None
) -> tuple[int | None, int | None]:
    end = answer_start if answer_start is not None else stop_start
    if end is None:
        end = lines[-1].end if lines else 0
    candidates = _nonblank_lines_between(lines, 0, end)
    if not candidates:
        return None, None

    chosen: LineSpan | None = None
    for line in reversed(candidates):
        if _line_has_math(line.text) or _line_has_conclusion_or_quantity(line.text):
            chosen = line
            break
    if chosen is None:
        chosen = candidates[-1]

    # Include trailing blank lines up to the answer marker so boundary formatting
    # stays attached to the final-calculation region rather than ordinary work.
    calc_end = end
    return chosen.start, calc_end


def _work_start(
    lines: list[LineSpan],
    *,
    intro_end: int,
    final_calc_start: int | None,
    stop_start: int | None,
) -> int | None:
    end = final_calc_start if final_calc_start is not None else stop_start
    if end is None:
        end = lines[-1].end if lines else 0
    content_lines = _nonblank_lines_between(lines, intro_end, end)
    if not content_lines:
        return None

    # If a clear setup block is separated from later work by a blank line, start
    # work after the first blank block.  This handles variable/rate setup traces
    # such as prompt 709 without token-local boundary switches.
    seen_content = False
    for line in lines:
        if line.end <= intro_end or line.start >= end:
            continue
        if line.text.strip():
            seen_content = True
            continue
        if seen_content:
            after_blank = _first_nonblank_line_after(lines, line.end, end)
            if after_blank is not None:
                return after_blank.start

    for line in content_lines:
        if WORK_CUE_RE.search(line.text):
            return line.start

    for idx, line in enumerate(content_lines):
        if _line_has_math(line.text):
            # Keep preceding pure-given/setup lines in problem_setup.  If the very
            # first content line is already arithmetic, there is no reliable setup
            # span, so work starts immediately after the intro.
            return line.start if idx > 0 else content_lines[0].start

    # No arithmetic body found; treat the first content after intro as work so the
    # whole trajectory does not collapse into setup.
    return content_lines[0].start


def _build_region_plan(
    rows: list[TokenRow], text: str, lines: list[LineSpan]
) -> RegionPlan:
    answer = _find_answer_marker(text)
    stop_at = _stop_start(rows)
    analysis_end = answer.start if answer is not None else stop_at
    if analysis_end is None:
        analysis_end = len(text)
    intro_end = _intro_end(lines, analysis_end)
    final_start, final_end = _final_calculation_bounds(
        lines,
        answer_start=answer.start if answer is not None else None,
        stop_start=stop_at,
    )
    work_start = _work_start(
        lines,
        intro_end=intro_end,
        final_calc_start=final_start,
        stop_start=analysis_end,
    )
    return RegionPlan(
        answer=answer,
        intro_end=intro_end,
        work_start=work_start,
        final_calc_start=final_start,
        final_calc_end=final_end,
        stop_start=stop_at,
    )


def _region_for(row: TokenRow, plan: RegionPlan) -> tuple[str, str]:
    if row.text == STOP or STOP in row.text:
        return "stop", "stop_token"
    if plan.answer is not None and row.char_start >= plan.answer.start:
        return "answer_statement", "answer_region_last_marker"
    if (
        plan.final_calc_start is not None
        and plan.final_calc_end is not None
        and plan.final_calc_start <= row.char_start < plan.final_calc_end
    ):
        return "final_calculation", "final_calculation_line"
    if row.char_start < plan.intro_end:
        return "intro_planning", "intro_opening_line"
    if plan.work_start is not None and row.char_start < plan.work_start:
        return "problem_setup", "setup_before_work_start"
    return "work_step", "work_body"


def _overlaps_dollar_math(
    char_start: int, char_end: int, dollar_spans: list[tuple[int, int]]
) -> bool:
    return any(start < char_end and char_start < end for start, end in dollar_spans)


def _dollar_math_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    positions = [match.start() for match in re.finditer(r"\$", text)]
    i = 0
    while i < len(positions) - 1:
        start = positions[i]
        end = positions[i + 1]
        if "\n" in text[start:end]:
            i += 1
            continue
        content = text[start + 1 : end]
        first = _next_nonspace(text, start + 1)
        next_after_close = _next_nonspace(text, end + 1)
        latexish = bool(re.search(r"\\|[{}]|\\times|\\frac", content))
        arithmeticish = bool(re.search(r"\d.*(?:[+=×÷*/-]).*\d", content))
        variableish = bool(re.fullmatch(r"\s*[A-Za-z]\s*", content))
        # Currency expressions such as "$105 - $90 = $15" have many dollar
        # signs but are not paired LaTeX delimiters.  Do not pair a digit-starting
        # dollar with the next digit-starting dollar unless the enclosed content
        # has explicit LaTeX syntax.
        currency_pair = first.isdigit() and next_after_close.isdigit() and not latexish
        if not currency_pair and (
            latexish or arithmeticish or variableish or not first.isdigit()
        ):
            spans.append((start, end + 1))
            i += 2
        else:
            i += 1
    return spans


def _line_prefix(full: str, line: LineSpan, row: TokenRow) -> str:
    return full[line.start : row.char_end]


def _line_suffix(full: str, row: TokenRow, line: LineSpan) -> str:
    return full[row.char_start : line.end]


def _next_nonspace(full: str, idx: int) -> str:
    match = re.search(r"\S", full[idx:])
    return full[idx + match.start()] if match else ""


def _prev_nonspace(full: str, idx: int) -> str:
    prefix = full[:idx]
    for char in reversed(prefix):
        if not char.isspace():
            return char
    return ""


def _is_list_marker_token(row: TokenRow, full: str, line: LineSpan) -> bool:
    stripped = row.text.strip()
    if stripped not in {".", ")", "-", "*", "•"} and not stripped.isdigit():
        return False
    prefix = _line_prefix(full, line, row)
    normalized_prefix = prefix.lstrip()
    if normalized_prefix.startswith("**"):
        normalized_prefix = normalized_prefix[2:]
    if not LIST_MARKER_PREFIX_RE.fullmatch(normalized_prefix):
        return False
    next_char = _next_nonspace(full, row.char_end)
    if stripped.isdigit() and next_char == ".":
        dot_idx = full.find(".", row.char_end, line.end)
        after_dot = _next_nonspace(full, dot_idx + 1) if dot_idx >= 0 else ""
        return bool(after_dot and not after_dot.isdigit())
    return bool(next_char and (next_char.isalpha() or next_char in "*(["))


def _is_bold_marker(row: TokenRow) -> bool:
    stripped = row.text.strip()
    return stripped in {"**", "***"} or "**" in stripped


def _is_bullet_marker(row: TokenRow, full: str, line: LineSpan) -> bool:
    stripped = row.text.strip()
    if stripped not in {"*", "-", "•"}:
        return False
    prefix_before = full[line.start : row.char_start]
    if prefix_before.strip():
        return False
    next_char = _next_nonspace(full, row.char_end)
    return bool(next_char and next_char.isalpha())


def _is_markdown_or_list_marker(row: TokenRow, full: str, line: LineSpan) -> bool:
    return (
        _is_bold_marker(row)
        or _is_list_marker_token(row, full, line)
        or _is_bullet_marker(row, full, line)
    )


def _is_currency_symbol(row: TokenRow, full: str, in_dollar_math: bool) -> bool:
    stripped = row.text.strip()
    if stripped not in {"$", "US$"}:
        return False
    if in_dollar_math:
        return False
    next_char = _next_nonspace(full, row.char_end)
    return bool(next_char and next_char.isdigit())


def _unit_in_math_context(full: str, row: TokenRow, line: LineSpan) -> bool:
    stripped = row.text.strip()
    if "/" in stripped or stripped == "%":
        return True
    before = full[max(line.start, row.char_start - 10) : row.char_start]
    after = full[row.char_end : min(line.end, row.char_end + 10)]
    before_stripped = before.rstrip()
    if re.search(r"\d\s*$", before):
        return True
    if not before_stripped.endswith("**") and re.search(r"[$%*/=×÷+\-\\]\s*$", before):
        return True
    if re.search(r"^\s*(?:/|[*/=×÷+\-]|per\s+(?:\d|[$]))", after, re.I):
        return True
    return False


def _near_quantity_or_operator(full: str, row: TokenRow, window: int = 12) -> bool:
    local = full[
        max(0, row.char_start - window) : min(len(full), row.char_end + window)
    ]
    return bool(re.search(r"\d|[+\-*/=×÷$%]|\\times|\\frac", local))


def _looks_like_math_context(
    row: TokenRow,
    full: str,
    line: LineSpan,
    *,
    in_dollar_math: bool,
) -> bool:
    stripped = row.text.strip()
    if in_dollar_math:
        return True
    if "\\" in row.text or stripped in {"{", "}"}:
        return True
    if OPERATOR_RE.fullmatch(row.text):
        local_without_token = (
            full[max(line.start, row.char_start - 12) : row.char_start]
            + full[row.char_end : min(line.end, row.char_end + 12)]
        )
        return bool(
            re.search(r"\d|[$%]|\\frac", local_without_token)
        ) or _line_has_math(line.text)
    if NUMBER_RE.fullmatch(row.text) or NUMBER_PART_RE.fullmatch(row.text):
        return _line_has_math(line.text) or _near_quantity_or_operator(full, row)
    if UNIT_WORD_RE.fullmatch(row.text) or "/" in stripped:
        return _unit_in_math_context(full, row, line)
    return False


def _inside_step_header(row: TokenRow, line: LineSpan) -> bool:
    header_match = re.search(r"(?:(?:\*\*)?[^\n]{0,100}?:\*\*)", line.text)
    if header_match is None:
        return False
    header_end = line.start + header_match.end()
    return row.char_start < header_end


def _answer_numeric_token(row: TokenRow, full: str) -> bool:
    stripped = row.text.strip()
    if not stripped:
        return False
    if NUMBER_RE.fullmatch(row.text) or NUMBER_PART_RE.fullmatch(row.text):
        return True
    if stripped in {".", ",", "/"}:
        return (
            _prev_nonspace(full, row.char_start).isdigit()
            and _next_nonspace(full, row.char_end).isdigit()
        )
    if stripped in {"$", "%"}:
        return True
    return False


def _answer_marker_token(row: TokenRow, plan: RegionPlan) -> bool:
    if plan.answer is None:
        return False
    if row.char_start < plan.answer.end:
        return True
    return bool(re.search(r"(?i)final|answer", row.text))


def _analysis_group_for(token_group: str, *, pure_horizontal: bool = False) -> str:
    if pure_horizontal:
        return "excluded_whitespace"
    if token_group in {"reasoning_text"}:
        return "reasoning_prose"
    if token_group in {
        "math_quantity",
        "math_operator",
        "unit_symbol",
        "equation_punctuation",
    }:
        return "math_core"
    if token_group in {"answer_marker", "answer_quantity"}:
        return "answer_core"
    if token_group == "format_marker":
        return "formatting_control"
    if token_group == "sentence_punctuation":
        return "punctuation_control"
    if token_group == "stop_special":
        return "stop_control"
    return "unknown_analysis_group"


def _classify_token(
    row: TokenRow,
    *,
    full: str,
    line: LineSpan,
    plan: RegionPlan,
    dollar_spans: list[tuple[int, int]],
) -> dict[str, object]:
    text = row.text
    stripped = text.strip()
    tags: list[str] = []
    rules: list[str] = []
    notes: list[str] = []
    review_priority = "none"
    confidence = 0.82

    region, region_rule = _region_for(row, plan)
    rules.append(region_rule)

    if region == "stop":
        token_group = "stop_special"
        fine_role = "stop_special"
        span_type = "stop"
        confidence = 1.0
        rules.append("stop_token")
    elif "\n" in text:
        token_group = "format_marker"
        fine_role = "structural_newline"
        span_type = "formatting"
        tags.append("structural_newline")
        confidence = 1.0
        rules.append("newline_format")
    elif text.strip(" \t") == "":
        token_group = "format_marker"
        fine_role = "whitespace"
        span_type = "formatting"
        tags.append("pure_horizontal_whitespace")
        confidence = 1.0
        rules.append("horizontal_whitespace")
        if len(text.expandtabs(2)) >= 2 and (
            row.char_start == line.start
            or full[line.start : row.char_start].strip() == ""
        ):
            tags.append("line_start_indentation")
            rules.append("indentation_detected")
    elif region == "answer_statement" and _answer_marker_token(row, plan):
        token_group = "answer_marker"
        fine_role = "answer_marker"
        span_type = "final_answer_marker"
        tags.append("answer_region")
        confidence = 1.0
        rules.append("answer_marker_token")
    elif region == "answer_statement" and _answer_numeric_token(row, full):
        token_group = "answer_quantity"
        fine_role = "final_answer_number"
        span_type = "final_answer_marker"
        tags.append("answer_region")
        confidence = 1.0
        review_priority = "high"
        rules.append("answer_numeric")
    elif _is_markdown_or_list_marker(row, full, line):
        token_group = "format_marker"
        fine_role = "markdown_bold" if _is_bold_marker(row) else "list_marker"
        span_type = (
            "step_header"
            if _inside_step_header(row, line) or LIST_LINE_RE.search(line.text)
            else "formatting"
        )
        tags.append("markdown_marker")
        confidence = 1.0
        rules.append("markdown_or_list")
    else:
        in_dollar_math = _overlaps_dollar_math(
            row.char_start, row.char_end, dollar_spans
        )
        math_context = _looks_like_math_context(
            row, full, line, in_dollar_math=in_dollar_math
        )
        if _is_currency_symbol(row, full, in_dollar_math):
            token_group = "unit_symbol"
            fine_role = "currency_symbol"
            span_type = "math_expression"
            tags.append("currency")
            confidence = 0.95
            rules.append("currency_symbol")
        elif stripped == "%":
            token_group = "unit_symbol"
            fine_role = "percent_symbol"
            span_type = "math_expression"
            tags.append("percent")
            confidence = 0.95
            rules.append("percent_symbol")
        elif OPERATOR_RE.fullmatch(text) and math_context:
            token_group = "math_operator"
            fine_role = (
                "operator_symbol"
                if re.search(r"[+\-*/=×÷\\]", text)
                else "operator_word"
            )
            span_type = "math_expression"
            confidence = 1.0
            rules.append("operator")
        elif NUMBER_RE.fullmatch(text) or NUMBER_PART_RE.fullmatch(text):
            token_group = "math_quantity"
            fine_role = (
                "given_number" if region == "problem_setup" else "derived_number"
            )
            span_type = "math_expression" if math_context else "intermediate_result"
            confidence = 0.96
            rules.append("numeric")
        elif "\\" in text or (
            in_dollar_math
            and (stripped in {"{", "}"} or LATEX_COMMAND_PART_RE.fullmatch(text))
        ):
            token_group = "equation_punctuation"
            fine_role = (
                "latex_command"
                if "\\" in text or LATEX_COMMAND_PART_RE.fullmatch(text)
                else "latex_brace"
            )
            span_type = "latex_format"
            tags.append("latex")
            confidence = 0.92
            rules.append("latexish")
        elif stripped in {"{", "}"} and math_context:
            token_group = "equation_punctuation"
            fine_role = "latex_brace"
            span_type = "latex_format"
            tags.append("latex")
            confidence = 0.92
            rules.append("latex_brace")
        elif UNIT_WORD_RE.fullmatch(text) and (
            in_dollar_math or _unit_in_math_context(full, row, line)
        ):
            token_group = "unit_symbol"
            fine_role = "unit_symbol"
            span_type = "math_expression"
            tags.append("unit_or_currency")
            confidence = 0.88
            rules.append("unit_symbol_math_context")
        elif PUNCT_RE.fullmatch(text):
            prev_char = _prev_nonspace(full, row.char_start)
            next_char = _next_nonspace(full, row.char_end)
            is_decimal = (
                stripped in {".", ","} and prev_char.isdigit() and next_char.isdigit()
            )
            punctuation_math_context = math_context or _near_quantity_or_operator(
                full, row
            )
            if (
                in_dollar_math
                or is_decimal
                or (
                    punctuation_math_context
                    and stripped in {"(", ")", "[", "]", "{", "}"}
                )
            ):
                token_group = "equation_punctuation"
                fine_role = (
                    "equation_decimal_point" if is_decimal else "equation_punctuation"
                )
                span_type = "math_expression" if not in_dollar_math else "latex_format"
                confidence = 0.86
                rules.append("punctuation_math_context")
            else:
                token_group = "sentence_punctuation"
                fine_role = (
                    "sentence_period" if "." in stripped else "sentence_punctuation"
                )
                span_type = "reasoning_prose"
                confidence = 0.9
                rules.append("punctuation_sentence")
        elif "/" in stripped and math_context:
            token_group = "unit_symbol"
            fine_role = "unit_symbol"
            span_type = "math_expression"
            tags.append("unit_or_currency")
            confidence = 0.88
            rules.append("slash_unit_fragment")
        elif in_dollar_math and stripped in {"$"}:
            token_group = "equation_punctuation"
            fine_role = "latex_delimiter"
            span_type = "latex_format"
            tags.append("latex")
            confidence = 0.9
            rules.append("latex_dollar_delimiter")
        elif LATEX_COMMAND_PART_RE.fullmatch(text) and in_dollar_math:
            token_group = "equation_punctuation"
            fine_role = "latex_command"
            span_type = "latex_format"
            tags.append("latex")
            confidence = 0.88
            rules.append("latex_command_part")
        else:
            token_group = "reasoning_text"
            fine_role = (
                "operation_word" if OPERATION_WORD_RE.search(text) else "entity_word"
            )
            span_type = (
                "step_header" if _inside_step_header(row, line) else "reasoning_prose"
            )
            confidence = 0.82
            rules.append("default_reasoning_text")
            if UNIT_WORD_RE.fullmatch(text):
                tags.append("unit_word_prose")
                rules.append("unit_word_outside_math")

    pure_horizontal = (
        "pure_horizontal_whitespace" in tags and "line_start_indentation" not in tags
    )
    analysis_group = _analysis_group_for(token_group, pure_horizontal=pure_horizontal)
    if region == "answer_statement" and "answer_region" not in tags:
        tags.append("answer_region")
    if region == "final_calculation":
        tags.append("final_calculation_region")
    if confidence < 0.7:
        tags.append("low_confidence")
        review_priority = "high"

    return {
        "region": region,
        "span_type": span_type,
        "token_group": token_group,
        "fine_role": fine_role,
        "tags": tags,
        "confidence": confidence,
        "role_notes": ";".join(notes),
        "analysis_group": analysis_group,
        "span_id": f"{region}:{span_type}",
        "rule_ids": rules,
        "review_priority": review_priority,
    }


def classify_trajectory(rows: list[TokenRow]) -> list[dict[str, str]]:
    full = "".join(row.text for row in rows)
    lines = _line_spans(full)
    plan = _build_region_plan(rows, full, lines)
    dollar_spans = _dollar_math_spans(full)
    out: list[dict[str, str]] = []
    for row in rows:
        line = _line_for_char(lines, row.char_start)
        role = _classify_token(
            row,
            full=full,
            line=line,
            plan=plan,
            dollar_spans=dollar_spans,
        )
        out.append(
            {
                **{
                    k: row.source.get(k, "")
                    for k in [
                        "wave",
                        "prompt_id",
                        "label",
                        "trajectory_name",
                        "generated_index",
                    ]
                },
                "region": str(role["region"]),
                "span_type": str(role["span_type"]),
                "token_group": str(role["token_group"]),
                "fine_role": str(role["fine_role"]),
                "tags": ";".join(str(tag) for tag in role["tags"]),
                "confidence": f"{float(role['confidence']):.2f}",
                "role_notes": str(role["role_notes"]),
                "analysis_group": str(role["analysis_group"]),
                "span_id": str(role["span_id"]),
                "char_start": str(row.char_start),
                "char_end": str(row.char_end),
                "rule_ids": ";".join(str(rule) for rule in role["rule_ids"]),
                "review_priority": str(role["review_priority"]),
                "target_token_text": row.text,
            }
        )
    return out


def _sort_key(
    key: tuple[str, str, str, str],
) -> tuple[str, tuple[int, int | str], str, str]:
    wave, prompt_id, label, trajectory_name = key
    try:
        prompt_key: tuple[int, int | str] = (0, int(prompt_id))
    except ValueError:
        prompt_key = (1, prompt_id)
    return wave, prompt_key, label, trajectory_name


def build_roles(
    tokens_csv: Path, trajectory_texts_jsonl: Path | None, output_dir: Path
) -> dict[str, object]:
    grouped, texts, diagnostics = load_inventory(tokens_csv, trajectory_texts_jsonl)
    if diagnostics:
        raise ValueError(
            "Generated text reconstruction mismatches trajectory_texts_jsonl: "
            + "; ".join(diagnostics[:5])
        )
    roles_dir = output_dir / "roles"
    roles_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        role
        for key in sorted(grouped, key=_sort_key)
        for role in classify_trajectory(grouped[key])
    ]
    roles_path = roles_dir / "roles_regex.csv"
    with roles_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, sum(len(v) for v in grouped.values()), diagnostics)
    (roles_dir / "roles_regex_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_review_packet(roles_dir / "roles_regex_review_packet.md", rows, texts)
    return summary


def summarize(
    rows: list[dict[str, str]], token_count: int, diagnostics: list[str] | None = None
) -> dict[str, object]:
    unknown = sum(
        1
        for row in rows
        if row["region"].startswith("unknown")
        or row["span_type"].startswith("unknown")
        or row["token_group"].startswith("unknown")
    )
    marker_traj = {
        row["trajectory_name"]
        for row in rows
        if "answer_marker_token" in row["rule_ids"]
    }
    answer_region = {
        row["trajectory_name"] for row in rows if row["region"] == "answer_statement"
    }
    trajectories = {row["trajectory_name"] for row in rows}
    final_calc_traj = {
        row["trajectory_name"] for row in rows if row["region"] == "final_calculation"
    }
    return {
        "row_count": len(rows),
        "token_count": token_count,
        "row_count_equals_token_count": len(rows) == token_count,
        "exactly_one_role_per_token": all(
            row.get("region") and row.get("span_type") and row.get("token_group")
            for row in rows
        ),
        "stop_count": sum(1 for row in rows if row["token_group"] == "stop_special"),
        "stop_label_count": dict(
            Counter(
                row["region"] for row in rows if row["token_group"] == "stop_special"
            )
        ),
        "unknown_label_counts": {"any_unknown": unknown},
        "unknown_fraction": unknown / len(rows) if rows else 0.0,
        "answer_statement_missing_for_marker_trajectories": sorted(
            marker_traj - answer_region
        ),
        "answer_quantity_counts_by_trajectory": dict(
            Counter(
                row["trajectory_name"]
                for row in rows
                if row["token_group"] == "answer_quantity"
            )
        ),
        "final_calculation_missing_trajectories": sorted(
            trajectories - final_calc_traj
        ),
        "region_counts": dict(Counter(row["region"] for row in rows)),
        "span_type_counts": dict(Counter(row["span_type"] for row in rows)),
        "token_group_counts": dict(Counter(row["token_group"] for row in rows)),
        "analysis_group_counts": dict(Counter(row["analysis_group"] for row in rows)),
        "review_priority_counts": dict(Counter(row["review_priority"] for row in rows)),
        "diagnostics": diagnostics or [],
    }


def write_review_packet(
    path: Path, rows: list[dict[str, str]], texts: dict[tuple[str, str, str, str], str]
) -> None:
    sample = {
        "486_correct_seed3003_attempt000004",
        "486_wrong_seed3000_attempt000001",
        "709_correct_seed3013_attempt000014",
        "709_wrong_seed3012_attempt000013",
        "401_correct_seed3013_attempt000014",
        "401_wrong_seed3014_attempt000015",
        "877_wrong_seed3005_attempt000006",
    }
    by_traj: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["trajectory_name"] in sample:
            by_traj[row["trajectory_name"]].append(row)
    lines = ["# Regex role review packet", ""]
    for name in sorted(by_traj):
        group = by_traj[name]
        counts = Counter(row["region"] for row in group)
        lines += [
            f"## {name}",
            "",
            f"Region counts: `{dict(counts)}`",
            "",
            "```text",
            next((value for key, value in texts.items() if key[3] == name), ""),
            "```",
            "",
            "|idx|repr|region|span|group|analysis|fine|conf|tags|notes|priority|rules|",
            "|-:|---|---|---|---|---|---|---:|---|---|---|---|",
        ]
        for row in group:
            lines.append(
                f"|{row['generated_index']}|{row['target_token_text']!r}|{row['region']}|{row['span_type']}|{row['token_group']}|{row['analysis_group']}|{row['fine_role']}|{row['confidence']}|{row['tags']}|{row['role_notes']}|{row['review_priority']}|{row['rule_ids']}|"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def freeze_roles(
    output_dir: Path, reviewed: Path | None = None, make_review_iter01: bool = False
) -> None:
    roles_dir = output_dir / "roles"
    src = reviewed or roles_dir / "roles_regex.csv"
    if make_review_iter01:
        dst = roles_dir / "roles_review_iter01.csv"
        with src.open(newline="") as inp, dst.open("w", newline="") as out:
            reader = csv.DictReader(inp)
            fields = list(reader.fieldnames or []) + [
                "review_status",
                "reviewer_notes",
                "source_role_file",
                "review_iteration",
            ]
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                row.update(
                    {
                        "review_status": "unreviewed",
                        "reviewer_notes": "",
                        "source_role_file": str(src),
                        "review_iteration": "iter01",
                    }
                )
                writer.writerow(row)
    else:
        shutil.copyfile(src, roles_dir / "roles_v1.csv")
        with src.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        summary = summarize(rows, len(rows), [])
        (roles_dir / "roles_v1_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
