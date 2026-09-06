"""Structured report and dataset contracts for Google Workspace artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


OPERATIONAL_LINE_PATTERN = re.compile(
    r"(?i)(?:"
    r"(?:do not have|lack|without|no)\s+(?:the\s+)?(?:tool|permission|access|api)|"
    r"(?:cannot|can't|unable to)\s+(?:create|save|access|use).{0,60}(?:google|docs|sheets|image|workspace|project)|"
    r"(?:tool|api|workspace|project).{0,40}(?:permission|unavailable|not available|required)|"
    r"(?:권한|도구|API|workspace|워크스페이스|경로).{0,50}(?:없|부족|필요|요구)|"
    r"(?:만들|생성|저장|수행).{0,30}(?:수 없|불가)|"
    r"(?:다음|별도).{0,20}(?:agent|에이전트).{0,30}(?:필요|수행)"
    r")"
)
TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")
HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
UNORDERED_PATTERN = re.compile(r"^\s*[-*+]\s+(.+)$")
ORDERED_PATTERN = re.compile(r"^\s*\d+[.)]\s+(.+)$")
BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
COLUMN_TYPES = frozenset({"text", "integer", "decimal", "percent", "currency", "date"})
NUMERIC_COLUMN_TYPES = frozenset({"integer", "decimal", "percent", "currency"})
CHART_TYPES = frozenset({"LINE", "BAR", "COLUMN", "PIE"})


def sanitize_report(content: str) -> str:
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept: list[str] = []
    for line in lines:
        if OPERATIONAL_LINE_PATTERN.search(line):
            continue
        kept.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


@dataclass(frozen=True)
class TextRange:
    start: int
    end: int


@dataclass(frozen=True)
class ParagraphFormat:
    start: int
    end: int
    style: str
    list_kind: Literal["bullet", "number"] | None = None


@dataclass(frozen=True)
class MarkdownTable:
    placeholder_index: int
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class DocsRenderPlan:
    text: str
    paragraphs: tuple[ParagraphFormat, ...]
    bold: tuple[TextRange, ...]
    tables: tuple[MarkdownTable, ...]


def _table_cells(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def render_markdown_report(content: str) -> DocsRenderPlan:
    lines = sanitize_report(content).split("\n")
    output: list[str] = []
    paragraphs: list[ParagraphFormat] = []
    bold_ranges: list[TextRange] = []
    tables: list[MarkdownTable] = []
    index = 1
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        if (
            "|" in line and line_index + 1 < len(lines)
            and TABLE_SEPARATOR_PATTERN.fullmatch(lines[line_index + 1])
        ):
            rows = [_table_cells(line)]
            line_index += 2
            while line_index < len(lines) and "|" in lines[line_index] and lines[line_index].strip():
                rows.append(_table_cells(lines[line_index]))
                line_index += 1
            width = max(len(row) for row in rows)
            normalized = tuple(tuple((*row, *("" for _ in range(width - len(row))))) for row in rows)
            output.append("\n")
            tables.append(MarkdownTable(index, normalized))
            index += 1
            continue

        heading = HEADING_PATTERN.match(line)
        unordered = UNORDERED_PATTERN.match(line)
        ordered = ORDERED_PATTERN.match(line)
        if heading:
            raw_text = heading.group(2)
            style = f"HEADING_{len(heading.group(1))}"
            list_kind = None
        elif unordered:
            raw_text = unordered.group(1)
            style = "NORMAL_TEXT"
            list_kind = "bullet"
        elif ordered:
            raw_text = ordered.group(1)
            style = "NORMAL_TEXT"
            list_kind = "number"
        else:
            raw_text = line
            style = "NORMAL_TEXT"
            list_kind = None

        clean_parts: list[str] = []
        cursor = 0
        clean_length = 0
        for match in BOLD_PATTERN.finditer(raw_text):
            prefix = raw_text[cursor:match.start()]
            value = match.group(1)
            clean_parts.extend((prefix, value))
            clean_length += len(prefix)
            bold_ranges.append(TextRange(index + clean_length, index + clean_length + len(value)))
            clean_length += len(value)
            cursor = match.end()
        clean_parts.append(raw_text[cursor:])
        clean = "".join(clean_parts)
        rendered = clean + "\n"
        output.append(rendered)
        if clean:
            paragraphs.append(ParagraphFormat(index, index + len(rendered), style, list_kind))
        index += len(rendered)
        line_index += 1
    return DocsRenderPlan("".join(output), tuple(paragraphs), tuple(bold_ranges), tuple(tables))


@dataclass(frozen=True)
class ChartCandidate:
    title: str
    chart_type: str
    category_column: int
    series_columns: tuple[int, ...]


@dataclass(frozen=True)
class TabularDataset:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    column_types: tuple[str, ...]
    units: tuple[str | None, ...]
    sources: tuple[str, ...]
    chart_candidates: tuple[ChartCandidate, ...]

    @classmethod
    def from_mapping(cls, value: object) -> "TabularDataset":
        if not isinstance(value, dict):
            raise ValueError("dataset must be an object")
        title = value.get("title")
        columns = value.get("columns")
        rows = value.get("rows")
        column_types = value.get("column_types")
        units = value.get("units", [])
        sources = value.get("sources")
        candidates = value.get("chart_candidates", [])
        if not isinstance(title, str) or not title.strip() or len(title) > 300:
            raise ValueError("dataset title is invalid")
        if not isinstance(columns, list) or not 2 <= len(columns) <= 50 or any(
            not isinstance(item, str) or not item.strip() for item in columns
        ):
            raise ValueError("dataset columns are invalid")
        if not isinstance(column_types, list) or len(column_types) != len(columns) or any(
            item not in COLUMN_TYPES for item in column_types
        ):
            raise ValueError("dataset column_types are invalid")
        if not isinstance(units, list) or len(units) not in {0, len(columns)} or any(
            item is not None and not isinstance(item, str) for item in units
        ):
            raise ValueError("dataset units are invalid")
        if not isinstance(sources, list) or not sources or any(not isinstance(item, str) or not item.strip() for item in sources):
            raise ValueError("dataset sources are required")
        source_columns = [index for index, name in enumerate(columns) if re.search(r"(?i)source|evidence|출처|근거", name)]
        if not source_columns:
            raise ValueError("dataset requires a source or evidence column")
        if not isinstance(rows, list) or not 2 <= len(rows) <= 499:
            raise ValueError("dataset requires at least two data rows")
        normalized_rows: list[tuple[object, ...]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) != len(columns):
                raise ValueError("dataset row width is invalid")
            normalized: list[object] = []
            for cell, column_type in zip(row, column_types):
                if cell is None:
                    normalized.append(None)
                elif column_type in NUMERIC_COLUMN_TYPES:
                    if isinstance(cell, bool) or not isinstance(cell, (int, float)):
                        raise ValueError("numeric dataset cells must be JSON numbers or null")
                    normalized.append(cell)
                elif not isinstance(cell, str):
                    raise ValueError("text and date dataset cells must be strings or null")
                else:
                    normalized.append(cell[:2_000])
            normalized_rows.append(tuple(normalized))
        parsed_candidates: list[ChartCandidate] = []
        if not isinstance(candidates, list):
            raise ValueError("chart_candidates must be an array")
        for candidate in candidates[:4]:
            if not isinstance(candidate, dict):
                raise ValueError("chart candidate is invalid")
            chart_type = str(candidate.get("chart_type", "")).upper()
            category = candidate.get("category_column")
            series = candidate.get("series_columns")
            candidate_title = candidate.get("title")
            if (
                chart_type not in CHART_TYPES or not isinstance(candidate_title, str) or not candidate_title.strip()
                or isinstance(category, bool) or not isinstance(category, int) or not 0 <= category < len(columns)
                or not isinstance(series, list) or not series
                or any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < len(columns) for item in series)
            ):
                raise ValueError("chart candidate is invalid")
            parsed_candidates.append(ChartCandidate(candidate_title[:300], chart_type, category, tuple(dict.fromkeys(series))))
        return cls(
            title.strip(), tuple(item.strip()[:300] for item in columns), tuple(normalized_rows),
            tuple(column_types), tuple(units or [None] * len(columns)),
            tuple(item.strip()[:500] for item in sources), tuple(parsed_candidates),
        )

    def values(self) -> list[list[object]]:
        headers = [
            f"{column} ({unit})" if unit else column
            for column, unit in zip(self.columns, self.units)
        ]
        return [headers, *(list(row) for row in self.rows)]

    def valid_chart(self, candidate_index: int = 0) -> ChartCandidate | None:
        if not 0 <= candidate_index < len(self.chart_candidates):
            return None
        candidate = self.chart_candidates[candidate_index]
        if (
            self.column_types[candidate.category_column] != "text"
            or candidate.category_column != 0
            or candidate.series_columns != tuple(range(1, 1 + len(candidate.series_columns)))
        ):
            return None
        numeric_series = tuple(
            column for column in candidate.series_columns
            if self.column_types[column] in NUMERIC_COLUMN_TYPES
            and sum(isinstance(row[column], (int, float)) and not isinstance(row[column], bool) for row in self.rows) >= 2
        )
        if not numeric_series or any(not str(row[candidate.category_column] or "").strip() for row in self.rows):
            return None
        return ChartCandidate(candidate.title, candidate.chart_type, candidate.category_column, numeric_series)

    def chart_values(self, candidate: ChartCandidate) -> list[list[object]]:
        columns = (candidate.category_column, *candidate.series_columns)
        return [[self.columns[index] for index in columns], *(
            [row[index] for index in columns] for row in self.rows
        )]