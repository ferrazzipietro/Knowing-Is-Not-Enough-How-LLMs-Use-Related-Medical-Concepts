#!/usr/bin/env python3
"""Create an Excel summary table from zero-shot evaluation metric files.

Examples:
    python summarize_zero_shot.py
    python summarize_zero_shot.py --input-dir data/zero_shot --output results.xlsx

The workbook contains one row for each completed (model, setting, system prompt)
combination. Every scalar metric is included. Nested metrics, such as per-item
and per-class F1 scores, are flattened into separate columns.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape


IDENTITY_COLUMNS = ("Model", "Setting", "System prompt", "Result directory")
# Add model directory names here to omit their results by default. Command-line
# ``--skip-model`` values are added to this list for a single invocation.
DEFAULT_SKIPPED_MODELS = () # ("gemma-4-E2B-it", "gemma-4-E4B-it", "gemma-4-12B-it")


def read_metrics(path: Path) -> dict[str, Any] | None:
    """Read one metrics JSON object, returning ``None`` when it is invalid."""
    try:
        with path.open(encoding="utf-8") as file:
            metrics = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Warning: skipping {path}: {error}")
        return None
    if not isinstance(metrics, dict):
        print(f"Warning: skipping {path}: expected a JSON object")
        return None
    return metrics


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten every scalar metric value, retaining the full nested key path."""
    flattened: dict[str, Any] = {}

    def visit(key: str, value: Any) -> None:
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                separator = "." if key else ""
                visit(f"{key}{separator}{nested_key}", nested_value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[key] = value
        else:
            flattened[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)

    for key, value in metrics.items():
        visit(str(key), value)
    return flattened


def prompt_label(result_dir: Path, setting_dir: Path) -> str:
    """Use the explicit prompt directory, or label old no-prompt runs clearly."""
    return result_dir.name if result_dir != setting_dir else "default"


def collect_rows(input_dir: Path, skipped_models: set[str] | None = None) -> list[dict[str, Any]]:
    """Collect completed validation/test result pairs under ``input_dir``."""
    skipped_models = set(DEFAULT_SKIPPED_MODELS) if skipped_models is None else skipped_models
    rows: list[dict[str, Any]] = []
    for test_path in sorted(input_dir.rglob("test_metrics.json")):
        result_dir = test_path.parent
        validation_path = result_dir / "validation_metrics.json"
        if not validation_path.is_file():
            print(f"Warning: skipping {result_dir}: validation_metrics.json is missing")
            continue

        try:
            relative_parts = result_dir.relative_to(input_dir).parts
        except ValueError:
            continue
        if len(relative_parts) < 2:
            print(f"Warning: skipping {result_dir}: expected model/setting directory layout")
            continue

        validation_metrics = read_metrics(validation_path)
        test_metrics = read_metrics(test_path)
        if validation_metrics is None or test_metrics is None:
            continue

        model, setting = relative_parts[:2]
        if model in skipped_models:
            continue
        setting_dir = input_dir / model / setting
        row: dict[str, Any] = {
            "Model": model,
            "Setting": setting,
            "System prompt": prompt_label(result_dir, setting_dir),
            "Result directory": result_dir.relative_to(input_dir).as_posix(),
        }
        row.update(flatten_metrics(validation_metrics))
        row.update(flatten_metrics(test_metrics))
        rows.append(row)

    return sorted(
        rows,
        key=lambda row: (
            str(row["Model"]).casefold(),
            str(row["Setting"]),
            str(row["System prompt"]),
        ),
    )


def columns_for(rows: list[dict[str, Any]]) -> list[str]:
    """Return identity columns followed by all flattened metric columns."""
    metric_columns = sorted(
        {column for row in rows for column in row if column not in IDENTITY_COLUMNS},
        key=lambda column: (not column.startswith("validation_"), column.casefold()),
    )
    return [*IDENTITY_COLUMNS, *metric_columns]


def excel_column_name(index: int) -> str:
    """Convert a zero-based index to its spreadsheet column name."""
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def excel_cell(reference: str, value: Any) -> str:
    """Serialize a value as an XLSX cell without treating strings as formulas."""
    if value is None or value == "":
        return f'<c r="{reference}"/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)):
        if not isinstance(value, float) or math.isfinite(value):
            return f'<c r="{reference}"><v>{value}</v></c>'
    return f'<c r="{reference}" t="inlineStr"><is><t>{xml_escape(str(value))}</t></is></c>'


def write_xlsx(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write the summary table as a dependency-free XLSX workbook."""
    columns = columns_for(rows)
    table = [columns] + [[row.get(column) for column in columns] for row in rows]
    last_column = excel_column_name(len(columns) - 1) if columns else "A"
    sheet_rows = []
    for row_number, row in enumerate(table, start=1):
        cells = "".join(
            excel_cell(f"{excel_column_name(column_number)}{row_number}", value)
            for column_number, value in enumerate(row)
        )
        sheet_rows.append(f'<row r="{row_number}">{cells}</row>')

    widths = []
    for column_number, column in enumerate(columns):
        widest = max(len(str(row[column_number] or "")) for row in table)
        width = min(max(widest + 2, 12), 35)
        widths.append(
            f'<col min="{column_number + 1}" max="{column_number + 1}" width="{width}" customWidth="1"/>'
        )

    worksheet = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <cols>{"".join(widths)}</cols>
  <sheetData>{"".join(sheet_rows)}</sheetData>
  <autoFilter ref="A1:{last_column}{len(table)}"/>
</worksheet>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>'''
    root_relationships = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''
    workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Zero-shot summary" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''
    workbook_relationships = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as workbook_file:
        workbook_file.writestr("[Content_Types].xml", content_types)
        workbook_file.writestr("_rels/.rels", root_relationships)
        workbook_file.writestr("xl/workbook.xml", workbook)
        workbook_file.writestr("xl/_rels/workbook.xml.rels", workbook_relationships)
        workbook_file.writestr("xl/worksheets/sheet1.xml", worksheet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/zero_shot"), help="Directory containing zero-shot result folders."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/zero_shot_summary.xlsx"), help="Path for the Excel workbook."
    )
    parser.add_argument(
        "--skip-model",
        action="append",
        default=[],
        metavar="MODEL",
        help="Model directory name to exclude; can be provided more than once.",
    )
    arguments = parser.parse_args()

    if not arguments.input_dir.is_dir():
        parser.error(f"input directory does not exist: {arguments.input_dir}")
    skipped_models = set(DEFAULT_SKIPPED_MODELS) | set(arguments.skip_model)
    rows = collect_rows(arguments.input_dir, skipped_models)
    if not rows:
        parser.error(f"no completed validation/test metric pairs found in {arguments.input_dir}")

    write_xlsx(rows, arguments.output)
    print(f"Wrote {len(rows)} zero-shot result rows to {arguments.output}.")


if __name__ == "__main__":
    main()
