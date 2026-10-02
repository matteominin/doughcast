"""Deterministic, text-only extraction evaluation."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from dough.voice.extract import parse_transcript_for_eval

FIELDS = ("date", "pizzas_sold", "dough_balls_prepared", "dough_balls_left", "sold_out",
          "sold_out_time", "closed", "weather", "events", "notes")


def _json_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def evaluate(cases_path: Path) -> dict[str, Any]:
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = Counter()
    totals = Counter()
    hallucinations = Counter()
    exact = 0
    for case in cases:
        reference_date = date.fromisoformat(case["reference_date"])
        predicted = parse_transcript_for_eval(case["transcript"], reference_date).model_dump()
        expected = case["expected"]
        case_exact = True
        for field in FIELDS:
            actual_value = _json_value(predicted.get(field))
            expected_value = expected.get(field)
            totals[field] += 1
            matches[field] += actual_value == expected_value
            hallucinations[field] += expected_value is None and actual_value is not None
            case_exact &= actual_value == expected_value
        exact += case_exact
    total_fields = len(cases) * len(FIELDS)
    return {
        "mode": "deterministic_text_only",
        "cases": len(cases),
        "fields": {field: {"correct": matches[field], "total": totals[field],
                            "accuracy": matches[field] / totals[field],
                            "hallucinations": hallucinations[field]}
                    for field in FIELDS},
        "exact_match": exact / len(cases) if cases else 0.0,
        "hallucination_rate": sum(hallucinations.values()) / total_fields if total_fields else 0.0,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    report = evaluate(root / "eval" / "extraction_cases.jsonl")
    output = root / "reports" / "extraction_eval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()