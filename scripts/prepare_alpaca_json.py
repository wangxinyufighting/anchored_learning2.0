#!/usr/bin/env python3
"""Convert Alpaca-style JSON array into KDFlow's prompt/response JSON.

Input format:
[
  {"instruction": "...", "input": "...", "output": "..."},
  ...
]

Output format:
[
  {"input": "<prompt>", "output": "<answer>"},
  ...
]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty file: {path}")

    if text[0] == "[":
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(data, dict):
        # Accept {"train": [...]} or similar wrappers.
        for key in ("train", "data", "records", "examples"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array or a JSONL file.")

    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Record {i} is not an object: {type(row)}")
    return data


def build_prompt(instruction: str, extra_input: str, template: str) -> str:
    instruction = (instruction or "").strip()
    extra_input = (extra_input or "").strip()

    if template == "simple":
        return f"{instruction}\n\n{extra_input}".strip() if extra_input else instruction

    if template == "alpaca":
        if extra_input:
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context.\n"
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{extra_input}\n\n"
                "### Response:\n"
            )
        return (
            "Below is an instruction that describes a task.\n"
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )

    raise ValueError(f"Unknown template: {template}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--instruction_key", default="instruction")
    parser.add_argument("--input_key", default="input")
    parser.add_argument("--output_key", default="output")
    parser.add_argument("--template", choices=["simple", "alpaca"], default="simple")
    parser.add_argument("--ensure_ascii", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output_file)
    records = load_records(in_path)

    converted = []
    for i, row in enumerate(records):
        if args.instruction_key not in row:
            raise KeyError(f"Record {i} is missing key `{args.instruction_key}`")
        if args.output_key not in row:
            raise KeyError(f"Record {i} is missing key `{args.output_key}`")

        prompt = build_prompt(
            instruction=str(row.get(args.instruction_key, "")),
            extra_input=str(row.get(args.input_key, "") or ""),
            template=args.template,
        )
        answer = str(row.get(args.output_key, ""))
        converted.append({"input": prompt, "output": answer})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(converted, ensure_ascii=args.ensure_ascii, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(converted)} records to {out_path}")


if __name__ == "__main__":
    main()
