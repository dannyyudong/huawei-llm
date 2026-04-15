#!/usr/bin/env python3
import ast
import json
import re
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "result_deepseek_1.5b" / "deepseek-ai_DeepSeek-R1" / "non_live"
DST_DIR = ROOT / "result_deepseek_1.5b_sanitized" / "deepseek-ai_DeepSeek-R1" / "non_live"

FILES = [
    "BFCL_v4_simple_python_result.json",
    "BFCL_v4_simple_java_result.json",
    "BFCL_v4_simple_javascript_result.json",
    "BFCL_v4_multiple_result.json",
    "BFCL_v4_parallel_result.json",
    "BFCL_v4_parallel_multiple_result.json",
    "BFCL_v4_irrelevance_result.json",
]

END_TOKEN_PATTERN = re.compile(r"<｜[^>]*｜>")
CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")


def strip_noise(text: str) -> str:
    s = text or ""
    s = END_TOKEN_PATTERN.sub("", s)
    s = s.replace("</think>", "\n")
    s = s.replace("<think>", "\n")
    s = CODE_FENCE_PATTERN.sub(lambda m: m.group(0).replace("```", ""), s)
    s = s.replace("&&", "\n")
    s = s.replace("\r", "")
    return s.strip()


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch in "_."


def _find_matching_paren(s: str, open_idx: int) -> int:
    depth = 0
    in_str = False
    quote = ""
    escape = False
    for i in range(open_idx, len(s)):
        c = s[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == quote:
                in_str = False
            continue

        if c in ("'", '"'):
            in_str = True
            quote = c
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_calls(text: str) -> List[str]:
    s = text
    out: List[str] = []
    i = 0
    while i < len(s):
        if s[i].isalpha() or s[i] == "_":
            j = i
            while j < len(s) and _is_ident_char(s[j]):
                j += 1
            k = j
            while k < len(s) and s[k].isspace():
                k += 1
            if k < len(s) and s[k] == "(":
                end = _find_matching_paren(s, k)
                if end != -1:
                    cand = s[i : end + 1].strip()
                    out.append(cand)
                    i = end + 1
                    continue
        i += 1
    return out


def is_python_call(expr: str) -> bool:
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    return isinstance(node.body, ast.Call)


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        key = x.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def sanitize_for_category(category_file: str, raw_result: str) -> str:
    cleaned = strip_noise(raw_result)
    calls = dedupe_preserve_order(extract_calls(cleaned))
    valid_calls = [c for c in calls if is_python_call(c)]

    # Keep irrelevant untouched except token cleanup, since it's a non-call task.
    if "irrelevance" in category_file:
        return cleaned

    if category_file in {
        "BFCL_v4_simple_python_result.json",
        "BFCL_v4_simple_java_result.json",
        "BFCL_v4_simple_javascript_result.json",
        "BFCL_v4_multiple_result.json",
    }:
        if valid_calls:
            return valid_calls[0]
        return cleaned

    # parallel/parallel_multiple: keep multiple calls when possible.
    if valid_calls:
        return "\n".join(valid_calls[:8])

    return cleaned


def process_file(file_name: str) -> Tuple[int, int]:
    src = SRC_DIR / file_name
    dst = DST_DIR / file_name
    dst.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    changed = 0

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1

            original = obj.get("result", "")
            sanitized = sanitize_for_category(file_name, original)
            if sanitized != original:
                changed += 1

            obj["result"] = sanitized
            if "reasoning_content" in obj:
                obj["reasoning_content"] = sanitized

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return total, changed


def main() -> None:
    if not SRC_DIR.exists():
        raise SystemExit(f"Source directory not found: {SRC_DIR}")

    summary = {}
    for fn in FILES:
        total, changed = process_file(fn)
        summary[fn] = {"total": total, "changed": changed}

    print("Sanitization done. Summary:")
    for fn in FILES:
        item = summary[fn]
        print(f"- {fn}: {item['changed']}/{item['total']} changed")


if __name__ == "__main__":
    main()
