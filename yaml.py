"""Tiny YAML compatibility shim for environments without PyYAML.

It supports the simple mapping/list/inline forms used by this repository's
configs and OpenAPI smoke tests. Docker installs real PyYAML from requirements.
"""
from __future__ import annotations

import ast
import json
from typing import Any



def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in {'\"', "'"}:
            quote = ch
        elif ch in '[{(':
            depth += 1
        elif ch in ']})':
            depth -= 1
        elif ch == ',' and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]

def _scalar(value: str) -> Any:
    value = value.strip()
    if value == '':
        return None
    if value in {'true', 'True'}:
        return True
    if value in {'false', 'False'}:
        return False
    if value in {'null', 'None', '~'}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_scalar(part.strip()) for part in _split_top_level(inner)]
    if value.startswith('{') and value.endswith('}'):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        out: dict[str, Any] = {}
        for part in _split_top_level(inner):
            key, val = part.split(':', 1)
            out[key.strip().strip('"\'')] = _scalar(val.strip())
        return out
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == '#' and not in_single and not in_double:
            return line[:i]
    return line


def _prepare(text: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for raw in text.splitlines():
        clean = _strip_comment(raw).rstrip()
        if not clean.strip():
            continue
        rows.append((len(clean) - len(clean.lstrip(' ')), clean.strip()))
    return rows


def _parse_block(rows: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(rows):
        return {}, index
    is_list = rows[index][1].startswith('- ')
    if is_list:
        out: list[Any] = []
        while index < len(rows):
            cur_indent, text = rows[index]
            if cur_indent < indent or not text.startswith('- '):
                break
            item = text[2:].strip()
            if not item:
                value, index = _parse_block(rows, index + 1, cur_indent + 2)
                out.append(value)
                continue
            if ':' in item and not item.startswith(('"', "'")):
                key, val = item.split(':', 1)
                obj: dict[str, Any] = {key.strip(): _scalar(val.strip()) if val.strip() else {}}
                index += 1
                while index < len(rows) and rows[index][0] > cur_indent:
                    child_indent, child = rows[index]
                    if ':' not in child:
                        break
                    ckey, cval = child.split(':', 1)
                    if cval.strip():
                        obj[ckey.strip()] = _scalar(cval.strip())
                        index += 1
                    else:
                        parsed, index = _parse_block(rows, index + 1, child_indent + 2)
                        obj[ckey.strip()] = parsed
                out.append(obj)
            else:
                out.append(_scalar(item))
                index += 1
        return out, index

    out: dict[str, Any] = {}
    while index < len(rows):
        cur_indent, text = rows[index]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            index += 1
            continue
        if ':' not in text:
            index += 1
            continue
        key, val = text.split(':', 1)
        key = key.strip().strip('"\'')
        val = val.strip()
        if val:
            out[key] = _scalar(val)
            index += 1
        else:
            parsed, index = _parse_block(rows, index + 1, cur_indent + 2)
            out[key] = parsed
    return out, index


def safe_load(text: str) -> Any:
    text = text or ''
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith('{') or stripped.startswith('['):
        return json.loads(stripped)
    rows = _prepare(text)
    parsed, _ = _parse_block(rows, 0, rows[0][0] if rows else 0)
    return parsed
