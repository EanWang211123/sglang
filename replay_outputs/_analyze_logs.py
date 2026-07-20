import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "analysis.txt"


def get_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            t = block.get("type")
            if t == "text":
                parts.append(block.get("text", ""))
            elif t == "thinking":
                parts.append("[thinking]" + (block.get("thinking") or "")[:300])
            elif t == "tool_use":
                parts.append(f"[tool_use name={block.get('name')}]")
            elif t == "tool_result":
                parts.append("[tool_result]")
            else:
                parts.append(f"[{t}]")
        return "".join(parts)
    return str(content)


def is_systemish(text: str) -> bool:
    s = text.strip()
    return s.startswith("<system-reminder>") or "SessionStart hook" in s[:300]


def analyze_garble(content: str) -> list[str]:
    markers = [
        "电话打完",
        "我造做了",
        "hitssing",
        "deaddrop",
        "方案真相",
        "不送给 NULL",
        "deg deaddrop",
        "取 hitssing",
    ]
    hits = []
    for marker in markers:
        pos = content.find(marker)
        if pos >= 0:
            hits.append(f"  marker {marker!r} at char {pos}")
    return hits


def split_sections(content: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?=^## )", content, flags=re.M)
    out = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        title = part.splitlines()[0][:80]
        out.append((title, part))
    return out


lines: list[str] = []
for name in [
    "000031_20260718021118414349793Bq8xEpGG.json",
    "20260717085444934755845QBsJUv9X.json",
]:
    data = json.loads((ROOT / name).read_text(encoding="utf-8"))
    req = data["request_body"]
    resp = data.get("response_body", {})
    msgs = req.get("messages", [])

    lines.append("=" * 80)
    lines.append(name)
    lines.append(f"messages: {len(msgs)}")

    last_user_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") != "user":
            continue
        text = get_text(msgs[i].get("content"))
        if not is_systemish(text):
            last_user_idx = i
            break

    if last_user_idx is not None:
        lines.append(f"last real user msg index: {last_user_idx + 1}")
        lines.append("last real user msg:")
        lines.append(get_text(msgs[last_user_idx].get("content"))[:4000])

    lines.append("--- last 10 turns ---")
    for i, m in enumerate(msgs[-10:], start=len(msgs) - 9):
        role = m.get("role")
        text = get_text(m.get("content")).replace("\n", " ")[:300]
        lines.append(f"[{i}] {role}: {text}")

    lines.append("--- response ---")
    msg = resp.get("choices", [{}])[0].get("message") or {}
    content = msg.get("content") or ""
    lines.append(f"content length: {len(content)}")
    lines.append(content)
    if msg.get("tool_calls"):
        lines.append("tool_calls:")
        lines.append(json.dumps(msg["tool_calls"], ensure_ascii=False, indent=2))

    garble_hits = analyze_garble(content)
    if garble_hits:
        lines.append("--- garble markers ---")
        lines.extend(garble_hits)
        first_pos = min(content.find(m.strip("'")) for m in garble_hits if content.find(m.split()[1].strip("'")) >= 0)
        # simpler: find first weird marker position
        positions = []
        for marker in ["电话打完", "我造做了", "hitssing", "deaddrop"]:
            pos = content.find(marker)
            if pos >= 0:
                positions.append(pos)
        if positions:
            pos = min(positions)
            lines.append(f"first garble around char {pos}:")
            lines.append(content[max(0, pos - 120) : pos + 220])

    sections = split_sections(content)
    if sections:
        lines.append("--- response sections ---")
        for title, body in sections:
            weird = any(x in body for x in ["电话打完", "hitssing", "deaddrop", "我造做了"])
            flag = " [GARBLE]" if weird else ""
            lines.append(f"{title}{flag} ({len(body)} chars)")

    lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(OUT)
