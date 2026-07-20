import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                parts.append("[tool_result]")
            elif isinstance(block, dict) and block.get("type") == "tool_use":
                parts.append(f"[tool_use:{block.get('name')}]")
        return "".join(parts)
    return str(content)


def find_garble_start(content: str, good_prefix: str) -> int:
    # longest common prefix with a manually verified good portion
    i = 0
    while i < len(content) and i < len(good_prefix) and content[i] == good_prefix[i]:
        i += 1
    return i


files = [
    "000031_20260718021118414349793Bq8xEpGG.json",
    "20260717085444934755845QBsJUv9X.json",
]

for name in files:
    data = json.loads((ROOT / name).read_text(encoding="utf-8"))
    content = data["response_body"]["choices"][0]["message"]["content"]
    print("=" * 80)
    print(name)
    print("total response chars:", len(content))

    # split by markdown headings
    import re

    headings = [(m.start(), m.group()) for m in re.finditer(r"^#{1,3} .+$", content, re.M)]
    print("sections:")
    for idx, (pos, title) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(content)
        body = content[pos:end]
        weird = any(
            x in body
            for x in [
                "电话打完",
                "hitssing",
                "deaddrop",
                "我造做了",
                "登记代码内容",
                "encodenge",
                "g.la",
                "penalized",
                "à`",
            ]
        )
        print(f"  [{pos:4d}] {title.strip()} weird={weird} len={len(body)}")

    # find first user question in this request thread (early)
    msgs = data["request_body"]["messages"]
    for i, m in enumerate(msgs[:30], 1):
        if m.get("role") == "user":
            t = get_text(m.get("content"))
            if t and not t.startswith("<system-reminder>") and len(t) < 500:
                print(f"early user [{i}]: {t[:200]}")
                break

    # last few assistant before response in file1
    if name.startswith("000031"):
        print("last assistant text before response:")
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                t = get_text(m.get("content"))
                if t and not t.startswith("[tool_use"):
                    print(t[:800])
                    break
