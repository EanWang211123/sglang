#!/usr/bin/env python3
"""Render branch dependency and stack diagrams from the registry YAML."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency bootstrap path
    raise SystemExit(
        "PyYAML is required; run: python -m pip install -r requirements.txt"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "branches.yaml"
STACKS_DIR = ROOT / "stacks"
OUTPUT_PATH = ROOT / "BRANCH_GRAPH.md"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _node_id(prefix: str, value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", f"{prefix}_{value}").strip("_")


def _label(*parts: object) -> str:
    return "<br/>".join(html.escape(str(part), quote=True) for part in parts if part)


def _short_sha(value: object) -> str:
    return str(value)[:10] if value else "unknown"


def _stack_documents() -> list[tuple[Path, dict]]:
    return [(path, _load_yaml(path)) for path in sorted(STACKS_DIR.glob("*.yaml"))]


def _base_heads(stacks: list[tuple[Path, dict]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for _path, stack in stacks:
        base = stack.get("base") or {}
        if base.get("branch") and base.get("head_at_rebuild"):
            result[base["branch"]] = str(base["head_at_rebuild"])
    return result


def _render_overview(registry: dict, stacks: list[tuple[Path, dict]]) -> str:
    branches = registry.get("branches") or []
    by_name = {branch["name"]: branch for branch in branches}
    base_heads = _base_heads(stacks)
    bases = sorted({branch["upstream_base"] for branch in branches})

    lines = ["```mermaid", "flowchart TD"]
    for base in bases:
        node = _node_id("base", base)
        lines.append(f'    {node}["{_label(base, _short_sha(base_heads.get(base)))}"]')

    for branch in branches:
        node = _node_id("branch", branch["id"])
        branch_type = branch["type"]
        if branch_type == "community_pr":
            pr = branch.get("upstream_pr") or {}
            kind = f"PR #{pr.get('number', '?')} / {branch.get('status', 'unknown')}"
        else:
            kind = f"Maintained feature / {branch.get('status', 'unknown')}"
        lines.append(
            f'    {node}["{_label(kind, branch["name"], _short_sha(branch.get("remote_head")))}"]'
        )

        if branch_type == "community_pr":
            base_node = _node_id("base", branch["upstream_base"])
            lines.append(f"    {base_node} --> {node}")

        url = (branch.get("upstream_pr") or {}).get("url")
        if not url:
            repository = registry["repository"]
            encoded_branch = quote(branch["name"], safe="/")
            url = f"https://github.com/{repository}/tree/{encoded_branch}"
        lines.append(f'    click {node} "{url}"')

    for branch in branches:
        target = _node_id("branch", branch["id"])
        for dependency in branch.get("dependencies") or []:
            source_branch = by_name.get(dependency["branch"])
            if source_branch is None:
                raise ValueError(
                    f"Unknown dependency {dependency['branch']!r} for {branch['name']!r}"
                )
            source = _node_id("branch", source_branch["id"])
            relation = dependency.get("relation", "depends_on")
            commit = dependency.get("integrated_commit")
            edge_label = relation.replace("_", " ")
            if commit:
                edge_label += f" / {_short_sha(commit)}"
            if relation == "rebased_patch":
                lines.append(f'    {source} -.->|"{edge_label}"| {target}')
            else:
                lines.append(f'    {source} -->|"{edge_label}"| {target}')

    base_nodes = ",".join(_node_id("base", base) for base in bases)
    pr_nodes = ",".join(
        _node_id("branch", branch["id"])
        for branch in branches
        if branch["type"] == "community_pr"
    )
    feature_nodes = ",".join(
        _node_id("branch", branch["id"])
        for branch in branches
        if branch["type"] == "maintained_feature"
    )
    lines.extend(
        [
            "    classDef base fill:#e8eef7,stroke:#607d9b,color:#17202a",
            "    classDef pr fill:#fff3cd,stroke:#b58900,color:#3d3200",
            "    classDef feature fill:#d9f7e8,stroke:#238636,color:#12351d",
        ]
    )
    if base_nodes:
        lines.append(f"    class {base_nodes} base")
    if pr_nodes:
        lines.append(f"    class {pr_nodes} pr")
    if feature_nodes:
        lines.append(f"    class {feature_nodes} feature")
    lines.append("```")
    return "\n".join(lines)


def _render_stack(path: Path, stack: dict) -> str:
    name = stack["name"]
    components = sorted(stack.get("components") or [], key=lambda item: item["order"])
    base = stack.get("base") or {}
    lines = [
        f"## Stack: `{name}`",
        "",
        f"Source: [`{path.name}`](stacks/{path.name})",
        "",
    ]
    lines.extend(["```mermaid", "flowchart LR"])

    previous = _node_id("stack_base", name)
    lines.append(
        f'    {previous}["{_label(base.get("branch"), _short_sha(base.get("head_at_rebuild")))}"]'
    )
    for component in components:
        current = _node_id(f"stack_{name}", str(component["order"]))
        commit = component.get("integrated_commit") or component.get("integrated_head")
        lines.append(
            f'    {current}["{_label(component["id"], component.get("inclusion"), _short_sha(commit))}"]'
        )
        lines.append(f"    {previous} --> {current}")
        if component.get("source_pr"):
            lines.append(f'    click {current} "{component["source_pr"]}"')
        previous = current

    output = _node_id("stack_output", name)
    lines.append(
        f'    {output}["{_label(stack.get("output_branch"), stack.get("status"))}"]'
    )
    lines.append(f"    {previous} --> {output}")
    lines.append("```")
    return "\n".join(lines)


def render(registry: dict, stacks: list[tuple[Path, dict]]) -> str:
    sections = [
        "# Branch Relationship Graph",
        "",
        "<!-- Generated by scripts/render_branch_graph.py; edit the YAML sources instead. -->",
        "",
        f"Registry updated: `{registry.get('last_updated', 'unknown')}`",
        "",
        "Solid arrows are direct bases or ordered stack composition. Dotted arrows are rebased patches.",
        "",
        "## Managed branches",
        "",
        _render_overview(registry, stacks),
    ]
    for path, stack in stacks:
        sections.extend(["", _render_stack(path, stack)])
    return "\n".join(sections) + "\n"


def _remote_head(remote: str, branch: str) -> str:
    result = subprocess.run(
        ["git", "ls-remote", remote, f"refs/heads/{branch}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    if not output:
        raise RuntimeError(f"Remote branch not found: {remote}/{branch}")
    return output.split()[0]


def refresh_heads(registry: dict) -> None:
    heads = {
        branch["name"]: _remote_head(branch.get("remote", "origin"), branch["name"])
        for branch in registry.get("branches") or []
    }
    lines = REGISTRY_PATH.read_text(encoding="utf-8").splitlines()
    current_branch: str | None = None
    for index, line in enumerate(lines):
        if line.startswith("last_updated:"):
            lines[index] = f"last_updated: {dt.date.today().isoformat()}"
        elif line.startswith("    name: "):
            current_branch = line.split(":", 1)[1].strip()
        elif line.startswith("    remote_head: ") and current_branch in heads:
            lines[index] = f"    remote_head: {heads[current_branch]}"
    REGISTRY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if the graph is stale")
    mode.add_argument(
        "--refresh-heads",
        action="store_true",
        help="refresh remote_head values before rendering",
    )
    args = parser.parse_args()

    registry = _load_yaml(REGISTRY_PATH)
    if args.refresh_heads:
        refresh_heads(registry)
        registry = _load_yaml(REGISTRY_PATH)
    content = render(registry, _stack_documents())

    if args.check:
        existing = (
            OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
        )
        if existing != content:
            print(
                f"{OUTPUT_PATH.name} is stale; run {Path(__file__).name}",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT_PATH.name} is up to date")
        return 0

    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
