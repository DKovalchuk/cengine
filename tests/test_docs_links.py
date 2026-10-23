"""
Documentation links must resolve: README + docs form a bilingual graph.

Every local markdown link ([text](target)) must point at an existing
file or directory, and every in-page anchor must match a heading slug —
a broken link silently breaks navigation (README once linked to docs/
before the directory existed).

External links (http/https/mailto) are not fetched.

Run: pytest tests/test_docs_links.py -v
"""
import os
import re

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def _slugify(heading: str) -> str:
    """GitHub heading anchor: lowercase, drop punctuation, spaces -> dashes.

    Unicode letters (Cyrillic) are kept — GitHub anchors for Russian
    headings render as-is, e.g. "О проекте" -> "о-проекте".
    """
    h = heading.strip().lower()
    h = re.sub(r"[^\w\- ]", "", h, flags=re.UNICODE)
    return h.replace(" ", "-")


def _markdown_files() -> list[str]:
    files = [
        os.path.join(PROJECT_ROOT, "README.md"),
        os.path.join(PROJECT_ROOT, "README.ru.md"),
    ]
    for base, _dirs, names in os.walk(DOCS_DIR):
        for name in sorted(names):
            if name.endswith(".md"):
                files.append(os.path.join(base, name))
    return [f for f in files if os.path.isfile(f)]


def _headings(md_path: str) -> set[str]:
    slugs: set[str] = set()
    with open(md_path, encoding="utf-8") as f:
        for line in f:
            m = HEADING_RE.match(line)
            if m:
                slugs.add(_slugify(m.group(2)))
    return slugs


def _local_links(md_path: str) -> list[str]:
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    out = []
    for target in LINK_RE.findall(text):
        target = target.strip()
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        out.append(target)
    return out


def _collect_links() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for md in _markdown_files():
        rel = os.path.relpath(md, PROJECT_ROOT)
        for target in _local_links(md):
            pairs.append((rel, target))
    return pairs


@pytest.mark.parametrize(
    "md_file,target",
    _collect_links(),
    ids=lambda v: str(v),
)
def test_local_link_resolves(md_file, target):
    """The link target must exist (file, directory or in-page anchor)."""
    source_dir = os.path.dirname(os.path.join(PROJECT_ROOT, md_file))

    if target.startswith("#"):
        anchor = target[1:]
        assert anchor in _headings(os.path.join(PROJECT_ROOT, md_file)), (
            f"{md_file}: anchor #{anchor} matches no heading"
        )
        return

    if "#" in target:
        path_part, anchor = target.split("#", 1)
    else:
        path_part, anchor = target, None

    resolved = os.path.normpath(os.path.join(source_dir, path_part))
    assert os.path.exists(resolved), (
        f"{md_file}: broken link -> {target} "
        f"(no such path: {os.path.relpath(resolved, PROJECT_ROOT)})"
    )

    if anchor is not None:
        if os.path.isdir(resolved):
            target_md = os.path.join(resolved, "README.md")
        else:
            target_md = resolved
        assert os.path.isfile(target_md), (
            f"{md_file}: anchor link {target} points at a non-markdown file"
        )
        assert anchor in _headings(target_md), (
            f"{md_file}: anchor #{anchor} matches no heading in "
            f"{os.path.relpath(target_md, PROJECT_ROOT)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
