"""Check that FunLib Python source files contain only ASCII characters.

Usage:
    python3 FunLib/validation/check_unicode.py            # scan all FunLib .py files
    python3 FunLib/validation/check_unicode.py path/a.py  # scan specific files
"""

import sys
import unicodedata
from pathlib import Path


def check_file(path: Path) -> list[tuple[int, int, str]]:
    """Return list of (line, col, char) for every non-ASCII character."""
    hits = []
    try:
        text = path.read_bytes()
    except OSError as e:
        print(f"  ERROR reading {path}: {e}", file=sys.stderr)
        return hits
    for lineno, raw_line in enumerate(text.split(b"\n"), 1):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            line = raw_line.decode("latin-1")
        for col, ch in enumerate(line, 1):
            if ord(ch) > 127:
                hits.append((lineno, col, ch))
    return hits


def collect_files(roots: list[str]) -> list[Path]:
    paths = []
    for root in roots:
        p = Path(root)
        if p.is_file():
            paths.append(p)
        else:
            paths.extend(sorted(p.rglob("*.py")))
    return paths


def main(argv: list[str]) -> int:
    if argv:
        files = collect_files(argv)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        files = collect_files([str(repo_root / "FunLib")])

    total_hits = 0
    for path in files:
        hits = check_file(path)
        if hits:
            rel = (
                path.relative_to(Path.cwd())
                if path.is_relative_to(Path.cwd())
                else path
            )
            for lineno, col, ch in hits:
                name = unicodedata.name(ch, f"U+{ord(ch):04X}")
                print(f"  {rel}:{lineno}:{col}  U+{ord(ch):04X}  {name!r}  {ch!r}")
            total_hits += len(hits)

    if total_hits:
        print(
            f"\n  {total_hits} non-ASCII character(s) found -- replace with ASCII equivalents"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
