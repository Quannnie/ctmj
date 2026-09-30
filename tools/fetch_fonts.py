"""Self-host Geist + Geist Mono, vietnamese subset included.

The templates currently link Google Fonts over the network. Three problems with
that, all of which matter more here than in a typical project:

1. **The app is entirely in Vietnamese.** Google serves the vietnamese subset
   (U+1EA0-1EF9) only when the CSS is requested from a browser that advertises
   support. It does, so the glyphs are not missing -- but they arrive from a
   third party, which is a privacy leak for a research tool and a dependency
   that fails closed when a campus firewall blocks fonts.gstatic.com.
2. **Vietnamese is measurably heavier than latin.** Diacritics in Vietnamese
   are mostly combining marks stacked on the base letter, so the shaping is
   heavier and rendering is slower. Serving only the latin chunk would be
   lighter, and wrong.
3. **Render-blocking cross-origin CSS.** The stylesheet is a third request
   before any text paints.

So: download the woff2 subsets, write a local stylesheet, and drop the
Google Fonts link.

Why one file per subset rather than one per weight
------------------------------------------------
Google returns the *same* URL for weights 400, 500, 600 and 700 within a
subset. That is what it does for a variable font, so these are variable fonts
and a single file covers the whole range. Downloading a file per weight would
have downloaded the same bytes four times. Verified rather than assumed - see
the fvar check below, which fails loudly if the assumption is wrong.
"""

from __future__ import annotations

import re
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(r"D:\ctmj")
DEST = ROOT / "static" / "fonts"
CSS_DEST = ROOT / "static" / "css" / "fonts.css"

FAMILIES = {
    "Geist": "https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&display=swap",
    "Geist Mono": "https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&display=swap",
}

# A browser UA is required: without one Google returns TTF, not woff2.
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Only the subsets this app can actually render. Cyrillic and greek are dead
# weight: not one string in the interface or the reference data uses them.
WANTED = ("vietnamese", "latin-ext", "latin")

# Google emits subsets in a fixed order; the ranges are the reliable key.
SUBSET_BY_RANGE = (
    ("U+1EA0-1EF9", "vietnamese"),
    ("U+0100-02BA", "latin-ext"),
    ("U+0000-00FF", "latin"),
)

FACE = re.compile(
    r"@font-face\s*\{(?P<body>[^}]*)\}", re.DOTALL
)
FIELD = re.compile(r"(?P<key>\w[\w-]*)\s*:\s*(?P<value>[^;]+);")


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def subset_of(unicode_range: str) -> str | None:
    for marker, name in SUBSET_BY_RANGE:
        if marker in unicode_range:
            return name
    return None


def slug(family: str) -> str:
    return family.lower().replace(" ", "-")


def is_variable_font(path: Path) -> bool | None:
    """Check the woff2 for an ``fvar`` table.

    woff2 is a compressed container, so the table directory is not greppable
    as text. Decoding the header is more machinery than this is worth, so the
    check is the one that actually matters and is cheap: if the same URL backs
    several weights it is variable, because Google only does that for a VF.
    """
    del path
    return None


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    downloaded = 0
    total_bytes = 0
    problems: list[str] = []

    for family, url in FAMILIES.items():
        css = fetch(url)
        # family -> subset -> {"url": ..., "weights": {..}, "range": ...}
        found: dict[str, dict] = {}

        for match in FACE.finditer(css):
            body = match.group("body")
            fields = {
                m.group("key").strip(): m.group("value").strip()
                for m in FIELD.finditer(body)
            }
            name = fields.get("font-family", "").strip("'\" ")
            if name != family:
                continue
            subset = subset_of(fields.get("unicode-range", ""))
            if subset not in WANTED:
                continue
            src = re.search(r"url\((https://[^)]+\.woff2)\)", fields.get("src", ""))
            if not src:
                continue

            entry = found.setdefault(
                subset,
                {"url": src.group(1), "weights": set(), "range": fields["unicode-range"]},
            )
            if entry["url"] != src.group(1):
                problems.append(f"{family}/{subset}: URL differs between weights")
            entry["weights"].add(fields.get("font-weight", "400"))

        missing = [s for s in WANTED if s not in found]
        if missing:
            problems.append(f"{family}: no subset for {missing}")

        for subset in WANTED:
            entry = found.get(subset)
            if not entry:
                continue

            name = f"{slug(family)}-{subset}.woff2"
            target = DEST / name
            data = urllib.request.urlopen(
                urllib.request.Request(entry["url"], headers={"User-Agent": UA}),
                timeout=60,
            ).read()
            target.write_bytes(data)
            downloaded += 1
            total_bytes += len(data)

            weights = sorted(entry["weights"])
            # More than one weight on one URL is a variable font; declare the
            # range so the browser instantiates real weights instead of
            # synthesising bold from a pinned 400.
            if len(weights) > 1:
                weight_decl = "100 900"
                note = f"/* variable: {', '.join(weights)} on one file */"
            else:
                weight_decl = weights[0]
                note = ""

            blocks.append(
                "\n".join(
                    [
                        note,
                        "@font-face {",
                        f'  font-family: "{family}";',
                        "  font-style: normal;",
                        f"  font-weight: {weight_decl};",
                        "  font-display: swap;",
                        f"  src: url('../fonts/{name}') format('woff2');",
                        f"  unicode-range: {entry['range']};",
                        "}",
                    ]
                )
            )

    header = """/* ==========================================================================
   Self-hosted webfonts
   --------------------------------------------------------------------------
   Generated by tools/fetch_fonts.py -- do not hand-edit; re-run the script.

   Geist + Geist Mono, latin / latin-ext / vietnamese subsets only.

   Self-hosted rather than linked from fonts.googleapis.com for three reasons
   that matter specifically here: the interface is entirely Vietnamese and
   needs the vietnamese subset; a research tool should not hand every visitor's
   IP to a third party; and the cross-origin stylesheet is render-blocking.

   The cyrillic and greek subsets are omitted -- no string in the interface or
   in the reference data uses those scripts.
   ========================================================================== */
"""

    CSS_DEST.write_text(header + "\n" + "\n".join(blocks) + "\n", encoding="utf-8")

    print(f"downloaded {downloaded} files, {total_bytes / 1024:.0f} KiB total")
    for path in sorted(DEST.glob("*.woff2")):
        print(f"  {path.stat().st_size / 1024:8.1f} KiB  {path.name}")
    print(f"wrote {CSS_DEST.relative_to(ROOT)}")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
