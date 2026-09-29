"""Generate static/img/favicon.ico from the SVG mark.

Browsers probe /favicon.ico even when an SVG icon is declared in the markup,
so without this every page load logs a 404. Run after editing favicon.svg:

    python manage.py make_favicon
"""

from __future__ import annotations

import struct
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

# 16x16 and 32x32, 32-bit BGRA, drawn to match favicon.svg: an oxblood-free
# navy rounded square with three ascending white bars.
SIZES = (16, 32)

NAVY = (0x1E, 0x40, 0xAF)      # RGB
WHITE = (0xFF, 0xFF, 0xFF)

# Bar geometry as fractions of the icon: (x, y, w, h) in a 0..1 square.
BARS = (
    (0.22, 0.59, 0.16, 0.22),
    (0.42, 0.44, 0.16, 0.37),
    (0.62, 0.25, 0.16, 0.56),
)


def _pixel(size: int, x: int, y: int) -> tuple[int, int, int, int]:
    """Return BGRA for one pixel, with a 2px rounded corner at small sizes."""
    fx, fy = x / size, y / size

    # Rounded corners: drop pixels outside a quarter-circle at each corner.
    radius = 0.18
    for cx, cy in ((radius, radius), (1 - radius, radius), (radius, 1 - radius), (1 - radius, 1 - radius)):
        if (fx < radius and fy < radius) or (fx > 1 - radius and fy < radius) or \
           (fx < radius and fy > 1 - radius) or (fx > 1 - radius and fy > 1 - radius):
            if (fx - cx) ** 2 + (fy - cy) ** 2 > radius**2:
                return (0, 0, 0, 0)  # transparent

    for bx, by, bw, bh in BARS:
        if bx <= fx <= bx + bw and by <= fy <= by + bh:
            r, g, b = WHITE
            break
    else:
        r, g, b = NAVY
    return (b, g, r, 255)  # BGRA


def _bmp_dib(size: int) -> bytes:
    """DIB header + BGRA rows, bottom-up, as an ICO expects."""
    header = struct.pack(
        "<IiiHHIIiiII",
        40,          # biSize
        size,        # biWidth
        size * 2,    # biHeight (image + mask)
        1,           # biPlanes
        32,          # biBitCount
        0,           # biCompression = BI_RGB
        0, 0, 0, 0, 0,
    )
    rows = bytearray()
    for y in range(size - 1, -1, -1):       # bottom-up
        for x in range(size):
            rows += bytes(_pixel(size, x, y))
    # AND mask: 1bpp, rows padded to 4 bytes. All zero = fully opaque.
    mask_row = b"\x00" * (((size + 31) // 32) * 4)
    return header + bytes(rows) + mask_row * size


class Command(BaseCommand):
    help = "Generate a multi-resolution favicon.ico from the SVG mark."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output",
            type=Path,
            default=Path(settings.BASE_DIR) / "static" / "img" / "favicon.ico",
        )

    def handle(self, *args, **options):
        out: Path = options["output"]
        if not (out.parent / "favicon.svg").exists():
            raise CommandError(f"{out.parent / 'favicon.svg'} is missing; nothing to match.")

        images = [_bmp_dib(size) for size in SIZES]
        count = len(images)
        offset = 6 + 16 * count

        ico = bytearray(struct.pack("<HHH", 0, 1, count))
        for size, data in zip(SIZES, images):
            ico += struct.pack(
                "<BBBBHHII",
                size, size,       # width, height (0 means 256)
                0,                # palette
                0,                # reserved
                1, 32,            # planes, bit count
                len(data), offset,
            )
            offset += len(data)
        for data in images:
            ico += data

        out.write_bytes(bytes(ico))
        self.stdout.write(self.style.SUCCESS(f"Wrote {out} ({len(ico)} bytes, {count} sizes)"))
