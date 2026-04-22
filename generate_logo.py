"""Generate GH Buddy SVG and PNG logos with interlocked G and H."""

from __future__ import annotations

from pathlib import Path

import svgwrite
from PIL import Image, ImageDraw, ImageFont


def generate_svg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dwg = svgwrite.Drawing(str(path), size=("420px", "220px"))
    dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), rx=20, ry=20, fill="#f7f9fc"))

    # Interlocked letter design: a bold G ring with H crossing through.
    dwg.add(dwg.circle(center=(135, 110), r=62, fill="none", stroke="#1f77b4", stroke_width=22))
    dwg.add(dwg.line(start=(135, 110), end=(188, 110), stroke="#1f77b4", stroke_width=22))

    dwg.add(dwg.line(start=(170, 52), end=(170, 168), stroke="#ff7f0e", stroke_width=18))
    dwg.add(dwg.line(start=(225, 52), end=(225, 168), stroke="#ff7f0e", stroke_width=18))
    dwg.add(dwg.line(start=(170, 110), end=(225, 110), stroke="#ff7f0e", stroke_width=18))

    dwg.add(
        dwg.text(
            "GH Buddy",
            insert=(265, 118),
            fill="#0f172a",
            style="font-size:36px;font-family:Arial;font-weight:700",
        )
    )
    dwg.save()


def generate_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (420, 220), "#f7f9fc")
    draw = ImageDraw.Draw(img)

    draw.ellipse((73, 48, 197, 172), outline="#1f77b4", width=20)
    draw.line((135, 110, 188, 110), fill="#1f77b4", width=20)
    draw.line((170, 52, 170, 168), fill="#ff7f0e", width=16)
    draw.line((225, 52, 225, 168), fill="#ff7f0e", width=16)
    draw.line((170, 110, 225, 110), fill="#ff7f0e", width=16)

    font = ImageFont.load_default()
    draw.text((265, 105), "GH Buddy", fill="#0f172a", font=font)
    img.save(path)


if __name__ == "__main__":
    generate_svg(Path("logo/gh_logo.svg"))
    generate_png(Path("logo/gh_logo.png"))
    print("Logo files created in logo/ directory.")
