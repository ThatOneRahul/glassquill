"""Draw GlassQuill's icons and social image into web/public (run from the repo root).

    .venv/Scripts/python.exe web/scripts/make_icons.py

The mark matches LogoMark in src/components/ui.tsx: a violet pen nib on near-black with an
orange ink dot. Drawn at 4x and downsampled for clean edges.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "public"
INK, LILAC, VIOLET, FLAME = (7, 7, 10), (221, 211, 255), (124, 58, 237), (255, 106, 31)

SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#ddd3ff"/><stop offset="1" stop-color="#7c3aed"/>
  </linearGradient></defs>
  <rect width="64" height="64" rx="16" fill="#07070a"/>
  <path d="M32 9 46 31 32 55 18 31Z" fill="url(#g)"/>
  <path d="M32 31v20" stroke="#07070a" stroke-width="3" stroke-linecap="round"/>
  <circle cx="32" cy="29" r="4" fill="#07070a"/>
  <circle cx="47" cy="50" r="4.5" fill="#ff6a1f"/>
</svg>
"""


def gradient(size):
    """Diagonal lilac -> violet fill, as in the SVG."""
    g = Image.new("RGB", (size, size))
    px = g.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * (size - 1))
            px[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(LILAC, VIOLET))
    return g


def mark(size, pad=0.0, rounded=True):
    """The logo at `size` px. pad = share of the size kept free (for maskable icons)."""
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if rounded:
        d.rounded_rectangle([0, 0, s - 1, s - 1], radius=s * 16 / 64, fill=INK)
    else:
        d.rectangle([0, 0, s, s], fill=INK)
    inner = s * (1 - 2 * pad)
    k, o = inner / 64, s * pad
    P = lambda x, y: (o + x * k, o + y * k)
    nib = Image.new("L", (s, s), 0)
    ImageDraw.Draw(nib).polygon([P(32, 9), P(46, 31), P(32, 55), P(18, 31)], fill=255)
    img.paste(gradient(s), (0, 0), nib)
    d.line([P(32, 31), P(32, 51)], fill=INK, width=round(3 * k))
    for cx, cy, r, col in [(32, 29, 4, INK), (47, 50, 4.5, FLAME)]:
        x, y = P(cx, cy)
        d.ellipse([x - r * k, y - r * k, x + r * k, y + r * k], fill=col)
    return img.resize((size, size), Image.LANCZOS)


def og_image():
    """1200x630 social card: black, white title, orange accent line, the mark."""
    W, H = 1200, 630
    img = Image.new("RGB", (W, H), INK)
    glow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(glow).ellipse([700, -250, 1500, 450], fill=110)
    from PIL import ImageFilter
    img.paste(Image.new("RGB", (W, H), VIOLET), (0, 0), glow.filter(ImageFilter.GaussianBlur(140)))
    img.paste(mark(140), (80, 80), mark(140))
    d = ImageDraw.Draw(img)

    def font(size, bold=True):
        for name in (["arialbd.ttf", "seguisb.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default(size)

    d.text((80, 270), "Human or AI?", font=font(96), fill="white")
    d.text((80, 380), "See the evidence.", font=font(96), fill=FLAME)
    d.text((80, 520), "GlassQuill · explainable AI-text detection", font=font(34, bold=False), fill=(200, 200, 210))
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "favicon.svg").write_text(SVG, encoding="utf-8")
    mark(256).save(OUT / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    mark(180, rounded=False).convert("RGB").save(OUT / "apple-touch-icon.png")   # iOS rounds it
    mark(192).save(OUT / "icon-192.png")
    mark(512).save(OUT / "icon-512.png")
    mark(512, pad=0.1, rounded=False).save(OUT / "icon-maskable-512.png")        # safe zone
    og_image().save(OUT / "og-image.png", optimize=True)
    manifest = {
        "name": "GlassQuill",
        "short_name": "GlassQuill",
        "description": "Explainable detection of AI-generated text.",
        "start_url": "/detect",
        "display": "standalone",
        "background_color": "#07070a",
        "theme_color": "#07070a",
        "icons": [
            {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"},
            {"src": "/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
        ],
    }
    (OUT / "site.webmanifest").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
