"""
Annotate each PNG with a small corner label (bottom-right).
Uses PIL/Pillow: white text on a dark semi-transparent background box.
"""

from PIL import Image, ImageDraw, ImageFont
import os

labels = {
    "01_poloidal_contours.png":  "01 \u2013 Poloidal Contours",
    "02_axial_contours.png":     "02 \u2013 Axial Contours",
    "03_radial_profiles.png":    "03 \u2013 Radial Profiles",
    "04_particle_orbits.png":    "04 \u2013 Particle Orbits",
    "05_toroidal_sections.png":  "05 \u2013 Toroidal Sections",
    "06_transport_hierarchy.png":"06 \u2013 Transport Hierarchy",
    "07_energy_balance.png":     "07 \u2013 Energy Balance",
}

base = r"C:\Users\sudho\Desktop\ICTS Simulation\tokamak-sim"

# Try to load a reasonably nice font; fall back to default
def load_font(size=28):
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    # Pillow built-in bitmap font (no size arg)
    return ImageFont.load_default()

font = load_font(28)

for fname, label in labels.items():
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        print(f"  [WARN] Not found: {fname}")
        continue

    img = Image.open(path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Measure text
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    W, H = img.size
    pad_x, pad_y = 8, 6
    margin = 20

    x = W - tw - margin - pad_x
    y = H - th - margin - pad_y

    # Dark semi-transparent background box (RGBA)
    box_x0 = x - pad_x
    box_y0 = y - pad_y
    box_x1 = x + tw + pad_x
    box_y1 = y + th + pad_y
    draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=(20, 20, 20, 200))

    # White text
    draw.text((x, y), label, fill=(255, 255, 255, 255), font=font)

    # Composite over original
    combined = Image.alpha_composite(img, overlay).convert("RGB")
    combined.save(path)
    print(f"  Annotated: {fname}")

print("Done.")
