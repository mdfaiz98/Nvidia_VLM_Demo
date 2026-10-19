"""Generate the AGX Orin versions of the demo diagrams into docs/diagrams/.

Same generator, skin and layouts as the Qualcomm IQ9 demo's
scripts/gen_cascade_diagrams.py (drawn to the diagram-design skill's rules:
github.com/cathrynlavery/diagram-design - default skin, slide-16x9 preset,
4px grid, orthogonal connectors), so the two boards' diagrams line up side by
side. Only the content differs: Qwen3-VL-4B on the Ampere GPU, YOLOv8n on DLA 0.

    python3 scripts/gen_orin_diagrams.py [output_dir]

Text widths and the 14-character arrow-label limit are asserted while
building, so an over-long label fails here instead of overflowing its box.
"""
import math
import os
import sys
from html import escape

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "docs", "diagrams")

# --- default skin tokens (style-guide.md) ---
PAPER, INK, MUTED, SOFT = "#f5f5f5", "#2d3142", "#4f5d75", "#7a8399"
ACCENT, LINK = "#eb6c36", "#2e5aa8"
KIND = {  # node type -> (fill, stroke, tag stroke, tag text)
    "focal": ("rgba(235,108,54,0.08)", ACCENT, "rgba(235,108,54,0.50)", ACCENT),
    "backend": ("#ffffff", INK, "rgba(45,49,66,0.40)", INK),
    "store": ("rgba(45,49,66,0.05)", MUTED, "rgba(79,93,117,0.50)", MUTED),
    "input": ("rgba(79,93,117,0.10)", SOFT, "rgba(122,131,153,0.40)", SOFT),
}
# 24x24 stroked icons: Tabler Icons (MIT), as shipped in the diagram-design skill's
# primitive-icons.md (database, users, desktop, terminal); camera, movie and cpu are
# Tabler icons in the same style, which the skill's set doesn't include.
ICONS = {
    "camera": ["M5 7h1a2 2 0 0 0 2 -2a1 1 0 0 1 1 -1h6a1 1 0 0 1 1 1a2 2 0 0 0 2 2h1a2 2 0 0 1 2 2v9a2 2 0 0 1 -2 2h-14a2 2 0 0 1 -2 -2v-9a2 2 0 0 1 2 -2",
               "M9 13a3 3 0 1 0 6 0a3 3 0 0 0 -6 0"],
    "movie": ["M4 6a2 2 0 0 1 2 -2h12a2 2 0 0 1 2 2v12a2 2 0 0 1 -2 2h-12a2 2 0 0 1 -2 -2z", "M8 4v16", "M16 4v16",
              "M4 8h4", "M4 16h4", "M4 12h16", "M16 8h4", "M16 16h4"],
    "cpu": ["M5 6a1 1 0 0 1 1 -1h12a1 1 0 0 1 1 1v12a1 1 0 0 1 -1 1h-12a1 1 0 0 1 -1 -1z", "M9 9h6v6h-6z",
            "M3 10h2", "M3 14h2", "M10 3v2", "M14 3v2", "M21 10h-2", "M21 14h-2", "M14 21v-2", "M10 21v-2"],
    "crop": ["M8 5v10a1 1 0 0 0 1 1h10", "M5 8h10a1 1 0 0 1 1 1v10"],
    "database": ["M4 6a8 3 0 1 0 16 0a8 3 0 1 0 -16 0", "M4 6v6a8 3 0 0 0 16 0v-6", "M4 12v6a8 3 0 0 0 16 0v-6"],
    "users": ["M5 7a4 4 0 1 0 8 0a4 4 0 1 0 -8 0", "M3 21v-2a4 4 0 0 1 4 -4h4a4 4 0 0 1 4 4v2", "M16 3.13a4 4 0 0 1 0 7.75", "M21 21v-2a4 4 0 0 0 -3 -3.85"],
    "desktop": ["M3 5a1 1 0 0 1 1 -1h16a1 1 0 0 1 1 1v10a1 1 0 0 1 -1 1h-16a1 1 0 0 1 -1 -1v-10", "M7 20h10", "M9 16v4", "M15 16v4"],
    "terminal": ["M5 7l5 5l-5 5", "M12 19l7 0"],
    "photo": ["M15 8h.01", "M3 6a3 3 0 0 1 3 -3h12a3 3 0 0 1 3 3v12a3 3 0 0 1 -3 3h-12a3 3 0 0 1 -3 -3v-12z",
              "M3 16l5 -5c.928 -.893 2.072 -.893 3 0l5 5", "M14 14l1 -1c.928 -.893 2.072 -.893 3 0l3 3"],
    "cctv": ["M3 4a1 1 0 0 1 1 -1h16a1 1 0 0 1 1 1v2a1 1 0 0 1 -1 1h-16a1 1 0 0 1 -1 -1z", "M8 14a4 4 0 1 0 8 0a4 4 0 1 0 -8 0",
             "M19 7v7a7 7 0 0 1 -14 0v-7", "M12 14l.01 0"],
    "chart": ["M4 19l16 0", "M4 15l4 -6l4 2l4 -5l4 4"],
}


def icon(name, x, y, size, color):
    paths = "".join(f'<path d="{d}"/>' for d in ICONS[name])
    return (f'<g transform="translate({x},{y}) scale({size / 24:g})" fill="none" stroke="{color}" stroke-width="1.5" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths}</g>')


# presentation ramp (output-spec.md)
NAME, SUB, ALABEL, TAG, LEGEND = 16, 12, 12, 8, 11
SANS, MONO = "'Geist', sans-serif", "'Geist Mono', monospace"


def r4(v):
    return int(math.ceil(v / 4.0) * 4)


def w_sans(s, px):
    return len(s) * 0.60 * px


def w_mono(s, px, track=0.0):
    return len(s) * (0.62 + track) * px


class SVG:
    def __init__(self, slug, title, desc, h=720):
        self.slug, self.title, self.desc, self.h = slug, title, desc, h
        self.zones, self.arrows, self.labels, self.nodes, self.extra = [], [], [], [], []

    # ---------- primitives ----------
    def zone(self, x, y, w, h, label):
        lw = r4(w_mono(label, TAG, 0.14) + 12)
        cx = x + w // 2
        self.zones.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="rgba(45,49,66,0.02)" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>'
            f'<rect x="{cx - lw // 2}" y="{y + 4}" width="{lw}" height="12" rx="2" fill="{PAPER}"/>'
            f'<text x="{cx}" y="{y + 13}" fill="rgba(45,49,66,0.45)" font-size="{TAG}" font-family="{MONO}" text-anchor="middle" letter-spacing="0.14em">{escape(label)}</text>'
        )

    def arrow(self, d, kind="muted"):
        stroke, marker, extra = {
            "muted": (MUTED, "arrow", 'stroke-width="1.2"'),
            "link": (LINK, "arrow-link", 'stroke-width="1.2"'),
            "accent": (ACCENT, "arrow-accent", 'stroke-width="1.4"'),
            "dashed": (MUTED, "arrow", 'stroke-width="1" stroke-dasharray="4,3"'),
            "dashed-link": (LINK, "arrow-link", 'stroke-width="1" stroke-dasharray="4,3"'),
            "async": (MUTED, "arrow-open", 'stroke-width="1" stroke-dasharray="4,3"'),
            "plain": (MUTED, None, 'stroke-width="1.2"'),
        }[kind]
        m = f' marker-end="url(#{marker})"' if marker else ""
        self.arrows.append(f'<path d="{d}" fill="none" stroke="{stroke}" {extra}{m}/>')

    def label(self, cx, top, text, color=SOFT, anchor="middle"):
        """Arrow label: opaque mask 16 tall starting at `top`, text 12px mono."""
        assert len(text) <= 14, ("arrow label over 14 chars", text)
        w = r4(w_mono(text, ALABEL, 0.06) + 12)
        x = cx - w // 2 if anchor == "middle" else cx
        tx = cx if anchor == "middle" else cx + w // 2
        self.labels.append(
            f'<rect x="{x}" y="{top}" width="{w}" height="16" rx="2" fill="{PAPER}"/>'
            f'<text x="{tx}" y="{top + 12}" fill="{color}" font-size="{ALABEL}" font-family="{MONO}" text-anchor="middle" letter-spacing="0.06em">{escape(text)}</text>'
        )
        return x, x + w

    def node(self, x, y, w, h, kind, tag, name, subs=(), num=None, header=False):
        fill, stroke, tstroke, ttext = KIND[kind]
        assert w_sans(name, NAME) <= w - 16, (name, w_sans(name, NAME), w)
        for s in subs:
            assert w_mono(s, SUB) <= w - 16, (s, w_mono(s, SUB), w)
        tw = r4(w_mono(tag, TAG, 0.08) + 12)
        cx = x + w // 2
        p = [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{PAPER}"/>',
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"/>',
            f'<rect x="{x + 8}" y="{y + 8}" width="{tw}" height="12" rx="2" fill="transparent" stroke="{tstroke}" stroke-width="0.8"/>',
            f'<text x="{x + 8 + tw / 2:g}" y="{y + 17}" fill="{ttext}" font-size="{TAG}" font-family="{MONO}" text-anchor="middle" letter-spacing="0.08em">{escape(tag)}</text>',
        ]
        if num:
            op = "0.10" if kind == "focal" else "0.06"
            col = f"rgba(235,108,54,{op})" if kind == "focal" else f"rgba(45,49,66,{op})"
            p.append(f'<text x="{x + w - 8}" y="{y + 32}" fill="{col}" font-size="24" font-weight="600" font-family="{MONO}" text-anchor="end">{num}</text>')
        # name + sublabels, block centred in the space under the tag (or at the top for a header node)
        block = NAME + 6 + (len(subs) - 1) * 18 + SUB if subs else NAME
        top = y + 24 if header else y + 20 + (h - 20 - block) / 2
        p.append(f'<text x="{cx}" y="{top + NAME - 3:g}" fill="{INK}" font-size="{NAME}" font-weight="600" font-family="{SANS}" text-anchor="middle">{escape(name)}</text>')
        for i, s in enumerate(subs):
            p.append(f'<text x="{cx}" y="{top + NAME + 6 + i * 18 + SUB - 2:g}" fill="{MUTED}" font-size="{SUB}" font-family="{MONO}" text-anchor="middle">{escape(s)}</text>')
        self.nodes.append("".join(p))

    def inode(self, x, y, w, h, kind, icon_name, name, subs=(), step=None):
        """Node with a left gutter (step badge + icon) and left-aligned text."""
        fill, stroke, _, _ = KIND[kind]
        tx = x + 64
        assert w_sans(name, NAME) <= w - 64 - 12, (name, w_sans(name, NAME), w)
        for s_ in subs:
            assert w_mono(s_, SUB) <= w - 64 - 12, (s_, w_mono(s_, SUB), w)
        p = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{PAPER}"/>',
             f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"/>']
        if step:
            assert h >= 96, ("stepped node needs h>=96 for badge + icon", name)
            p.append(f'<rect x="{x + 16}" y="{y + 16}" width="28" height="28" rx="4" fill="{INK}"/>'
                     f'<text x="{x + 30}" y="{y + 35}" fill="{PAPER}" font-size="{SUB}" font-weight="600" font-family="{MONO}" text-anchor="middle">{step}</text>')
            p.append(icon(icon_name, x + 14, y + 52, 32, stroke if kind == "focal" else MUTED))
        else:
            p.append(icon(icon_name, x + 14, y + (h - 32) // 2, 32, stroke if kind == "focal" else MUTED))
        p.append(f'<text x="{tx}" y="{y + 34}" fill="{INK}" font-size="{NAME}" font-weight="600" font-family="{SANS}">{escape(name)}</text>')
        for i, s_ in enumerate(subs):
            p.append(f'<text x="{tx}" y="{y + 56 + i * 18}" fill="{MUTED}" font-size="{SUB}" font-family="{MONO}">{escape(s_)}</text>')
        self.nodes.append("".join(p))

    def raw(self, s):
        self.extra.append(s)

    def legend(self, items, y=652):
        out = [f'<line x1="40" y1="{y - 8}" x2="1240" y2="{y - 8}" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>',
               f'<text x="40" y="{y + 8}" fill="{MUTED}" font-size="8" font-family="{MONO}" letter-spacing="0.18em">LEGEND</text>']
        x = 40
        iy = y + 24
        for kind, text in items:
            if kind in KIND:
                f, s, _, _ = KIND[kind]
                out.append(f'<rect x="{x}" y="{iy}" width="16" height="12" rx="2" fill="{f}" stroke="{s}" stroke-width="1"/>')
            else:
                st = {"muted": (MUTED, "arrow", "1.2", ""), "link": (LINK, "arrow-link", "1.2", ""),
                      "accent": (ACCENT, "arrow-accent", "1.4", ""), "dashed": (MUTED, "arrow", "1", ' stroke-dasharray="4,3"'),
                      "async": (MUTED, "arrow-open", "1", ' stroke-dasharray="4,3"')}[kind]
                out.append(f'<line x1="{x}" y1="{iy + 6}" x2="{x + 24}" y2="{iy + 6}" stroke="{st[0]}" stroke-width="{st[2]}"{st[3]} marker-end="url(#{st[1]})"/>')
            out.append(f'<text x="{x + 32}" y="{iy + 10}" fill="{MUTED}" font-size="{LEGEND}" font-family="{SANS}">{escape(text)}</text>')
            x += 32 + r4(w_sans(text, LEGEND)) + 36
        assert x - 36 <= 1240, ("legend too wide", x)
        self.extra.append("".join(out))

    def render(self):
        defs = (
            f'<marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/></marker>'
            f'<marker id="arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{ACCENT}"/></marker>'
            f'<marker id="arrow-link" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{LINK}"/></marker>'
            f'<marker id="arrow-open" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polyline points="0 0, 8 3, 0 6" fill="none" stroke="{MUTED}" stroke-width="1.2"/></marker>'
        )
        body = "\n".join(self.zones + self.arrows + self.labels + self.nodes + self.extra)
        return (
            f'<svg viewBox="0 0 1280 {self.h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{self.slug}-title {self.slug}-desc">\n'
            f'<title id="{self.slug}-title">{escape(self.title)}</title>\n'
            f'<desc id="{self.slug}-desc">{escape(self.desc)}</desc>\n'
            f"<defs>{defs}</defs>\n"
            f'<rect width="100%" height="100%" fill="{PAPER}"/>\n{body}\n</svg>'
        )


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --color-paper: #f5f5f5; --color-ink: #2d3142; --color-muted: #4f5d75; --color-accent: #eb6c36;
      --font-sans: 'Geist', system-ui, sans-serif;
      --font-serif: 'Instrument Serif', serif;
      --font-mono: 'Geist Mono', ui-monospace, monospace;
    }}
    body {{ font-family: var(--font-sans); background: var(--color-paper); color: var(--color-ink);
           min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 2.5rem 2rem; }}
    .frame {{ max-width: 1280px; width: 100%; min-width: 0; }}
    .diagram-container {{ width: 100%; overflow-x: auto; }}
    .eyebrow {{ font-family: var(--font-mono); font-size: 0.66rem; font-weight: 500; letter-spacing: 0.18em;
               text-transform: uppercase; color: var(--color-muted); margin-bottom: 0.5rem; }}
    h1 {{ font-family: var(--font-serif); font-size: clamp(1.75rem, 2.6vw + 0.75rem, 2.5rem); font-weight: 400;
          letter-spacing: -0.02em; line-height: 1.15; color: var(--color-ink); margin-bottom: 0.4rem; }}
    .subtitle {{ color: var(--color-muted); font-size: 0.95rem; margin-bottom: 1.25rem; }}
    .subtitle code {{ font-family: var(--font-mono); font-size: 0.85em; }}
    .nav {{ margin-top: 1rem; font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.08em; color: var(--color-muted); }}
    .nav a {{ color: #2e5aa8; text-decoration: none; margin-right: 1.25rem; }}
    svg {{ width: 100%; min-width: 1280px; display: block; }}
    @media print {{
      .diagram-container {{ overflow-x: visible; }}
      svg {{ min-width: 0; }}
    }}
  </style>
</head>
<body>
  <div class="frame">
    <p class="eyebrow">{eyebrow}</p>
    <h1>{h1}</h1>
    <p class="subtitle">{subtitle}</p>
    <div class="diagram-container">
{svg}
    </div>
    <p class="nav"><a href="live-vlm-how-it-works.html">Main demo · How it works</a><a href="cascade-how-it-works.html">Cascade · How it works</a><a href="demo-system.html">Full system</a></p>
  </div>
</body>
</html>
"""


def write(fname, svg, eyebrow, h1, subtitle):
    with open(f"{OUT}/{fname}", "w") as f:
        f.write(PAGE.format(title=escape(svg.title), eyebrow=escape(eyebrow), h1=escape(h1), subtitle=subtitle, svg=svg.render()))


PLATFORM = "ADVANTECH AFE-R750 · JETSON AGX ORIN 32GB"
FOCAL_LEGEND = ("focal", "AI model (GPU / DLA)")

# =====================================================================
# 1. Main demo: how it works (same layout as the IQ9's live-vlm-how-it-works)
# =====================================================================
d = SVG("live-vlm-how-it-works", "How the Live VLM main demo works on the AGX Orin",
        "A camera feed or video file is read on the board; every 30th frame is sent as a JPEG to Qwen3-VL-4B on the "
        "Ampere GPU (Ollama, llama.cpp with CUDA) together with the chosen prompt preset, an answer-length limit and a "
        "512-token cap; the answer streams onto the screen word by word, with TTFT, speed and system stats.")
d.zone(40, 96, 208, 528, "INPUT")
d.zone(288, 96, 600, 528, "ON-DEVICE · " + PLATFORM)
d.zone(920, 96, 320, 528, "RESULTS")
# arrows
d.arrow("M 240,172 H 304")                                                     # camera feed -> sampler
d.arrow("M 240,316 H 260 Q 268,316 268,308 V 212 Q 268,204 276,204 H 304")     # video file -> sampler
d.arrow("M 576,184 H 616")                                                     # sampler -> qwen
d.arrow("M 576,290 H 616")                                                     # prompt -> qwen
d.arrow("M 744,300 V 344")                                                     # qwen -> answer
d.arrow("M 872,420 H 936", "link")                                             # answer -> on screen
d.arrow("M 1080,240 V 344", "link")                                            # stats -> on screen
d.label(560, 252, "PROMPT", SOFT)
d.label(752, 312, "ANSWER", SOFT, anchor="left")
d.label(904, 396, "WS", LINK)
d.label(1088, 284, "+ STATS", LINK, anchor="left")
# input
d.inode(48, 128, 192, 96, "input", "camera", "Camera feed", ["USB · GMSL", "RTSP camera"], step=1)
d.inode(48, 272, 192, 88, "input", "movie", "Video file", ["MP4 upload", "looped"])
# on-device
d.inode(304, 128, 272, 112, "backend", "photo", "Frame sampling", ["read on the board", "every 30th frame", "≤1024 px JPEG"], step=2)
d.inode(616, 128, 256, 172, "focal", "cpu", "Qwen3-VL-4B", ["Instruct · Q4_K_M", "Ollama · llama.cpp", "Ampere GPU · CUDA"], step=4)
d.raw(f'<rect x="628" y="252" width="232" height="32" rx="4" fill="#ffffff" stroke="rgba(235,108,54,0.35)" stroke-width="0.8"/>'
      + icon("terminal", 636, 260, 16, ACCENT)
      + f'<text x="660" y="272" fill="{INK}" font-size="11" font-family="{MONO}">one frame at a time</text>')
# prompt card with the real presets from index.html, grouped as in the UI
d.inode(304, 280, 272, 328, "backend", "terminal", "Prompt", ["preset or custom text", "+ 1–2 sentences · 512 tok"], step=3)
CH = 10  # chip text px
groups = [("SCENE UNDERSTANDING", ["Scene", "Objects", "Activity", "Emotion", "Accessibility", "OCR / text", "Yes / No"]),
          ("MONITORING", ["Safety", "Pool safety", "Fall", "Security", "Retail", "Warehouse", "Phone use"]),
          ("ROBOT NAVIGATION", ["Navigation", "Drive commands"])]
y = 376
for gname, chips in groups:
    d.raw(f'<text x="320" y="{y + 8}" fill="{SOFT}" font-size="{TAG}" font-family="{MONO}" letter-spacing="0.14em">{escape(gname)}</text>')
    y += 14
    x = 320
    for c in chips:
        cw = r4(w_mono(c, CH) + 12)
        if x + cw > 560:
            x, y = 320, y + 24
        default = c == "Retail"
        d.raw(f'<rect x="{x}" y="{y}" width="{cw}" height="20" rx="3" fill="{"rgba(235,108,54,0.08)" if default else "#ffffff"}" '
              f'stroke="{ACCENT if default else "rgba(45,49,66,0.22)"}" stroke-width="0.8"/>'
              f'<text x="{x + cw / 2:g}" y="{y + 14}" fill="{ACCENT if default else INK}" font-size="{CH}" font-family="{MONO}" text-anchor="middle">{escape(c)}</text>')
        x += cw + 6
    y += 32
assert y - 12 + 8 <= 608, ("prompt chips overflow the card", y)
# illustrative answer for the Retail Monitoring preset at a checkout counter
ans = ["Two customers are queuing at", "the checkout while a cashier", "scans items; the floor is", "clear and no safety concerns", "are visible."]
d.raw(f'<rect x="616" y="344" width="256" height="152" rx="6" fill="{PAPER}"/>'
      f'<rect x="616" y="344" width="256" height="152" rx="6" fill="#ffffff" stroke="rgba(45,49,66,0.30)" stroke-width="1"/>'
      f'<text x="628" y="366" font-size="{SUB}" font-family="{MONO}"><tspan fill="{ACCENT}">&gt;</tspan><tspan fill="{SOFT}"> Retail Monitoring</tspan></text>')
for i, ln in enumerate(ans):
    assert w_mono(ln, SUB) <= 256 - 24, ln
    d.raw(f'<text x="628" y="{390 + i * 16}" fill="{INK}" font-size="{SUB}" font-family="{MONO}">{escape(ln)}</text>')
d.raw(f'<text x="628" y="482" fill="{SOFT}" font-size="11" font-family="{MONO}">example · checkout counter</text>')
d.raw(f'<text x="616" y="552" fill="{MUTED}" font-size="14" font-style="italic" font-family="\'Instrument Serif\', serif">'
      f'<tspan x="616" dy="0">The model runs on the board\'s Ampere</tspan>'
      f'<tspan x="616" dy="20">GPU — frames and answers never</tspan>'
      f'<tspan x="616" dy="20">leave the device.</tspan></text>')
# results
d.inode(936, 344, 288, 152, "backend", "desktop", "On-screen answer", ["VLM Output panel", "streamed word by word", "+ text overlay on video"], step=5)
d.inode(936, 128, 288, 112, "store", "chart", "Live metrics", ["TTFT · tok/s · latency", "CPU · GPU · power", "temperatures °C"])
d.legend([FOCAL_LEGEND, ("backend", "Processing / app"), ("input", "Source"),
          ("store", "Data"), ("muted", "On-device flow"), ("link", "Results out")])
write("live-vlm-how-it-works.html", d, "How it works · Advantech AFE-R750 · Jetson AGX Orin 32GB",
      "From a camera frame to a live answer — on one device",
      "Every 30th frame goes to Qwen3-VL-4B on the Ampere GPU with the chosen prompt; the answer streams onto the screen live.")

# =====================================================================
# 2. Cascade: how it works (same layout as the IQ9's cascade-how-it-works)
# =====================================================================
d = SVG("cascade-how-it-works", "How the detection to VLM cascade works on the AGX Orin",
        "A camera feed or video feeds YOLOv8n on the Orin's DLA (TensorRT; its detection head falls back to the GPU), "
        "which detects vehicles; a tracker crops the best view of each one; Qwen3-VL-4B on the Ampere GPU turns a crop "
        "into a record of colour, type, brand and number plate, which a deployment would store in a database and show "
        "to operators on a dashboard.")
d.zone(40, 96, 208, 528, "INPUT")
d.zone(288, 96, 600, 528, "ON-DEVICE · " + PLATFORM)
d.zone(920, 96, 320, 528, "RESULTS")
# arrows
d.arrow("M 240,172 H 304")                                            # camera -> yolo
d.arrow("M 240,316 H 260 Q 268,316 268,308 V 212 Q 268,204 276,204 H 304")  # video -> yolo
d.arrow("M 576,184 H 616")                                            # yolo -> tracker
d.arrow("M 744,240 V 304")                                            # tracker -> vlm (crops)
d.arrow("M 616,424 H 448 Q 440,424 440,432 V 496")                    # vlm -> record
d.arrow("M 576,552 H 936", "link")                                    # record -> database
d.arrow("M 1080,504 V 408", "link")                                   # database -> dashboard
d.arrow("M 1080,312 V 224", "link")                                   # dashboard -> operators
d.label(752, 264, "VEHICLE CROPS", SOFT, anchor="left")
d.label(528, 400, "RECORD", SOFT)
d.label(744, 528, "STORE RECORD", LINK)
d.label(1088, 452, "QUERY", LINK, anchor="left")
d.label(1088, 260, "LIVE + ALERTS", LINK, anchor="left")
# nodes
d.inode(48, 128, 192, 96, "input", "camera", "Camera feed", ["USB · GMSL", "RTSP camera"], step=1)
d.inode(48, 272, 192, 88, "input", "movie", "Video file", ["recorded", "MP4 · looped"])
d.inode(304, 128, 272, 112, "focal", "cpu", "YOLOv8n detection", ["TensorRT FP16 · 640×640", "DLA 0 · head on GPU", "~25 ms per frame"], step=2)
d.inode(616, 128, 256, 112, "backend", "crop", "Track + crop", ["IoU tracker", "entry zone filter", "best crop / vehicle"], step=3)
d.inode(616, 304, 256, 152, "focal", "cpu", "Qwen3-VL-4B", ["Instruct · Q4_K_M", "Ollama · Ampere GPU", "one crop at a time"], step=4)
# prompt chip inside the VLM node
d.raw(f'<rect x="628" y="412" width="232" height="32" rx="4" fill="#ffffff" stroke="rgba(235,108,54,0.35)" stroke-width="0.8"/>'
      + icon("terminal", 636, 420, 16, ACCENT)
      + f'<text x="660" y="432" fill="{INK}" font-size="11" font-family="{MONO}">color · type · brand · plate</text>')
# record card: the facts from a real Orin run on the highway video
# ("a white Ford Transit van ... the license plate ... reads C98191P")
lines = ['{', '  "vehicle_type": "van",', '  "color": "white",', '  "brand": "Ford Transit",', '  "number_plate": "C98191P"', '}']
d.raw(f'<rect x="304" y="496" width="272" height="112" rx="6" fill="{PAPER}"/>'
      f'<rect x="304" y="496" width="272" height="112" rx="6" fill="#ffffff" stroke="rgba(45,49,66,0.30)" stroke-width="1"/>')
for i, ln in enumerate(lines):
    assert w_mono(ln, SUB) <= 272 - 24, ln
    if ":" in ln:
        k, v = ln.split(":", 1)
        body = f'<tspan fill="{INK}">{escape(k)}:</tspan><tspan fill="{LINK}">{escape(v)}</tspan>'
    else:
        body = f'<tspan fill="{SOFT}">{escape(ln)}</tspan>'
    d.raw(f'<text x="320" y="{520 + i * 16}" font-size="{SUB}" font-family="{MONO}" xml:space="preserve">{body}</text>')
d.inode(936, 504, 288, 96, "store", "database", "Database", ["plate · time · camera", "+ vehicle crop"], step=5)
d.inode(936, 312, 288, 96, "backend", "desktop", "Dashboard", ["live view · alerts", "search by plate"])
d.inode(936, 128, 288, 96, "input", "users", "Operators", ["security desk", "facility team"])
d.raw(f'<text x="304" y="296" fill="{MUTED}" font-size="14" font-style="italic" font-family="\'Instrument Serif\', serif">'
      f'<tspan x="304" dy="0">Detector on the DLA, VLM on the GPU —</tspan>'
      f'<tspan x="304" dy="20">two engines on one board; video and</tspan>'
      f'<tspan x="304" dy="20">plates never leave the device.</tspan></text>')
d.legend([FOCAL_LEGEND, ("backend", "Processing / app"), ("input", "Source / people"),
          ("store", "Data"), ("muted", "On-device flow"), ("link", "Results out")])
write("cascade-how-it-works.html", d, "How it works · Advantech AFE-R750 · Jetson AGX Orin 32GB",
      "From camera to searchable vehicle records — on one device",
      "YOLOv8n spots and tracks each vehicle on the DLA; Qwen3-VL-4B on the GPU reads its colour, type, brand and plate "
      "from the best crop; a deployment would send the record to a database and the operators' dashboard.")

# =====================================================================
# 3. Full system: both demo pages, every process, and the engine it runs on
# =====================================================================
d = SVG("demo-system", "Live VLM demo: full system on the AFE-R750",
        "Both demo pages and every process behind them: Firefox shows the pages; the WebUI server (Python, port 8090) "
        "reads the camera, samples frames, tracks detections, streams video out over WebRTC, calls the VLM and monitors "
        "the system on the CPU; VLM requests go to Ollama (11434), which runs llama.cpp with CUDA; Qwen3-VL-4B runs on "
        "the Ampere GPU and YOLOv8n on DLA 0 through TensorRT.")
d.zone(40, 96, 1200, 120, "FIREFOX · BROWSER")
d.zone(40, 256, 808, 232, "WEBUI SERVER · :8090 · CPU")
d.zone(880, 256, 360, 232, "OLLAMA · :11434 · CPU HOST SIDE")
d.zone(256, 528, 192, 120, "DLA 0")
d.zone(680, 528, 560, 120, "AMPERE GPU · CUDA")

# arrows (before nodes)
d.arrow("M 112,280 V 200", "link")                                                  # WebRTC -> /traffic (video out)
d.arrow("M 188,280 V 248 Q 188,240 196,240 H 288 Q 296,240 296,232 V 200", "link")  # WebRTC -> main page (video out)
d.arrow("M 264,304 H 232")                                                          # frame sampler -> WebRTC
d.arrow("M 232,432 H 244 Q 252,432 252,424 V 384 a 8,8 0 0,0 0,-16 V 352 Q 252,344 260,344 H 264")  # camera -> sampler
d.arrow("M 440,320 H 472")                                                          # frame sampler -> VLM service
d.arrow("M 560,280 V 200", "dashed-link")                                           # VLM service -> main page (text)
d.arrow("M 648,320 H 896", "link")                                                  # VLM service -> ollama serve
# monitor -> page: hop over the VLM->ollama line at y=320
d.arrow("M 704,392 V 328 a 8,8 0 0,1 0,-16 V 256 Q 704,248 696,248 H 640 Q 632,248 632,240 V 200", "link")
d.arrow("M 472,432 H 440")                                                          # demo videos -> detection track
d.arrow("M 280,392 V 384 Q 280,376 272,376 H 228 Q 220,376 220,368 V 360")         # detection track -> WebRTC
d.arrow("M 424,392 V 384 Q 424,376 432,376 H 504 Q 512,376 512,368 V 360")          # detection track -> VLM (crops)
d.arrow("M 320,472 V 552", "link")                                                  # detection -> YOLO
d.arrow("M 392,552 V 472", "dashed-link")                                           # YOLO -> detection (boxes)
d.arrow("M 1060,360 V 392", "link")                                                 # ollama serve -> llama-server
d.arrow("M 1060,472 V 552", "link")                                                 # llama-server -> Qwen

d.label(120, 224, "VIDEO", LINK, anchor="left")
d.label(242, 232, "VIDEO", LINK)
d.label(568, 224, "TEXT", LINK, anchor="left")
d.label(668, 240, "STATS", LINK)
d.label(800, 296, "HTTP /V1/CHAT", LINK)
d.label(468, 368, "CROPS", SOFT)
d.label(328, 504, "FRAME", LINK, anchor="left")
d.label(400, 504, "BOXES", LINK, anchor="left")
d.label(1068, 504, "CUDA", LINK, anchor="left")

# nodes
d.node(56, 120, 176, 80, "input", "UI", "/traffic page", ["video + boxes", "vehicles · Describe"])
d.node(264, 120, 384, 80, "input", "UI", "Live VLM page /", ["camera video · VLM text · stats"])
d.node(680, 120, 248, 80, "input", "UI", "/compare page", ["Orin GPU vs IQ9 NPU"])
d.node(56, 280, 176, 80, "backend", "CPU", "WebRTC · aiortc", ["video out", "to both pages"])
d.node(56, 392, 176, 80, "input", "CAM", "Camera feed", ["USB · GMSL · RTSP", "read by server"])
d.node(264, 280, 176, 80, "backend", "CPU", "Frame sampler", ["every 30th frame", "JPEG → base64"])
d.node(472, 280, 176, 80, "backend", "CPU", "VLM service", ["streamed answer", "TTFT · tok/s"])
d.node(264, 392, 176, 80, "backend", "CPU", "Detection track", ["IoU track · zone", "draw boxes"])
d.node(472, 392, 176, 80, "store", "FILE", "Demo videos", ["PyAV decode", "highway video"])
d.node(680, 392, 152, 80, "backend", "CPU", "System monitor", ["every 0.25 s", "jtop · thermal"])
d.node(896, 280, 328, 80, "backend", "CPU", "ollama serve", [":11434 · OpenAI API", "loads models on demand"])
d.node(896, 392, 328, 80, "backend", "CPU", "llama-server", ["llama.cpp · started by Ollama"])
d.node(264, 552, 176, 80, "focal", "DLA", "YOLOv8n", ["TensorRT FP16", "~25 ms · head on GPU"])
d.node(896, 552, 328, 80, "focal", "GPU", "Qwen3-VL-4B", ["Q4_K_M · llama.cpp CUDA", "~3.7 s per answer"])
d.raw(f'<text x="696" y="584" fill="{MUTED}" font-size="14" font-style="italic" font-family="\'Instrument Serif\', serif">'
      f'<tspan x="696" dy="0">VLM on the GPU,</tspan>'
      f'<tspan x="696" dy="20">detector on the DLA:</tspan>'
      f'<tspan x="696" dy="20">two engines, one board.</tspan></text>')
d.legend([FOCAL_LEGEND, ("backend", "Process (chip in tag)"), ("input", "Page / camera"),
          ("store", "File"), ("muted", "In-process"), ("link", "WebRTC · WS · HTTP"), ("dashed", "Result back")], y=676)
write("demo-system.html", d, "Full system · Advantech AFE-R750 · Jetson AGX Orin 32GB",
      "Both demos, every process, and the engine it runs on",
      "The VLM on the Ampere GPU (Ollama, llama.cpp with CUDA), the vehicle detector on DLA 0 (TensorRT), everything "
      "else on the 8× Cortex-A78AE CPU. Stats go to both pages; only the main page's arrow is drawn.")
print("written")
