"""
Print-quality PDF rendering for a completed COMPASS run.

The renderer takes the JSON artifacts a run leaves behind (performance report,
patient report, deep phenotype markdown, data overview, deviation map) and lays
them out as a single standardized A4 document. Everything is drawn with
ReportLab primitives, so there is no plotting dependency, no network access and
no system binary in the path.

Two rules shape the whole module:

* It never raises on a missing or malformed optional section. Absent evidence is
  a finding in itself, so every section states explicitly that information was
  not recorded rather than quietly disappearing.
* Layout is a design system, not ad hoc geometry. Colours, the type scale and
  the vertical rhythm live in module constants at the top and nothing below them
  invents its own values.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ============================================================================
# Design system
# ============================================================================

#: Hex strings as well as colours, because the inline markup the paragraph
#: parser understands takes the string form and the two must not drift.
INK_HEX = "#101820"
MUTED_HEX = "#6B7684"
#: One accent for emphasis, one diverging pair for signed data, and nothing
#: else. The high end of the diverging pair doubles as the alert colour, so a
#: warning never needs a hue the reader has not already been taught to read.
ACCENT_HEX = "#1D3557"
DEV_HIGH_HEX = "#A8382B"
DEV_LOW_HEX = "#3C6E8F"

INK = colors.HexColor(INK_HEX)
MUTED = colors.HexColor(MUTED_HEX)
HAIRLINE = colors.HexColor("#E3E7EB")
#: A rule that has to be seen, under a table header or above a totals row.
RULE = colors.HexColor("#AEB7C0")
PAGE_TINT = colors.HexColor("#F4F6F8")
PAPER = colors.HexColor("#FFFFFF")
ACCENT = colors.HexColor(ACCENT_HEX)
DEV_HIGH = colors.HexColor(DEV_HIGH_HEX)
DEV_LOW = colors.HexColor(DEV_LOW_HEX)
ALERT = DEV_HIGH

#: Type scale: (font size, leading). Helvetica and Courier ship with every
#: viewer, so the document renders identically without embedded fonts.
FONT_SANS = "Helvetica"
FONT_SANS_BOLD = "Helvetica-Bold"
FONT_SANS_OBLIQUE = "Helvetica-Oblique"
FONT_MONO = "Courier"
FONT_MONO_BOLD = "Courier-Bold"

#: Three levels of heading and three of text, and the document uses no others.
#: A fourth heading level was what made the old layout read as a debug dump.
SIZE_DISPLAY, LEAD_DISPLAY = 25.0, 30.0
SIZE_H1, LEAD_H1 = 17.0, 22.0
SIZE_H2, LEAD_H2 = 11.0, 15.0
SIZE_H3, LEAD_H3 = 9.5, 13.0
SIZE_BODY, LEAD_BODY = 9.5, 14.5
SIZE_SMALL, LEAD_SMALL = 8.5, 12.5
SIZE_MICRO, LEAD_MICRO = 7.5, 10.0

PAGE_SIZE = A4
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_SIDE = 46.0
MARGIN_TOP = 58.0
MARGIN_BOTTOM = 50.0
CONTENT_WIDTH = PAGE_WIDTH - (2 * MARGIN_SIDE)
#: Inset of a fenced code panel, and the text width that leaves inside it.
CODE_PADDING = 8.0
CODE_WIDTH = CONTENT_WIDTH - (2 * CODE_PADDING)

#: Vertical rhythm. Every gap in the document is a multiple of these.
RHYTHM = 6.0
GAP_TIGHT = RHYTHM
GAP_BLOCK = RHYTHM * 2
GAP_SECTION = RHYTHM * 3
GAP_MAJOR = RHYTHM * 4

COVER_BAND_HEIGHT = 132.0
COVER_FRAME_TOP_GAP = 26.0
#: Usable height of the cover frame. The cover is a single frame of known size,
#: so the builder can measure what it has and decide what else will fit.
COVER_FRAME_HEIGHT = PAGE_HEIGHT - COVER_BAND_HEIGHT - COVER_FRAME_TOP_GAP - MARGIN_BOTTOM
#: A leftover taller than this is a void worth filling rather than leaving open.
COVER_GLANCE_MIN_SPACE = 118.0
#: Space that must survive the coverage band before a deviation strip is added.
COVER_DEVIATION_MIN_SPACE = 90.0
#: Slack kept under the cover frame so a measuring error cannot spill a page.
COVER_FIT_MARGIN = 6.0
#: Leaves the cover's deviation strip ranks.
COVER_DEVIATION_ROWS = 5
#: Geometry of the slim bar rows the cover uses, well below the in-body chart.
COVER_STRIP_ROW_HEIGHT = 11.5
COVER_STRIP_GAP = 3.0
COVER_STRIP_BAR_HEIGHT = 5.5

#: Separates the facts on the cover's one-line metadata strips. A middle dot
#: is in the built-in fonts, unlike the glyphs a designer would reach for.
META_SEPARATOR = "  \u00b7  "
DISCLAIMER = (
    "COMPASS is a research prototype and is not a certified medical device. "
    "Outputs require review by qualified domain experts."
)
#: The title page carries the claim only; every body page carries the full
#: sentence, so repeating the review clause under the wave would clip it.
COVER_CAVEAT = "Research prototype. Not a certified medical device."
NOT_AVAILABLE = "Not available"
#: Section titles and standfirsts, in document order. The cover contents
#: list and the section headers both read from here so they cannot drift.
SECTION_SPECS: Tuple[Tuple[str, str], ...] = (
    ("Prediction", "Task and primary output"),
    ("Evidence", "What the prediction rests on"),
    ("Uncertainty", "What the run could not see"),
    ("Coverage", "Source data present per domain"),
    ("Deviation profile", "Distance from the normative mean"),
    ("Critic evaluation", "Automated quality review"),
    ("Deep phenotype narrative", "Written synthesis of the run"),
    ("Execution and cost", "How the run was carried out"),
    ("Provenance", "How to reproduce this report"),
)

#: Caps on how much of a long list is printed before it is elided with a count.
#: A clinical report states its evidence, it does not transcribe the run log.
MISSING_FEATURE_CAP = 24
KEY_FINDING_CAP = 12
REASONING_STEP_CAP = 12
CHECKLIST_CAP = 16
#: Cap on how many leaf paths the deviation chart shows.
DEVIATION_CHART_ROWS = 12
#: Depth guard for the generic deviation-map walk.
MAX_TREE_DEPTH = 12

try:  # The engine version is nice to have, not worth an import failure.
    from ..config.settings import COMPASS_VERSION as _ENGINE_VERSION
except Exception:  # pragma: no cover - defensive, settings pulls heavy optionals
    _ENGINE_VERSION = ""

#: The repository root, which is where the reproduce command tells the reader to
#: stand. A participant directory underneath it is written relative to it, which
#: keeps the one unbreakable token in that command short enough to print whole.
_REPO_ROOT = Path(__file__).resolve().parents[4]


# ============================================================================
# Colour and formatting helpers
# ============================================================================


def tint(color: colors.Color, amount: float) -> colors.Color:
    """Blend ``color`` toward white. ``amount`` 0 keeps it, 1 makes it white."""
    ratio = min(max(float(amount), 0.0), 1.0)
    return colors.Color(
        color.red + (1.0 - color.red) * ratio,
        color.green + (1.0 - color.green) * ratio,
        color.blue + (1.0 - color.blue) * ratio,
    )


def _s(value: Any, default: str = "") -> str:
    """Coerce anything to a trimmed string without ever raising."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip() or default
    try:
        return str(value).strip() or default
    except Exception:
        return default


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return None if math.isnan(float(value)) or math.isinf(float(value)) else float(value)
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) or math.isinf(parsed) else parsed


def _as_int(value: Any) -> Optional[int]:
    parsed = _as_float(value)
    return None if parsed is None else int(round(parsed))


def fmt_int(value: Any, default: str = NOT_AVAILABLE) -> str:
    """Thousands-separated integer."""
    parsed = _as_int(value)
    return default if parsed is None else f"{parsed:,}"


def fmt_usd(value: Any, default: str = "Not priced") -> str:
    """USD with 4 decimals below one cent, 2 decimals above."""
    parsed = _as_float(value)
    if parsed is None:
        return default
    return f"${parsed:.4f}" if abs(parsed) < 0.01 else f"${parsed:,.2f}"


def fmt_z(value: Any, default: str = "n/a") -> str:
    """Signed z-score to two decimals."""
    parsed = _as_float(value)
    return default if parsed is None else f"{parsed:+.2f}"


def fmt_pct(value: Any, default: str = NOT_AVAILABLE) -> str:
    """Percentage to one decimal. Input is already 0-100."""
    parsed = _as_float(value)
    return default if parsed is None else f"{parsed:.1f}%"


def fmt_prob(value: Any, default: str = NOT_AVAILABLE) -> str:
    """Probability in [0,1] shown as a one-decimal percentage."""
    parsed = _as_float(value)
    return default if parsed is None else f"{parsed * 100:.1f}%"


def fmt_number(value: Any, default: str = NOT_AVAILABLE) -> str:
    """A model output value: integers stay integral, floats get three decimals."""
    parsed = _as_float(value)
    if parsed is None:
        return default
    if abs(parsed - round(parsed)) < 1e-9 and abs(parsed) >= 1000:
        return f"{int(round(parsed)):,}"
    return f"{parsed:,.3f}"


def fmt_duration(seconds: Any, default: str = NOT_AVAILABLE) -> str:
    """Run duration in a compact human form."""
    parsed = _as_float(seconds)
    if parsed is None or parsed < 0:
        return default
    if parsed < 90:
        return f"{parsed:.1f} s"
    minutes, rest = divmod(parsed, 60.0)
    if minutes < 60:
        return f"{int(minutes)} m {int(rest):02d} s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours} h {minutes:02d} m"


def fmt_timestamp(value: Any) -> str:
    """ISO timestamp rendered as a readable local-style stamp."""
    text = _s(value)
    if not text:
        return NOT_AVAILABLE
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate).strftime("%d %b %Y, %H:%M")
        except ValueError:
            continue
    return text


def humanize(token: Any) -> str:
    """``binary_classification`` becomes ``Binary classification``."""
    text = _s(token)
    if not text:
        return NOT_AVAILABLE
    cleaned = text.replace("_", " ").replace("-", " ").strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else NOT_AVAILABLE


def escape(text: Any) -> str:
    """Escape for the ReportLab mini-HTML parser."""
    raw = "" if text is None else str(text)
    return raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fit_text(
    text: str,
    font: str,
    size: float,
    max_width: float,
    min_size: Optional[float] = None,
) -> Tuple[str, float]:
    """Shrink then, if needed, ellipsize ``text`` so it fits ``max_width``."""
    if max_width <= 0:
        return "", size
    floor = min_size if min_size is not None else size
    current = size
    while current > floor and stringWidth(text, font, current) > max_width:
        current -= 0.5
    if stringWidth(text, font, current) <= max_width:
        return text, current
    ellipsis = "..."
    trimmed = text
    while trimmed and stringWidth(trimmed + ellipsis, font, current) > max_width:
        trimmed = trimmed[:-1]
    return (trimmed + ellipsis) if trimmed else "", current


def _draw_tracked(
    canvas: Any,
    x: float,
    y: float,
    text: str,
    font: str,
    size: float,
    color: colors.Color,
    tracking: float = 1.0,
) -> None:
    """Draw letterspaced text. Only text objects expose character spacing."""
    if not text:
        return
    canvas.saveState()
    obj = canvas.beginText(x, y)
    obj.setFont(font, size)
    obj.setFillColor(color)
    obj.setCharSpace(tracking)
    obj.textOut(text)
    obj.setCharSpace(0)  # character spacing is graphics state, so reset it here
    canvas.drawText(obj)
    canvas.restoreState()


# ============================================================================
# Paragraph styles
# ============================================================================


def build_styles() -> Dict[str, ParagraphStyle]:
    """The document's paragraph styles, keyed by name."""
    styles: Dict[str, ParagraphStyle] = {}

    def add(name: str, **kwargs: Any) -> None:
        params: Dict[str, Any] = {
            "name": name,
            "fontName": FONT_SANS,
            "fontSize": SIZE_BODY,
            "leading": LEAD_BODY,
            "textColor": INK,
            "spaceBefore": 0,
            "spaceAfter": 0,
        }
        params.update(kwargs)
        styles[name] = ParagraphStyle(**params)

    add("Display", fontName=FONT_SANS_BOLD, fontSize=SIZE_DISPLAY, leading=LEAD_DISPLAY)
    add("H1", fontName=FONT_SANS_BOLD, fontSize=SIZE_H1, leading=LEAD_H1, keepWithNext=1)
    # The single sub-heading level inside a section. It earns its weight from
    # the space above it rather than from size, colour or a rule.
    add(
        "H2",
        fontName=FONT_SANS_BOLD,
        fontSize=SIZE_H2,
        leading=LEAD_H2,
        spaceBefore=GAP_SECTION,
        spaceAfter=GAP_TIGHT,
        keepWithNext=1,
    )
    add(
        "H3",
        fontName=FONT_SANS_BOLD,
        fontSize=SIZE_H3,
        leading=LEAD_H3,
        spaceBefore=GAP_BLOCK,
        spaceAfter=3,
        keepWithNext=1,
    )
    add("Body", spaceAfter=GAP_TIGHT)
    add("BodyTight", spaceAfter=2)
    add("BodyMuted", textColor=MUTED, spaceAfter=GAP_TIGHT)
    add("Lead", fontSize=SIZE_H2, leading=LEAD_H2 + 2, spaceAfter=GAP_TIGHT)
    add("Small", fontSize=SIZE_SMALL, leading=LEAD_SMALL)
    add("SmallMuted", fontSize=SIZE_SMALL, leading=LEAD_SMALL, textColor=MUTED)
    add("Micro", fontSize=SIZE_MICRO, leading=LEAD_MICRO, textColor=MUTED)
    add("Mono", fontName=FONT_MONO, fontSize=SIZE_SMALL, leading=LEAD_SMALL, wordWrap="CJK")
    add(
        "MonoMicro",
        fontName=FONT_MONO,
        fontSize=SIZE_MICRO,
        leading=LEAD_MICRO,
        textColor=MUTED,
        wordWrap="CJK",
    )
    add("BigValue", fontName=FONT_SANS_BOLD, fontSize=21, leading=25)
    add(
        "TableHeader",
        fontName=FONT_SANS_BOLD,
        fontSize=SIZE_MICRO,
        leading=LEAD_MICRO,
        textColor=MUTED,
    )
    add(
        "TableHeaderRight",
        fontName=FONT_SANS_BOLD,
        fontSize=SIZE_MICRO,
        leading=LEAD_MICRO,
        textColor=MUTED,
        alignment=TA_RIGHT,
    )
    add("TableCell", fontSize=SIZE_SMALL, leading=LEAD_SMALL)
    add("TableCellMuted", fontSize=SIZE_SMALL, leading=LEAD_SMALL, textColor=MUTED)
    # Ontology domain names are long compound identifiers, so the column that
    # carries them is set smaller to keep them on one line.
    add("TableCellLabel", fontSize=SIZE_MICRO, leading=LEAD_SMALL, textColor=MUTED)
    add("TableCellMono", fontName=FONT_MONO, fontSize=SIZE_MICRO + 0.5, leading=LEAD_SMALL, wordWrap="CJK")
    add("TableCellRight", fontSize=SIZE_SMALL, leading=LEAD_SMALL, alignment=TA_RIGHT)
    # Courier is the only tabular-figure face guaranteed to be present, so
    # every column of numerals uses it and every one of them is right aligned.
    add(
        "TableCellNumeric",
        fontName=FONT_MONO,
        fontSize=SIZE_MICRO + 0.5,
        leading=LEAD_SMALL,
        alignment=TA_RIGHT,
    )
    add("TableCellBold", fontName=FONT_SANS_BOLD, fontSize=SIZE_SMALL, leading=LEAD_SMALL)
    add(
        "TableCellNumericBold",
        fontName=FONT_MONO_BOLD,
        fontSize=SIZE_MICRO + 0.5,
        leading=LEAD_SMALL,
        alignment=TA_RIGHT,
    )
    add("SpecKey", fontSize=SIZE_SMALL, leading=LEAD_SMALL, textColor=MUTED)
    add("SpecValue", fontSize=SIZE_SMALL, leading=LEAD_SMALL, textColor=INK)
    add("SpecValueMono", fontName=FONT_MONO, fontSize=SIZE_MICRO + 0.5, leading=LEAD_SMALL, wordWrap="CJK")
    add("Bullet", spaceAfter=3, leftIndent=13, bulletIndent=2)
    add("BulletNested", spaceAfter=3, leftIndent=26, bulletIndent=15, textColor=colors.HexColor("#3A4550"))
    add("Numbered", spaceAfter=3, leftIndent=18, bulletIndent=2)
    add("CalloutBody", fontSize=SIZE_SMALL, leading=LEAD_SMALL + 0.5)
    add("Code", fontName=FONT_MONO, fontSize=SIZE_SMALL, leading=LEAD_SMALL + 1, wordWrap="CJK")
    # The narrative arrives as markdown written by the Communicator, so its
    # headings map onto the same three levels the rest of the document uses.
    add("MdH1", fontName=FONT_SANS_BOLD, fontSize=SIZE_H1 - 3.0, leading=LEAD_H1 - 3.0, spaceBefore=GAP_SECTION, spaceAfter=GAP_TIGHT, keepWithNext=1)
    add("MdH2", fontName=FONT_SANS_BOLD, fontSize=SIZE_H2, leading=LEAD_H2, spaceBefore=GAP_SECTION, spaceAfter=3, keepWithNext=1)
    add("MdH3", fontName=FONT_SANS_BOLD, fontSize=SIZE_H3, leading=LEAD_H3, spaceBefore=GAP_BLOCK, spaceAfter=2, keepWithNext=1)
    add(
        "MdH4",
        fontName=FONT_SANS_BOLD,
        fontSize=SIZE_SMALL,
        leading=LEAD_SMALL,
        textColor=MUTED,
        spaceBefore=GAP_BLOCK,
        spaceAfter=2,
        keepWithNext=1,
    )
    return styles


_STYLE_CACHE: Dict[str, ParagraphStyle] = {}


def _styles() -> Dict[str, ParagraphStyle]:
    """Module-level style singleton so flowables can build their own text."""
    global _STYLE_CACHE
    if not _STYLE_CACHE:
        _STYLE_CACHE = build_styles()
    return _STYLE_CACHE


def _style(styles: Any, name: str) -> ParagraphStyle:
    """Look a style up in a dict or a ReportLab stylesheet, with a fallback."""
    try:
        found = styles[name]
        if isinstance(found, ParagraphStyle):
            return found
    except Exception:
        pass
    fallback = _styles()
    return fallback.get(name) or fallback["Body"]


def _indented(style: ParagraphStyle, indent: float) -> ParagraphStyle:
    """A clone of ``style`` pushed right by ``indent`` points."""
    clone = ParagraphStyle(name=f"{style.name}_i{int(indent)}", parent=style)
    clone.leftIndent = style.leftIndent + indent
    return clone


def P(text: Any, style_name: str = "Body", styles: Optional[Any] = None) -> Paragraph:
    """Escaped paragraph shorthand. Text is treated as plain text, not markup."""
    return Paragraph(escape(text), _style(styles or _styles(), style_name))


def RP(markup: str, style_name: str = "Body", styles: Optional[Any] = None) -> Paragraph:
    """Paragraph for text that already contains intentional ReportLab markup."""
    return Paragraph(markup, _style(styles or _styles(), style_name))


# ============================================================================
# Custom flowables
# ============================================================================


class HRule(Flowable):
    """A hairline rule with breathing room above and below."""

    def __init__(
        self,
        color: colors.Color = HAIRLINE,
        thickness: float = 0.6,
        space_before: float = GAP_TIGHT,
        space_after: float = GAP_TIGHT,
        width_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.color = color
        self.thickness = thickness
        self.space_before = space_before
        self.space_after = space_after
        self.width_ratio = width_ratio
        self.width = 0.0
        self.height = space_before + space_after + thickness

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        canvas.setStrokeColor(self.color)
        canvas.setLineWidth(self.thickness)
        y = self.space_after + (self.thickness / 2.0)
        canvas.line(0, y, self.width * self.width_ratio, y)


class SectionHeader(Flowable):
    """
    A section opening: a letterspaced number label, the title, a standfirst.

    The rule sits above the block rather than below it, so the title reads as
    the start of what follows instead of the end of what came before.
    """

    #: Space between the rule above and the number label.
    GAP_UNDER_RULE = 9.0
    LABEL_GAP = 5.0

    def __init__(self, number: Any, title: str, subtitle: str = "") -> None:
        super().__init__()
        self.number = _s(number, "")
        self.title = _s(title, "Section")
        self.subtitle = _s(subtitle, "")
        self.width = 0.0
        self.height = 0.0
        self._subtitle_para: Optional[Paragraph] = None
        self._subtitle_height = 0.0

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        self._subtitle_height = 0.0
        self._subtitle_para = None
        if self.subtitle:
            self._subtitle_para = P(self.subtitle, "SmallMuted")
            _, self._subtitle_height = self._subtitle_para.wrap(availWidth, availHeight)
        self.height = (
            self.GAP_UNDER_RULE
            + LEAD_MICRO
            + self.LABEL_GAP
            + LEAD_H1
            + (self._subtitle_height + 2.0 if self._subtitle_para else 0.0)
        )
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(1.4)
        canvas.line(0, self.height, self.width, self.height)

        label_baseline = self.height - self.GAP_UNDER_RULE - SIZE_MICRO
        if self.number:
            _draw_tracked(
                canvas,
                0,
                label_baseline,
                f"SECTION {self.number}",
                FONT_SANS_BOLD,
                SIZE_MICRO,
                ACCENT,
                1.6,
            )

        subtitle_height = self._subtitle_height + 2.0 if self._subtitle_para else 0.0
        title_baseline = subtitle_height + (LEAD_H1 - SIZE_H1) * 0.5
        canvas.setFillColor(INK)
        title, size = _fit_text(self.title, FONT_SANS_BOLD, SIZE_H1, self.width, min_size=12.0)
        canvas.setFont(FONT_SANS_BOLD, size)
        canvas.drawString(0, title_baseline, title)

        if self._subtitle_para is not None:
            self._subtitle_para.drawOn(canvas, 0, 0)


class Callout(Flowable):
    """
    A note set off by a rule on its left edge, on a neutral ground.

    Only the rule and the title carry the tone. The panel itself stays neutral
    so a page with two callouts does not turn into a colour chart.
    """

    PAD_X = 11.0
    PAD_Y = 8.0
    SPINE = 2.2
    MAX_CHARS = 1600

    def __init__(
        self,
        text: str,
        tone: colors.Color = ACCENT,
        title: str = "",
        space_after: float = GAP_BLOCK,
    ) -> None:
        super().__init__()
        self.text = _s(text, "")[: self.MAX_CHARS]
        self.tone = tone
        self.title = _s(title, "")
        self.space_after = space_after
        self.width = 0.0
        self.height = 0.0
        self._paras: List[Paragraph] = []
        self._heights: List[float] = []

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        inner = max(availWidth - (2 * self.PAD_X) - self.SPINE, 40.0)
        self._paras = []
        self._heights = []
        if self.title:
            title_style = ParagraphStyle(
                name="CalloutTitle",
                parent=_style(_styles(), "Small"),
                fontName=FONT_SANS_BOLD,
                textColor=self.tone,
                spaceAfter=2,
            )
            para = Paragraph(escape(self.title.upper()), title_style)
            self._paras.append(para)
            self._heights.append(para.wrap(inner, availHeight)[1] + 2.0)
        body = Paragraph(escape(self.text), _style(_styles(), "CalloutBody"))
        self._paras.append(body)
        self._heights.append(body.wrap(inner, availHeight)[1])
        self.height = sum(self._heights) + (2 * self.PAD_Y) + self.space_after
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        box_height = self.height - self.space_after
        box_y = self.space_after
        canvas.setFillColor(PAGE_TINT)
        canvas.rect(0, box_y, self.width, box_height, stroke=0, fill=1)
        canvas.setFillColor(self.tone)
        canvas.rect(0, box_y, self.SPINE, box_height, stroke=0, fill=1)

        cursor = box_y + box_height - self.PAD_Y
        for para, height in zip(self._paras, self._heights):
            cursor -= height
            para.drawOn(canvas, self.PAD_X + self.SPINE, cursor)


class Chip(Flowable):
    """A small rounded label, sized to its text. Usable inside table cells."""

    PAD_X = 5.0
    PAD_Y = 2.5
    FONT_SIZE = SIZE_MICRO

    def __init__(self, text: str, color: colors.Color = ACCENT, solid: bool = False) -> None:
        super().__init__()
        self.text = _s(text, "-")
        self.color = color
        self.solid = solid
        self.width = 0.0
        self.height = self.FONT_SIZE + (2 * self.PAD_Y) + 2.0
        self._label = self.text

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        label, _ = _fit_text(self.text, FONT_SANS_BOLD, self.FONT_SIZE, max(availWidth - (2 * self.PAD_X), 8.0))
        self._label = label
        text_width = stringWidth(label, FONT_SANS_BOLD, self.FONT_SIZE)
        self.width = min(text_width + (2 * self.PAD_X), max(availWidth, 12.0))
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        if self.solid:
            canvas.setFillColor(self.color)
            canvas.rect(0, 1.0, self.width, self.height - 2.0, stroke=0, fill=1)
            text_color = PAPER
        else:
            canvas.setStrokeColor(self.color)
            canvas.setLineWidth(0.6)
            canvas.rect(0, 1.0, self.width, self.height - 2.0, stroke=1, fill=0)
            text_color = self.color
        canvas.setFillColor(text_color)
        canvas.setFont(FONT_SANS_BOLD, self.FONT_SIZE)
        canvas.drawString(self.PAD_X, 1.0 + self.PAD_Y + 1.0, self._label)


@dataclass
class KpiItem:
    """One tile in the cover grid."""

    label: str
    value: str
    note: str = ""
    accent: Optional[colors.Color] = None


class KpiGrid(Flowable):
    """Rounded KPI tiles laid out on a fixed column grid."""

    splittable = True


    def __init__(
        self,
        items: Sequence[KpiItem],
        cols: int = 3,
        gap: float = 9.0,
        tile_height: float = 62.0,
        space_after: float = 0.0,
    ) -> None:
        super().__init__()
        self.items = list(items)
        self.cols = max(int(cols), 1)
        self.gap = gap
        self.tile_height = tile_height
        self.space_after = space_after
        self.width = 0.0
        self.height = 0.0

    def _rows(self) -> int:
        return max(1, int(math.ceil(len(self.items) / float(self.cols)))) if self.items else 0

    def first_slice_height(self) -> float:
        """Smallest slice :meth:`split` can leave behind: one row of tiles."""
        return self.tile_height if self.items else 0.0

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        rows = self._rows()
        self.height = (
            0.0
            if rows == 0
            else rows * self.tile_height + (rows - 1) * self.gap + self.space_after
        )
        return self.width, self.height

    def split(self, availWidth: float, availHeight: float) -> List[Flowable]:
        """Split on tile-row boundaries so a tall grid never overflows a frame."""
        rows = self._rows()
        if rows <= 1:
            return []
        per_row = self.tile_height + self.gap
        fits = int((availHeight + self.gap) // per_row)
        if fits <= 0 or fits >= rows:
            return []
        cut = fits * self.cols
        head = KpiGrid(self.items[:cut], self.cols, self.gap, self.tile_height, self.gap)
        tail = KpiGrid(self.items[cut:], self.cols, self.gap, self.tile_height, self.space_after)
        return [head, tail]

    def draw(self) -> None:
        if not self.items:
            return
        canvas = self.canv
        tile_w = (self.width - self.gap * (self.cols - 1)) / float(self.cols)
        # Trailing space belongs under the block: a flowable draws from the top
        # of the box it was given, so the leftover has to fall out at the bottom.
        top = self.height
        for index, item in enumerate(self.items):
            row, col = divmod(index, self.cols)
            x = col * (tile_w + self.gap)
            y = top - (row + 1) * self.tile_height - row * self.gap
            self._draw_tile(canvas, item, x, y, tile_w)

    def _draw_tile(
        self, canvas: Any, item: KpiItem, x: float, y: float, tile_w: float
    ) -> None:
        # A rule over the tile, nothing around it. Boxes and coloured spines are
        # what made these read as dashboard widgets rather than a report.
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.8)
        canvas.line(x, y + self.tile_height, x + tile_w, y + self.tile_height)

        inner = tile_w - 6.0
        label, _ = _fit_text(item.label.upper(), FONT_SANS_BOLD, SIZE_MICRO - 0.5, inner)
        _draw_tracked(
            canvas,
            x,
            y + self.tile_height - 14.0,
            label,
            FONT_SANS_BOLD,
            SIZE_MICRO - 0.5,
            MUTED,
            1.1,
        )

        value_baseline = y + (17.0 if item.note else 11.0)
        canvas.setFillColor(item.accent or INK)
        value, size = _fit_text(
            item.value or NOT_AVAILABLE, FONT_SANS_BOLD, SIZE_H1, inner, min_size=8.5
        )
        canvas.setFont(FONT_SANS_BOLD, size)
        canvas.drawString(x, value_baseline, value)

        if item.note:
            note, note_size = _fit_text(item.note, FONT_SANS, SIZE_MICRO, inner, min_size=6.0)
            canvas.setFillColor(MUTED)
            canvas.setFont(FONT_SANS, note_size)
            canvas.drawString(x, y + 6.0, note)


def _rows_that_fit(
    avail_height: float, row_height: float, gap: float, total_rows: int
) -> int:
    """
    How many chart rows to place before breaking, or 0 to move the whole chart.

    A single row stranded on the next page reads as a mistake, so the split
    keeps at least two rows on either side of the break.
    """
    per_row = row_height + gap
    if per_row <= 0:
        return 0
    fits = int((avail_height + gap) // per_row)
    if fits >= total_rows:
        return 0
    if total_rows - fits == 1:
        fits -= 1
    return fits if fits >= 1 else 0


def _bar_slice_height(row_count: int, row_height: float, gap: float) -> float:
    """
    The least room a bar chart needs to leave a band behind under a heading.

    :func:`_rows_that_fit` refuses to strand a lone row, so two rows have to fit
    before any row is placed. A chart of two rows or fewer cannot split at all,
    and the same expression happens to give its whole height.
    """
    keep = min(max(row_count, 0), 2)
    return keep * row_height + max(keep - 1, 0) * gap


@dataclass
class BarRow:
    """One row of a horizontal bar chart."""

    label: str
    value: float
    text: str = ""
    color: Optional[colors.Color] = None
    emphasis: bool = False


class HBarChart(Flowable):
    """Label, track, filled bar and a right-aligned value, one row per entry."""

    splittable = True


    def __init__(
        self,
        rows: Sequence[BarRow],
        row_height: float = 15.0,
        gap: float = 5.0,
        label_ratio: float = 0.34,
        value_width: float = 92.0,
        bar_height: float = 8.0,
        space_after: float = GAP_BLOCK,
    ) -> None:
        super().__init__()
        self.rows = list(rows)
        self.row_height = row_height
        self.gap = gap
        self.label_ratio = label_ratio
        self.value_width = value_width
        self.bar_height = bar_height
        self.space_after = space_after
        self.width = 0.0
        self.height = 0.0

    def _content_height(self) -> float:
        if not self.rows:
            return 0.0
        return len(self.rows) * self.row_height + (len(self.rows) - 1) * self.gap

    def first_slice_height(self) -> float:
        """Smallest slice :meth:`split` can leave behind."""
        return _bar_slice_height(len(self.rows), self.row_height, self.gap)

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        self.height = self._content_height() + (self.space_after if self.rows else 0.0)
        return self.width, self.height

    def split(self, availWidth: float, availHeight: float) -> List[Flowable]:
        if len(self.rows) <= 1:
            return []
        fits = _rows_that_fit(availHeight, self.row_height, self.gap, len(self.rows))
        if fits <= 0:
            return []
        head = HBarChart(
            self.rows[:fits], self.row_height, self.gap, self.label_ratio,
            self.value_width, self.bar_height, self.gap,
        )
        tail = HBarChart(
            self.rows[fits:], self.row_height, self.gap, self.label_ratio,
            self.value_width, self.bar_height, self.space_after,
        )
        return [head, tail]

    def draw(self) -> None:
        if not self.rows:
            return
        canvas = self.canv
        label_w = max(self.width * self.label_ratio, 60.0)
        track_x = label_w + 8.0
        track_w = max(self.width - track_x - self.value_width - 6.0, 20.0)
        # Trailing space belongs under the block: a flowable draws from the top
        # of the box it was given, so the leftover has to fall out at the bottom.
        top = self.height
        for index, row in enumerate(self.rows):
            y = top - (index + 1) * self.row_height - index * self.gap
            mid = y + (self.row_height / 2.0)
            bar_y = mid - (self.bar_height / 2.0)

            label, size = _fit_text(row.label, FONT_SANS, SIZE_SMALL, label_w, min_size=6.5)
            canvas.setFillColor(INK if row.emphasis else colors.HexColor("#3A4550"))
            canvas.setFont(FONT_SANS_BOLD if row.emphasis else FONT_SANS, size)
            canvas.drawString(0, mid - (size * 0.35), label)

            canvas.setFillColor(HAIRLINE)
            canvas.rect(track_x, bar_y, track_w, self.bar_height, stroke=0, fill=1)

            ratio = min(max(_as_float(row.value) or 0.0, 0.0), 1.0)
            filled = track_w * ratio
            if filled > 0.4:
                canvas.setFillColor(row.color or ACCENT)
                canvas.rect(track_x, bar_y, filled, self.bar_height, stroke=0, fill=1)

            # Every bar carries its own value: a bar the reader has to measure
            # against a track is not a chart, it is a decoration.
            text = row.text or f"{ratio * 100:.1f}%"
            value, value_size = _fit_text(text, FONT_MONO, SIZE_MICRO + 0.5, self.value_width, min_size=6.0)
            canvas.setFillColor(INK)
            canvas.setFont(FONT_MONO, value_size)
            canvas.drawRightString(self.width, mid - (value_size * 0.35), value)


@dataclass
class DivRow:
    """One row of a diverging bar chart: a signed value against a shared scale."""

    label: str
    value: float
    text: str = ""


class DivergingBarChart(Flowable):
    """
    Signed bars around a centred zero rule, scaled to the largest magnitude.

    The scale is the only place in the document where two hues carry meaning:
    below the normative mean on the left, above it on the right. When ``axis``
    is set the chart prints its own end labels, so the reader never has to infer
    what the width of a bar is worth.
    """

    splittable = True

    AXIS_HEIGHT = 13.0

    def __init__(
        self,
        rows: Sequence[DivRow],
        row_height: float = 15.0,
        gap: float = 4.0,
        label_ratio: float = 0.42,
        value_width: float = 40.0,
        bar_height: float = 8.0,
        domain: Optional[float] = None,
        space_after: float = GAP_BLOCK,
        axis: bool = False,
    ) -> None:
        super().__init__()
        self.rows = list(rows)
        self.row_height = row_height
        self.gap = gap
        self.label_ratio = label_ratio
        self.value_width = value_width
        self.bar_height = bar_height
        self.domain = domain or self._auto_domain()
        self.space_after = space_after
        self.axis = bool(axis)
        self.width = 0.0
        self.height = 0.0

    def _auto_domain(self) -> float:
        magnitudes = [abs(_as_float(row.value) or 0.0) for row in self.rows]
        peak = max(magnitudes) if magnitudes else 0.0
        return peak if peak > 0.001 else 1.0

    def _content_height(self) -> float:
        if not self.rows:
            return 0.0
        return len(self.rows) * self.row_height + (len(self.rows) - 1) * self.gap

    def _axis_height(self) -> float:
        return self.AXIS_HEIGHT if (self.axis and self.rows) else 0.0

    def first_slice_height(self) -> float:
        """Smallest slice :meth:`split` can leave behind."""
        return _bar_slice_height(len(self.rows), self.row_height, self.gap)

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        self.height = (
            self._content_height()
            + self._axis_height()
            + (self.space_after if self.rows else 0.0)
        )
        return self.width, self.height

    def split(self, availWidth: float, availHeight: float) -> List[Flowable]:
        if len(self.rows) <= 1:
            return []
        fits = _rows_that_fit(availHeight, self.row_height, self.gap, len(self.rows))
        if fits <= 0:
            return []
        common = dict(
            row_height=self.row_height, gap=self.gap, label_ratio=self.label_ratio,
            value_width=self.value_width, bar_height=self.bar_height, domain=self.domain,
        )
        # The axis belongs under the last band, so only the tail carries it.
        head = DivergingBarChart(self.rows[:fits], space_after=self.gap, **common)
        tail = DivergingBarChart(
            self.rows[fits:], space_after=self.space_after, axis=self.axis, **common
        )
        return [head, tail]

    @staticmethod
    def _bar_color(value: float) -> colors.Color:
        return DEV_LOW if value < 0 else DEV_HIGH

    def draw(self) -> None:
        if not self.rows:
            return
        canvas = self.canv
        label_w = max(self.width * self.label_ratio, 70.0)
        chart_x = label_w + 8.0
        chart_w = max(self.width - chart_x - self.value_width - 6.0, 30.0)
        center = chart_x + (chart_w / 2.0)
        half = chart_w / 2.0
        # Trailing space belongs under the block: a flowable draws from the top
        # of the box it was given, so the leftover has to fall out at the bottom.
        top = self.height
        bottom = top - self._content_height() - self._axis_height()

        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.7)
        canvas.line(center, bottom, center, top)

        for index, row in enumerate(self.rows):
            y = top - (index + 1) * self.row_height - index * self.gap
            mid = y + (self.row_height / 2.0)
            bar_y = mid - (self.bar_height / 2.0)
            value = _as_float(row.value) or 0.0
            length = min(abs(value) / self.domain, 1.0) * half if self.domain else 0.0

            label, size = _fit_text(row.label, FONT_SANS, SIZE_MICRO + 0.5, label_w, min_size=5.5)
            canvas.setFillColor(colors.HexColor("#3A4550"))
            canvas.setFont(FONT_SANS, size)
            canvas.drawString(0, mid - (size * 0.35), label)

            if length > 0.3:
                canvas.setFillColor(self._bar_color(value))
                x = center - length if value < 0 else center
                canvas.rect(x, bar_y, max(length, 0.8), self.bar_height, stroke=0, fill=1)

            canvas.setFillColor(INK)
            canvas.setFont(FONT_MONO, SIZE_MICRO)
            canvas.drawRightString(self.width, mid - (SIZE_MICRO * 0.35), row.text or fmt_z(value))

        if self._axis_height():
            self._draw_axis(canvas, center, half, bottom)

    def _draw_axis(self, canvas: Any, center: float, half: float, bottom: float) -> None:
        baseline = bottom + 3.0
        canvas.setFillColor(MUTED)
        canvas.setFont(FONT_MONO, SIZE_MICRO - 0.5)
        canvas.drawString(center - half, baseline, f"{-self.domain:.1f}")
        canvas.drawCentredString(center, baseline, "0")
        canvas.drawRightString(center + half, baseline, f"+{self.domain:.1f}")


class ScoreMeter(Flowable):
    """A segmented 0-1 meter with the numeric value alongside."""

    def __init__(
        self,
        value01: Optional[float],
        segments: int = 10,
        height: float = 12.0,
        label: str = "",
        space_after: float = GAP_BLOCK,
    ) -> None:
        super().__init__()
        self.value01 = _as_float(value01)
        self.segments = max(int(segments), 1)
        self.bar_height = height
        self.label = _s(label, "")
        self.space_after = space_after
        self.width = 0.0
        self.height = height + space_after + (LEAD_MICRO if self.label else 0.0)

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        return self.width, self.height

    def _color(self) -> colors.Color:
        # A weak composite score is a finding, not a mood: it is the only case
        # that earns the alert colour here.
        return ALERT if (self.value01 or 0.0) < 0.6 else ACCENT

    def draw(self) -> None:
        canvas = self.canv
        value_width = 76.0
        track_w = max(self.width - value_width - 8.0, 40.0)
        gap = 2.5
        seg_w = (track_w - gap * (self.segments - 1)) / float(self.segments)
        y = self.space_after

        if self.label:
            _draw_tracked(
                canvas,
                0,
                y + self.bar_height + 4.0,
                self.label.upper(),
                FONT_SANS_BOLD,
                SIZE_MICRO - 0.5,
                MUTED,
                1.1,
            )

        filled = 0 if self.value01 is None else int(round(self.value01 * self.segments))
        color = self._color()
        for index in range(self.segments):
            x = index * (seg_w + gap)
            canvas.setFillColor(color if index < filled else HAIRLINE)
            canvas.rect(x, y, seg_w, self.bar_height, stroke=0, fill=1)

        # The scale is stated with the value so the meter needs no legend.
        canvas.setFillColor(INK if self.value01 is not None else MUTED)
        canvas.setFont(FONT_SANS_BOLD, SIZE_H3)
        text = "not recorded" if self.value01 is None else f"{self.value01:.2f} of 1.00"
        value, size = _fit_text(text, FONT_SANS_BOLD, SIZE_H3, value_width, min_size=6.5)
        canvas.setFont(FONT_SANS_BOLD, size)
        canvas.drawRightString(self.width, y + 2.0, value)


def _is_splittable(flowable: Any) -> bool:
    """Whether a block can give a heading company instead of jumping wholesale."""
    if isinstance(flowable, Table):
        return len(getattr(flowable, "_cellvalues", ()) or ()) > 2
    return bool(getattr(flowable, "splittable", False))


def _first_slice_height(block: Any, aW: float, aH: float) -> Optional[float]:
    """
    The smallest slice ``block`` can leave behind under a heading.

    A heading may only stay at the foot of a page if the block it introduces can
    actually put its first row, or its first band, underneath it. A table needs
    its repeated header rows plus one body row; the chart flowables answer for
    themselves. Anything that cannot say returns ``None``.
    """
    try:
        own = getattr(block, "first_slice_height", None)
        if callable(own):
            return _as_float(own())
        if isinstance(block, Table):
            if not getattr(block, "_rowHeights", None):
                block.wrapOn(None, aW, aH)
            heights = [_as_float(value) or 0.0 for value in getattr(block, "_rowHeights", ()) or ()]
            if not heights:
                return None
            repeat = getattr(block, "repeatRows", 0)
            # ReportLab accepts either a count or an explicit set of row indices.
            frozen = int(repeat) if isinstance(repeat, int) else (max(repeat) + 1 if repeat else 0)
            keep = min(len(heights), max(frozen, 0) + 1)
            return sum(heights[:keep])
    except Exception:
        return None
    return None


class KeepHeadingWith(KeepTogether):
    """
    Orphan control that does not waste half a page.

    Plain ``KeepTogether`` moves a heading and a tall table to the next page as a
    unit, which leaves a large hole behind. This variant keeps them together only
    when the block cannot split; when it can, the heading stays put as long as
    the block's own first slice, a table's header plus first row or a chart's
    first band, still fits underneath it. Every ``keepWithNext`` run in the
    document is handed to this class by the template, so the numbered section
    headers and the sub-headings inside a section behave identically.
    """

    #: Floor for how much of the following block must fit under a heading.
    MIN_TAIL = 54.0

    def _head_height(self, content: Sequence[Flowable], aW: float, aH: float) -> float:
        """Height of everything before the trailing block, spacing included."""
        total = 0.0
        previous_after = 0.0
        try:
            for index, item in enumerate(content[:-1]):
                height = item.wrapOn(None, aW, aH)[1]
                if index:
                    total += max(float(item.getSpaceBefore()) - previous_after, 0.0)
                total += height
                previous_after = float(item.getSpaceAfter())
                total += previous_after
            total += max(float(content[-1].getSpaceBefore()) - previous_after, 0.0)
        except Exception:
            return float(self._H0)
        return total

    def _required_tail(self, tail: Any, aW: float, aH: float) -> float:
        """How much room the trailing block needs for the heading to stay put."""
        measured = _first_slice_height(tail, aW, aH)
        return self.MIN_TAIL if measured is None else max(self.MIN_TAIL, measured)

    def split(self, aW: float, aH: float) -> List[Flowable]:
        if getattr(self, "_wrapInfo", None) != (aW, aH):
            self.wrap(aW, aH)
        content = list(self._content)
        if self._H <= aH:
            return content
        tail = content[-1] if len(content) > 1 else None
        room = aH - self._head_height(content, aW, aH)
        if _is_splittable(tail) and room >= self._required_tail(tail, aW, aH):
            return content
        frame = getattr(self, "_frame", None)
        if getattr(frame, "_atTop", False):
            return content  # Already at the top of a frame: breaking would loop.
        content.insert(0, self.FrameBreak())
        return content


# ============================================================================
# Table helpers
# ============================================================================

#: No boxes, no vertical rules, no tinted bands: a header rule and light row
#: separators are the whole vocabulary.
_BASE_TABLE_STYLE: List[Tuple[Any, ...]] = [
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LINEBELOW", (0, 0), (-1, -2), 0.35, HAIRLINE),
]


def make_table(
    data: List[List[Any]],
    col_widths: Sequence[float],
    header: bool = True,
    extra_styles: Optional[Sequence[Tuple[Any, ...]]] = None,
    space_after: float = GAP_BLOCK,
) -> Flowable:
    """A table with the document's shared look: a header rule and hairlines."""
    if not data:
        return Spacer(1, 0)
    style: List[Tuple[Any, ...]] = list(_BASE_TABLE_STYLE)
    if header:
        style.extend(
            [
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, RULE),
                ("TOPPADDING", (0, 0), (-1, 0), 0),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
            ]
        )
    if extra_styles:
        style.extend(extra_styles)
    table = Table(
        data,
        colWidths=list(col_widths),
        repeatRows=1 if header else 0,
        hAlign="LEFT",
        style=TableStyle(style),
    )
    # Trailing space rides on the flowable itself so long tables still split
    # natively across pages instead of being wrapped in a group.
    table.spaceAfter = max(space_after, 0.0)
    return table


def header_row(labels: Sequence[str], numeric: Sequence[int] = ()) -> List[Paragraph]:
    """Header cells. Columns listed in ``numeric`` align with their figures."""
    right = set(numeric)
    return [
        P(label.upper(), "TableHeaderRight" if index in right else "TableHeader")
        for index, label in enumerate(labels)
    ]


def totals_row_style(row_index: int) -> List[Tuple[Any, ...]]:
    """A summed final row: a rule above it and bold figures, no fill."""
    return [
        ("LINEABOVE", (0, row_index), (-1, row_index), 0.8, RULE),
        ("TOPPADDING", (0, row_index), (-1, row_index), 5),
    ]


def spec_table(
    pairs: Sequence[Tuple[str, Any]],
    columns: int = 2,
    mono: bool = False,
    space_after: float = GAP_BLOCK,
) -> Flowable:
    """
    A definition list, laid out over ``columns`` pairs per row.

    Short facts do not deserve a "Field / Value" header and a full-width row
    each. Two pairs to a line halves the height of every specification block in
    the document and reads as a designed grid rather than a dump.
    """
    entries = [(k, _s(v, NOT_AVAILABLE)) for k, v in pairs if _s(k)]
    if not entries:
        return Spacer(1, 0)
    count = max(int(columns), 1)
    value_style = "SpecValueMono" if mono else "SpecValue"
    rows: List[List[Any]] = []
    for start in range(0, len(entries), count):
        line: List[Any] = []
        for index in range(count):
            if start + index < len(entries):
                key, value = entries[start + index]
                line.extend([P(key, "SpecKey"), P(value, value_style)])
            else:
                line.extend(["", ""])
        rows.append(line)
    pair_width = CONTENT_WIDTH / float(count)
    key_width = pair_width * (0.46 if count > 1 else 0.24)
    widths: List[float] = []
    for _ in range(count):
        widths.extend([key_width, pair_width - key_width])
    return make_table(
        rows,
        widths,
        header=False,
        extra_styles=[("RIGHTPADDING", (0, 0), (-1, -1), 12)],
        space_after=space_after,
    )


def bullet_list(
    items: Sequence[Any],
    style_name: str = "Bullet",
    bullet: str = "•",
    limit: int = 0,
) -> List[Flowable]:
    """Bulleted paragraphs, optionally truncated with an explicit elision note."""
    flowables: List[Flowable] = []
    entries = [entry for entry in items if _s(entry)]
    shown = entries[:limit] if limit else entries
    for entry in shown:
        flowables.append(Paragraph(escape(_s(entry)), _style(_styles(), style_name), bulletText=bullet))
    if limit and len(entries) > limit:
        flowables.append(P(f"and {len(entries) - limit:,} more not shown", "Micro"))
    return flowables


def numbered_list(items: Sequence[Any], limit: int = 0) -> List[Flowable]:
    """Numbered paragraphs, optionally truncated with an explicit elision note."""
    flowables: List[Flowable] = []
    entries = [entry for entry in items if _s(entry)]
    shown = entries[:limit] if limit else entries
    for index, entry in enumerate(shown, start=1):
        flowables.append(
            Paragraph(escape(_s(entry)), _style(_styles(), "Numbered"), bulletText=f"{index}.")
        )
    if limit and len(entries) > limit:
        flowables.append(P(f"and {len(entries) - limit:,} more not shown", "Micro"))
    return flowables


def missing_note(text: str) -> Flowable:
    """The single, consistent way this document reports absent information."""
    note = P(text, "SmallMuted")
    note.spaceAfter = GAP_TIGHT
    return note


def sentence_list(items: Sequence[str]) -> str:
    """``a``, ``b`` and ``c``, for the one-line absence note each section ends on."""
    entries = [_s(item) for item in items if _s(item)]
    if not entries:
        return ""
    if len(entries) == 1:
        return entries[0]
    return ", ".join(entries[:-1]) + " and " + entries[-1]


# ============================================================================
# Markdown subset renderer
# ============================================================================

_RE_CODE_SPAN = re.compile(r"`([^`]+)`")
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_RE_BOLD_ALT = re.compile(r"__(.+?)__", re.DOTALL)
_RE_ITALIC = re.compile(r"(?<![\*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)")
_RE_ITALIC_ALT = re.compile(r"(?<![_\w])_(?!\s)([^_\n]+?)(?<!\s)_(?!\w)")
_RE_LINK = re.compile(r"\[([^\]\n]*)\]\(([^)\s]+)[^)]*\)")
_RE_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
_RE_BULLET = re.compile(r"^(\s*)([-*+])\s+(.*)$")
_RE_ORDERED = re.compile(r"^(\s*)(\d{1,3})[.)]\s+(.*)$")
_RE_HRULE = re.compile(r"^\s*([-*_])\1{2,}\s*$")
_RE_FENCE = re.compile(r"^\s*(```|~~~)")
_RE_TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")

_CODE_SENTINEL = "\x01CODE{}\x01"


def inline_markdown(text: str) -> str:
    """Escape, then apply the inline subset: code, bold, italic, links."""
    escaped = escape(text)
    spans: List[str] = []

    def _stash(match: "re.Match[str]") -> str:
        spans.append(match.group(1))
        return _CODE_SENTINEL.format(len(spans) - 1)

    staged = _RE_CODE_SPAN.sub(_stash, escaped)
    staged = _RE_BOLD.sub(r"<b>\1</b>", staged)
    staged = _RE_BOLD_ALT.sub(r"<b>\1</b>", staged)
    staged = _RE_ITALIC.sub(r"<i>\1</i>", staged)
    staged = _RE_ITALIC_ALT.sub(r"<i>\1</i>", staged)
    staged = _RE_LINK.sub(rf'<link href="\2" color="{ACCENT_HEX}">\1</link>', staged)
    for index, span in enumerate(spans):
        replacement = (
            f'<font face="{FONT_MONO}" size="{SIZE_SMALL}" color="{ACCENT_HEX}">{span}</font>'
        )
        staged = staged.replace(_CODE_SENTINEL.format(index), replacement)
    return staged


def _code_line(line: str) -> str:
    """Escape one code line, preserving its leading indentation."""
    expanded = line.replace("\t", "    ")
    stripped = expanded.lstrip(" ")
    indent = len(expanded) - len(stripped)
    if not stripped:
        return "&nbsp;"
    return ("&nbsp;" * indent) + escape(stripped)


def _code_block(lines: Sequence[str], styles: Any) -> Flowable:
    """A fenced code block: monospaced text on a tinted, hairline-boxed panel."""
    body = "<br/>".join(_code_line(line) for line in lines) or "&nbsp;"
    para = Paragraph(body, _style(styles, "Code"))
    return make_table(
        [[para]],
        [CONTENT_WIDTH],
        header=False,
        extra_styles=[
            ("BACKGROUND", (0, 0), (-1, -1), PAGE_TINT),
            ("BOX", (0, 0), (-1, -1), 0.6, HAIRLINE),
            ("LEFTPADDING", (0, 0), (-1, -1), CODE_PADDING),
            ("RIGHTPADDING", (0, 0), (-1, -1), CODE_PADDING),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -1), 0, PAGE_TINT),
        ],
        space_after=GAP_BLOCK,
    )


#: Continuation indent for a wrapped shell command, and the trailing escape.
_SHELL_INDENT = "  "
_SHELL_CONTINUATION = " \\"


def _code_fits(line: str) -> bool:
    """Whether one line of code sets inside a code panel without wrapping."""
    return stringWidth(line, FONT_MONO, SIZE_SMALL) <= CODE_WIDTH


def shell_command_lines(command: str, path: str, arguments: Sequence[str]) -> List[str]:
    """
    Wrap one shell command at argument boundaries so a reader can copy it.

    The code style wraps character by character, which is right for a log line
    and wrong for a command: it splits a path mid-token and the paste is broken.
    Wrapping here instead keeps every token whole, gives the path a line of its
    own because it is the one argument long enough to overflow alone, and ends
    each continued line with a backslash so the whole block still runs as typed.
    """
    groups: List[str] = [command]
    quoted = f'"{path}"'
    if _code_fits(_SHELL_INDENT + quoted + _SHELL_CONTINUATION):
        groups.append(_SHELL_INDENT + quoted)
    else:
        # A path too long for a line of its own would character-wrap and break
        # the paste, so the directory moves into a cd and only the leaf stays
        # as the argument. Both tokens are then short enough to survive.
        parent = str(Path(path).parent)
        leaf = Path(path).name or path
        groups = [f'cd "{parent}"', "&& " + command, _SHELL_INDENT + f'"{leaf}"']
    current = ""
    for argument in [_s(item) for item in arguments if _s(item)]:
        candidate = f"{current} {argument}" if current else argument
        if current and not _code_fits(_SHELL_INDENT + candidate + _SHELL_CONTINUATION):
            groups.append(_SHELL_INDENT + current)
            current = argument
        else:
            current = candidate
    if current:
        groups.append(_SHELL_INDENT + current)
    return [line + _SHELL_CONTINUATION for line in groups[:-1]] + groups[-1:]


def _split_table_row(line: str) -> List[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _markdown_table(block: Sequence[str], styles: Any) -> Flowable:
    """Render a GFM table block. The first row is the header."""
    rows = [_split_table_row(line) for line in block]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return Spacer(1, 0)
    columns = max(len(row) for row in rows)
    header_style = _style(styles, "TableHeader")
    cell_style = _style(styles, "TableCell")
    data: List[List[Any]] = []
    for index, row in enumerate(rows):
        padded = list(row) + [""] * (columns - len(row))
        style = header_style if index == 0 else cell_style
        data.append(
            [Paragraph(inline_markdown(cell) or "&nbsp;", style) for cell in padded]
        )
    width = CONTENT_WIDTH / float(columns)
    return make_table(data, [width] * columns, header=True)


def markdown_to_flowables(markdown_text: str, styles: Any) -> List[Flowable]:
    """
    Render a small markdown subset into ReportLab flowables.

    Supported: ATX headings ``#`` to ``####``, paragraphs, ``-``/``*``/``+``
    bullets with one nesting level, ``1.`` ordered lists, ``**bold**``,
    ``*italic*``, `` `code` ``, ``---`` rules, fenced code blocks and GFM
    tables. Anything else degrades to a paragraph. This never raises.

    Args:
        markdown_text: Raw markdown source. May be empty.
        styles: A style dict or ReportLab stylesheet. Missing names fall back to
            the module's own styles.

    Returns:
        The flowables, in document order. Empty input yields an empty list.
    """
    text = markdown_text if isinstance(markdown_text, str) else _s(markdown_text)
    if not text.strip():
        return []
    try:
        return _markdown_to_flowables(text, styles)
    except Exception:  # A malformed report must not sink the whole document.
        return [P(text[:8000], "Body", styles)]


def _markdown_to_flowables(text: str, styles: Any) -> List[Flowable]:
    heading_styles = {1: "MdH1", 2: "MdH2", 3: "MdH3", 4: "MdH4"}
    flowables: List[Flowable] = []
    paragraph: List[str] = []
    table_block: List[str] = []
    code_block: List[str] = []
    in_code = False

    def flush_paragraph() -> None:
        if paragraph:
            joined = " ".join(line.strip() for line in paragraph).strip()
            if joined:
                flowables.append(Paragraph(inline_markdown(joined), _style(styles, "Body")))
            paragraph.clear()

    def flush_table() -> None:
        if table_block:
            flowables.append(_markdown_table(table_block, styles))
            table_block.clear()

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for index, raw_line in enumerate(lines):
        line = raw_line.rstrip()

        if _RE_FENCE.match(line):
            if in_code:
                flowables.append(_code_block(code_block, styles))
                code_block = []
                in_code = False
            else:
                flush_paragraph()
                flush_table()
                in_code = True
            continue
        if in_code:
            code_block.append(raw_line)
            continue

        is_table_line = line.strip().startswith("|") and line.count("|") >= 2
        if is_table_line:
            flush_paragraph()
            if _RE_TABLE_DIVIDER.match(line):
                continue  # The separator row only marks the header boundary.
            table_block.append(line)
            continue
        flush_table()

        if not line.strip():
            flush_paragraph()
            continue

        if _RE_HRULE.match(line):
            flush_paragraph()
            flowables.append(HRule(space_before=GAP_TIGHT, space_after=GAP_TIGHT))
            continue

        heading = _RE_HEADING.match(line)
        if heading:
            flush_paragraph()
            level = min(len(heading.group(1)), 4)
            flowables.append(
                Paragraph(inline_markdown(heading.group(2)), _style(styles, heading_styles[level]))
            )
            continue

        bullet = _RE_BULLET.match(line)
        if bullet:
            flush_paragraph()
            nested = len(bullet.group(1).replace("\t", "  ")) >= 2
            flowables.append(
                Paragraph(
                    inline_markdown(bullet.group(3)),
                    _style(styles, "BulletNested" if nested else "Bullet"),
                    bulletText="-" if nested else "•",
                )
            )
            continue

        ordered = _RE_ORDERED.match(line)
        if ordered:
            flush_paragraph()
            flowables.append(
                Paragraph(
                    inline_markdown(ordered.group(3)),
                    _style(styles, "Numbered"),
                    bulletText=f"{ordered.group(2)}.",
                )
            )
            continue

        paragraph.append(line)

    if in_code and code_block:
        flowables.append(_code_block(code_block, styles))
    flush_paragraph()
    flush_table()

    return flowables


# ============================================================================
# Run data extraction
# ============================================================================

_Z_IN_TEXT = re.compile(r"\bz\s*=\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)

#: The direction of a finding is signed data, so it reads on the same
#: diverging scale as the deviation chart and gains no colour of its own.
_DIRECTION_COLORS: Dict[str, colors.Color] = {
    "ABNORMAL_HIGH": DEV_HIGH,
    "ABNORMAL_LOW": DEV_LOW,
    "NORMAL": MUTED,
}


@dataclass
class NodeRow:
    """One node of a (possibly hierarchical) prediction, flattened for layout."""

    depth: int
    node_id: str
    path: str
    mode: str
    output: str
    confidence_level: str
    confidence_score: Optional[float]
    payload: Dict[str, Any] = field(default_factory=dict)


def _node_output_text(node: Dict[str, Any]) -> str:
    """The human-readable primary output of a single prediction node."""
    classification = _as_dict(node.get("classification"))
    if classification:
        label = _s(classification.get("predicted_label"), NOT_AVAILABLE)
        probs = _as_dict(classification.get("probabilities"))
        top = _as_float(probs.get(label)) if probs else None
        return f"{label} ({fmt_prob(top)})" if top is not None else label
    regression = _as_dict(node.get("regression"))
    values = _as_dict(regression.get("values"))
    if values:
        return ", ".join(f"{_s(k)}: {fmt_number(v)}" for k, v in values.items())
    return NOT_AVAILABLE


def flatten_nodes(node: Any, depth: int = 0) -> List[NodeRow]:
    """Depth-first flattening of the NodePrediction tree."""
    payload = _as_dict(node)
    if not payload or depth > MAX_TREE_DEPTH:
        return []
    rows = [
        NodeRow(
            depth=depth,
            node_id=_s(payload.get("node_id"), "node"),
            path=_s(payload.get("path"), _s(payload.get("node_id"), "node")),
            mode=_s(payload.get("mode"), "unknown"),
            output=_node_output_text(payload),
            confidence_level=_s(payload.get("confidence_level"), NOT_AVAILABLE),
            confidence_score=_as_float(payload.get("confidence_score")),
            payload=payload,
        )
    ]
    for child in _as_list(payload.get("children")):
        rows.extend(flatten_nodes(child, depth + 1))
    return rows


def flatten_task_nodes(node: Any, depth: int = 0) -> List[Tuple[int, Dict[str, Any]]]:
    """Depth-first flattening of the prediction task spec tree."""
    payload = _as_dict(node)
    if not payload or depth > MAX_TREE_DEPTH:
        return []
    rows = [(depth, payload)]
    for child in _as_list(payload.get("children")):
        rows.extend(flatten_task_nodes(child, depth + 1))
    return rows


@dataclass
class DeviationLeaf:
    """A scored leaf of the hierarchical deviation map."""

    domain: str
    path: str
    score: float


def walk_deviation_map(deviation_map: Any) -> List[DeviationLeaf]:
    """
    Collect every scored leaf of the deviation map, generically.

    Internal nodes may carry a ``_stats`` dictionary; it is bookkeeping, never a
    child, so it is excluded from iteration. A node that has both a numeric
    ``score`` and dictionary children contributes its own score and is still
    descended into.
    """
    leaves: List[DeviationLeaf] = []
    root = _as_dict(deviation_map)
    if not root:
        return leaves

    def visit(node: Any, domain: str, trail: List[str], depth: int) -> None:
        payload = _as_dict(node)
        if not payload or depth > MAX_TREE_DEPTH:
            return
        children = {
            key: value
            for key, value in payload.items()
            if key != "_stats" and isinstance(value, dict)
        }
        score = _as_float(payload.get("score"))
        if score is not None:
            leaves.append(DeviationLeaf(domain=domain, path=".".join(trail) or domain, score=score))
        for key, value in children.items():
            visit(value, domain, trail + [_s(key, "?")], depth + 1)

    for domain, subtree in root.items():
        domain_name = _s(domain, "?")
        if isinstance(subtree, dict):
            visit(subtree, domain_name, [domain_name], 0)
        else:
            score = _as_float(subtree)
            if score is not None:
                leaves.append(DeviationLeaf(domain=domain_name, path=domain_name, score=score))
    return leaves


def summarize_deviation_domains(
    leaves: Sequence[DeviationLeaf],
) -> List[Tuple[str, int, float, float]]:
    """Per top-level domain: leaf count, mean absolute score, peak magnitude."""
    grouped: Dict[str, List[float]] = {}
    for leaf in leaves:
        grouped.setdefault(leaf.domain, []).append(leaf.score)
    summary: List[Tuple[str, int, float, float]] = []
    for domain, scores in grouped.items():
        magnitudes = [abs(score) for score in scores]
        mean_abs = sum(magnitudes) / len(magnitudes) if magnitudes else 0.0
        summary.append((domain, len(scores), mean_abs, max(magnitudes) if magnitudes else 0.0))
    summary.sort(key=lambda row: row[2], reverse=True)
    return summary


def _collect_from_nodes(rows: Sequence[NodeRow], key: str) -> List[str]:
    """Gather a list-valued field across every node, de-duplicated in order."""
    collected: List[str] = []
    seen = set()
    for row in rows:
        for entry in _as_list(row.payload.get(key)):
            text = _s(entry)
            if text and text not in seen:
                seen.add(text)
                collected.append(text)
    return collected


def _collect_key_findings(
    node_rows: Sequence[NodeRow], patient_report: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Key findings from the prediction tree, falling back to the report file."""
    findings: List[Dict[str, Any]] = []
    for row in node_rows:
        for entry in _as_list(row.payload.get("key_findings")):
            item = _as_dict(entry)
            if item:
                findings.append(item)
    if not findings:
        findings = [_as_dict(entry) for entry in _as_list(patient_report.get("key_findings"))]
    return [item for item in findings if item]


def _finding_z(item: Dict[str, Any]) -> Optional[float]:
    """The z-score of a finding: explicit field first, else parsed from the text."""
    for key in ("z_score", "z", "zscore"):
        value = _as_float(item.get(key))
        if value is not None:
            return value
    match = _Z_IN_TEXT.search(_s(item.get("finding")))
    return _as_float(match.group(1)) if match else None


def _group_token_calls(calls: Sequence[Any]) -> List[Tuple[str, int, int, int, int]]:
    """Token ledger grouped by component: (component, calls, prompt, completion, total)."""
    grouped: Dict[str, List[int]] = {}
    for entry in calls:
        call = _as_dict(entry)
        component = _s(call.get("component"), "unknown")
        prompt = _as_int(call.get("prompt_tokens")) or 0
        completion = _as_int(call.get("completion_tokens")) or 0
        total = _as_int(call.get("total"))
        if total is None:
            total = _as_int(call.get("total_tokens")) or (prompt + completion)
        bucket = grouped.setdefault(component, [0, 0, 0, 0])
        bucket[0] += 1
        bucket[1] += prompt
        bucket[2] += completion
        bucket[3] += total
    rows = [
        (component, values[0], values[1], values[2], values[3])
        for component, values in grouped.items()
    ]
    rows.sort(key=lambda row: row[4], reverse=True)
    return rows


# ============================================================================
# Page furniture, canvas and document template
# ============================================================================


@dataclass
class Furniture:
    """
    Everything the page callbacks draw outside the text frame.

    ``subtitle`` is the exception: it feeds the PDF's own subject metadata and is
    not drawn, because the task line it carries is printed in the cover's
    participant block instead.
    """

    title: str = "COMPASS Run Report"
    subtitle: str = ""
    participant_id: str = ""
    task_line: str = ""
    timestamp: str = ""
    footer: str = DISCLAIMER


class NumberedCanvas(pdfcanvas.Canvas):
    """Defers page output so every body page can print ``Page N of M``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_states: List[Dict[str, Any]] = []

    def showPage(self) -> None:  # noqa: N802 - ReportLab API name
        self._saved_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total = len(self._saved_states)
        for state in self._saved_states:
            self.__dict__.update(state)
            self._draw_page_number(total)
            super().showPage()
        self._saved_states = []
        super().save()

    def _draw_page_number(self, total: int) -> None:
        page = int(getattr(self, "_pageNumber", 1) or 1)
        if page <= 1:
            return  # The cover carries no pagination.
        self.saveState()
        self.setFont(FONT_SANS, SIZE_MICRO)
        self.setFillColor(MUTED)
        self.drawRightString(PAGE_WIDTH - MARGIN_SIDE, 30.0, f"{page} / {total}")
        self.restoreState()


def _furniture(doc: Any) -> Furniture:
    value = getattr(doc, "furniture", None)
    return value if isinstance(value, Furniture) else Furniture()


def _wave_path(canvas: Any, points: Sequence[Tuple[float, float]], floor: float) -> Any:
    """
    A smooth crest through `points`, closed down to `floor`.

    Control points are placed at the horizontal midpoint of each span, which
    turns a handful of coordinates into one continuous curve with no visible
    joins. Drawing the crest as a series of straight segments instead is what
    makes a decorative band read as a chart.
    """
    path = canvas.beginPath()
    path.moveTo(points[0][0], points[0][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        midpoint = (x0 + x1) / 2.0
        path.curveTo(midpoint, y0, midpoint, y1, x1, y1)
    path.lineTo(points[-1][0], floor)
    path.lineTo(points[0][0], floor)
    path.close()
    return path


#: Where the cover's wave field begins, as a share of the page height. The
#: title sits on clean paper above it, which is what keeps the page calm.
COVER_WAVE_TOP = 0.46
#: Crest heights of the three layers, as a share of the page height. They rise
#: to the right so the field reads as one diagonal sweep rather than a border.
COVER_WAVE_LAYERS: Tuple[Tuple[Tuple[float, ...], float, float], ...] = (
    ((0.30, 0.36, 0.31, 0.40, 0.46), 0.16, 1.00),
    ((0.24, 0.27, 0.35, 0.30, 0.38), 0.42, 1.00),
    ((0.17, 0.22, 0.19, 0.26, 0.29), 1.00, 1.00),
)


def draw_cover_wave(canvas: Any) -> None:
    """
    The cover's one piece of decoration: a diagonal field of layered waves.

    Three crests of the same accent, each lighter and higher than the one in
    front, rising left to right. They occupy the lower half of the page only,
    so the title above them stays on plain paper.
    """
    canvas.saveState()
    columns = 4
    for heights, lightness, alpha in COVER_WAVE_LAYERS:
        points = [
            (PAGE_WIDTH * (index / columns), PAGE_HEIGHT * height)
            for index, height in enumerate(heights)
        ]
        canvas.setFillColor(ACCENT if lightness >= 1.0 else tint(ACCENT, 1.0 - lightness))
        canvas.setFillAlpha(alpha)
        canvas.drawPath(_wave_path(canvas, points, 0.0), stroke=0, fill=1)
    canvas.setFillAlpha(1.0)
    canvas.restoreState()


def draw_cover_page(canvas: Any, doc: Any) -> None:
    """
    The title page: a wordmark, the title, and the wave field beneath it.

    Everything else a reader needs is one page further in. A title page that
    also carries the findings is not a title page, it is a first page with a
    larger heading, and it was the crowded version of this that read as a
    dashboard screenshot rather than as a document.
    """
    info = _furniture(doc)
    canvas.saveState()
    draw_cover_wave(canvas)

    _draw_tracked(
        canvas,
        MARGIN_SIDE,
        PAGE_HEIGHT - 52.0,
        "COMPASS ENGINE",
        FONT_SANS_BOLD,
        SIZE_MICRO,
        ACCENT,
        2.6,
    )
    if info.timestamp:
        canvas.setFillColor(MUTED)
        canvas.setFont(FONT_SANS, SIZE_MICRO)
        canvas.drawRightString(PAGE_WIDTH - MARGIN_SIDE, PAGE_HEIGHT - 52.0, info.timestamp)

    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(2.0)
    canvas.line(MARGIN_SIDE, PAGE_HEIGHT - 66.0, MARGIN_SIDE + 34.0, PAGE_HEIGHT - 66.0)

    title, size = _fit_text(info.title, FONT_SANS_BOLD, 32.0, CONTENT_WIDTH, min_size=18.0)
    canvas.setFillColor(INK)
    canvas.setFont(FONT_SANS_BOLD, size)
    canvas.drawString(MARGIN_SIDE, PAGE_HEIGHT * 0.72, title)

    baseline = PAGE_HEIGHT * 0.72 - size * 0.86
    if info.participant_id:
        identifier, id_size = _fit_text(
            info.participant_id, FONT_MONO, SIZE_H1, CONTENT_WIDTH, min_size=9.0
        )
        canvas.setFillColor(ACCENT)
        canvas.setFont(FONT_MONO, id_size)
        canvas.drawString(MARGIN_SIDE, baseline, identifier)
        baseline -= id_size + 6.0
    if info.task_line:
        task, task_size = _fit_text(
            info.task_line, FONT_SANS, SIZE_BODY, CONTENT_WIDTH, min_size=7.5
        )
        canvas.setFillColor(MUTED)
        canvas.setFont(FONT_SANS, task_size)
        canvas.drawString(MARGIN_SIDE, baseline, task)

    # Set on the deepest wave, so it reverses out rather than competing with
    # the title for the eye.
    canvas.setFillColor(tint(ACCENT, 0.72))
    canvas.setFont(FONT_SANS, SIZE_MICRO)
    canvas.drawString(MARGIN_SIDE, 34.0, f"Engine {_ENGINE_VERSION or 'version not recorded'}")
    # The full disclaimer runs under every body page. Here it shares a line with
    # the build, so the cover states the caveat without clipping it: the first
    # sentence carries the claim, and the second is one page away.
    supplied = _s(info.footer)
    caveat = supplied if supplied and supplied != DISCLAIMER else COVER_CAVEAT
    caveat, _ = _fit_text(caveat, FONT_SANS, SIZE_MICRO, CONTENT_WIDTH * 0.70)
    canvas.drawRightString(PAGE_WIDTH - MARGIN_SIDE, 34.0, caveat)
    canvas.restoreState()


def draw_body_page(canvas: Any, doc: Any) -> None:
    """
    The running header and footer, and nothing else.

    Participant above, disclaimer and folio below. Anything more on a body page
    competes with the content for the reader's attention.
    """
    info = _furniture(doc)
    canvas.saveState()
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT_MONO, SIZE_MICRO)
    identifier, _ = _fit_text(
        info.participant_id or "participant not identified",
        FONT_MONO,
        SIZE_MICRO,
        CONTENT_WIDTH - 140.0,
    )
    canvas.drawString(MARGIN_SIDE, PAGE_HEIGHT - 38.0, identifier)

    canvas.setStrokeColor(HAIRLINE)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN_SIDE, PAGE_HEIGHT - 46.0, PAGE_WIDTH - MARGIN_SIDE, PAGE_HEIGHT - 46.0)
    canvas.line(MARGIN_SIDE, 40.0, PAGE_WIDTH - MARGIN_SIDE, 40.0)

    canvas.setFillColor(MUTED)
    canvas.setFont(FONT_SANS, SIZE_MICRO)
    footer, _ = _fit_text(
        info.footer or DISCLAIMER, FONT_SANS, SIZE_MICRO, CONTENT_WIDTH - 60.0
    )
    canvas.drawString(MARGIN_SIDE, 30.0, footer)
    canvas.restoreState()


class RunDocTemplate(BaseDocTemplate):
    """A4 document with a cover template and a paginated body template."""

    def __init__(self, filename: str, furniture: Furniture, **kwargs: Any) -> None:
        super().__init__(
            filename,
            pagesize=PAGE_SIZE,
            leftMargin=MARGIN_SIDE,
            rightMargin=MARGIN_SIDE,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title=furniture.title,
            author="COMPASS Engine",
            subject=furniture.subtitle or "COMPASS run report",
            **kwargs,
        )
        self.furniture = furniture
        # BaseDocTemplate seeds this as an instance attribute, so it has to be
        # replaced after init for keepWithNext runs to use our smarter class.
        self.keepTogetherClass = KeepHeadingWith
        cover_top = PAGE_HEIGHT - COVER_BAND_HEIGHT - COVER_FRAME_TOP_GAP
        cover_frame = Frame(
            MARGIN_SIDE,
            MARGIN_BOTTOM,
            CONTENT_WIDTH,
            cover_top - MARGIN_BOTTOM,
            id="cover",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        body_frame = Frame(
            MARGIN_SIDE,
            MARGIN_BOTTOM,
            CONTENT_WIDTH,
            PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(id="Cover", frames=[cover_frame], onPage=draw_cover_page),
                PageTemplate(id="Body", frames=[body_frame], onPage=draw_body_page),
            ]
        )


# ============================================================================
# Section builders
# ============================================================================


class _Sections:
    """Section numbering. Numbers follow ``SECTION_SPECS``, not call order, so a
    section that fails to build cannot shift the ones after it."""

    def __init__(self) -> None:
        self._fallback = 0

    def header(self, spec: Tuple[str, str]) -> SectionHeader:
        title, subtitle = spec
        try:
            number = SECTION_SPECS.index(spec) + 1
        except ValueError:
            self._fallback += 1
            number = self._fallback
        return SectionHeader(str(number), title, subtitle)


_HEADING_STYLES = frozenset({"H1", "H2", "H3", "MdH1", "MdH2", "MdH3", "MdH4"})


def _is_heading(flowable: Any) -> bool:
    style = getattr(flowable, "style", None)
    return bool(style is not None and getattr(style, "name", "") in _HEADING_STYLES)


class _Body:
    """
    A section body that leaves out what it cannot fill.

    A heading followed by a sentence saying nothing was recorded is noise nine
    times over, so a block with no content is dropped, heading included, and
    its name is kept back. The section then closes on one plain line naming
    everything that was absent, which reports the gap without spending a page
    on it.
    """

    def __init__(self) -> None:
        self.flowables: List[Flowable] = []
        self.absent: List[str] = []

    def add(self, *blocks: Optional[Flowable]) -> None:
        self.flowables.extend([block for block in blocks if block is not None])

    def extend(self, blocks: Sequence[Optional[Flowable]]) -> None:
        self.flowables.extend([block for block in blocks if block is not None])

    def block(
        self,
        heading: str,
        content: Sequence[Optional[Flowable]],
        absent_as: str = "",
    ) -> None:
        """Emit ``heading`` and ``content`` together, or neither."""
        items = [block for block in (content or ()) if block is not None]
        if items:
            if heading:
                self.flowables.append(P(heading, "H2"))
            self.flowables.extend(items)
        elif absent_as:
            self.absent.append(absent_as)

    def close(self) -> List[Flowable]:
        out = list(self.flowables)
        if self.absent:
            out.append(missing_note(f"Not recorded by this run: {sentence_list(self.absent)}."))
        return out


def _section(
    sections: _Sections, spec: Tuple[str, str], body: Sequence[Flowable]
) -> List[Flowable]:
    """A section header glued to its first block so headings never orphan."""
    title, _ = spec
    header = sections.header(spec)
    content = [flowable for flowable in body if flowable is not None]
    if not content:
        content = [missing_note(f"{title}: no information was recorded for this run.")]
    header.spaceAfter = GAP_BLOCK
    lead: List[Flowable] = [header, content[0]]
    rest = content[1:]
    # A section that opens on a subheading needs the block under it too, or the
    # subheading would be regrouped on its own and strand the section header.
    if rest and _is_heading(content[0]):
        lead.append(rest[0])
        rest = rest[1:]
    flowables: List[Flowable] = [Spacer(1, GAP_MAJOR)]
    flowables.append(KeepHeadingWith(lead))
    flowables.extend(rest)
    return flowables


def _stack_height(
    flowables: Sequence[Flowable], width: float = CONTENT_WIDTH
) -> Optional[float]:
    """
    Measure what a run of flowables occupies in a frame, spacing included.

    This mirrors the frame's own space merging closely enough to plan the cover.
    A small overestimate is the safe direction, so nothing here rounds down.
    Returns ``None`` if anything refuses to measure, which the caller reads as
    "do not try to fill the space".
    """
    total = 0.0
    previous_after = 0.0
    try:
        for index, item in enumerate(flowables):
            # ``wrapOn`` is the measuring entry point: it lends the flowable the
            # canvas attribute that some containers reach for during a wrap.
            height = item.wrapOn(None, width, COVER_FRAME_HEIGHT)[1]
            if isinstance(item, KeepTogether):
                # A keep-together group reports a sentinel height to force a
                # split; its real measurement is stashed on the instance.
                height = float(getattr(item, "_H", 0.0))
            if index:
                total += max(float(item.getSpaceBefore()) - previous_after, 0.0)
            total += height
            previous_after = float(item.getSpaceAfter())
            total += previous_after
    except Exception:
        return None
    return total


def _cover_flowables(_ctx: "_RunContext") -> List[Flowable]:
    """
    Nothing. The title page is drawn entirely on the canvas.

    A frame still has to receive something for the page to exist at all, so it
    receives a hairline spacer. Everything the old cover carried now opens the
    body, where it has room to breathe.
    """
    return [Spacer(1, 1)]


def _summary_flowables(ctx: "_RunContext") -> List[Flowable]:
    """
    The opening spread: what was asked, what came back, and what it rested on.

    This is what used to be crowded onto the title page under the wordmark. On
    a page of its own the KPI grid, the contents list and the source-data glance
    each get their own band of white space instead of competing for one frame.
    """
    core: List[Flowable] = [
        P("Summary", "H1"),
        P(ctx.task_line, "BodyMuted"),
        Spacer(1, GAP_BLOCK),
        KpiGrid(ctx.kpi_items(), cols=3, tile_height=58.0),
        Spacer(1, GAP_SECTION),
        HRule(space_before=0, space_after=GAP_TIGHT),
        P(ctx.provenance_line(), "Micro"),
        Spacer(1, GAP_MAJOR),
        _contents_block(),
        Spacer(1, GAP_SECTION),
        Callout(DISCLAIMER, ACCENT, title="Pre-clinical use only", space_after=0),
    ]

    measured = _stack_height(core)
    if measured is None:
        return core
    # The body frame is a full page rather than the old short cover frame, so
    # there is usually room for the glance block that used to be dropped.
    free = (PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM) - measured
    return core + _cover_glance(ctx, free)


def _cover_glance(ctx: "_RunContext", free: float) -> List[Flowable]:
    """
    The cover's closing block: what the run actually had to work with.

    Everything here is optional. Candidates are measured against ``free``, the
    room the rest of the cover left behind, and the richest one that genuinely
    fits wins, so a full cover gains nothing and a sparse one is not pushed past
    the frame. An empty result tells the caller to centre the cover instead.
    """
    if free < COVER_GLANCE_MIN_SPACE:
        return []

    def fits(candidate: Sequence[Flowable]) -> bool:
        measured = _stack_height(candidate) if candidate else None
        return measured is not None and measured + COVER_FIT_MARGIN <= free

    per_row = COVER_STRIP_ROW_HEIGHT + COVER_STRIP_GAP
    ceiling = min(len(ctx.domain_coverage), int(free // per_row) if per_row > 0 else 0)
    coverage_rows = 0
    best: List[Flowable] = []
    for rows in range(ceiling, 0, -1):
        candidate = _cover_glance_blocks(ctx, rows, 0)
        if fits(candidate):
            best, coverage_rows = candidate, rows
            break
    used = _stack_height(best) if best else 0.0
    if free - (used or 0.0) > COVER_DEVIATION_MIN_SPACE:
        candidate = _cover_glance_blocks(ctx, coverage_rows, COVER_DEVIATION_ROWS)
        if fits(candidate):
            return candidate
    return best


def _cover_glance_blocks(
    ctx: "_RunContext", coverage_rows: int, deviation_rows: int
) -> List[Flowable]:
    """One candidate glance block at the given row allowance. May be empty."""
    body: List[Flowable] = list(_cover_coverage_strip(ctx, coverage_rows))
    deviation = _cover_deviation_strip(ctx, deviation_rows)
    if deviation:
        if body:
            body.append(Spacer(1, GAP_BLOCK))
        body.extend(deviation)
    if not body:
        return []
    label = P("Run at a glance".upper(), "Micro")
    label.spaceAfter = 4.0
    return [Spacer(1, GAP_SECTION), HRule(space_before=0, space_after=GAP_TIGHT), label] + body


def _cover_coverage_strip(ctx: "_RunContext", rows: int) -> List[Flowable]:
    """The footprint of the source data: coverage for the first ``rows`` domains."""
    shown = ctx.domain_coverage[: max(int(rows), 0)]
    if not shown:
        return []
    caption = P(
        "Source data footprint: the share of the ontology leaves in each domain that "
        "carried a value for this participant.",
        "Micro",
    )
    caption.spaceAfter = 4.0
    flowables: List[Flowable] = [
        caption,
        HBarChart(
            _coverage_bar_rows(shown),
            row_height=COVER_STRIP_ROW_HEIGHT,
            gap=COVER_STRIP_GAP,
            label_ratio=0.30,
            value_width=100.0,
            bar_height=COVER_STRIP_BAR_HEIGHT,
            space_after=0.0,
        ),
    ]
    hidden = len(ctx.domain_coverage) - len(shown)
    if hidden > 0:
        flowables.append(P(f"and {hidden:,} further domains, charted in section 4", "Micro"))
    return flowables


def _cover_deviation_strip(ctx: "_RunContext", rows: int) -> List[Flowable]:
    """The handful of ontology leaves furthest from the normative mean."""
    leaves = ctx.deviation_leaves
    limit = min(max(int(rows), 0), COVER_DEVIATION_ROWS)
    if not leaves or limit < 1:
        return []
    ranked = sorted(leaves, key=lambda leaf: abs(leaf.score), reverse=True)[:limit]
    heading = P(f"Furthest from the normative mean ({len(ranked)} of {len(leaves):,} leaves)", "Micro")
    heading.spaceAfter = 4.0
    return [
        heading,
        DivergingBarChart(
            [DivRow(label=leaf.path, value=leaf.score, text=fmt_z(leaf.score)) for leaf in ranked],
            row_height=COVER_STRIP_ROW_HEIGHT,
            gap=COVER_STRIP_GAP,
            label_ratio=0.46,
            value_width=38.0,
            bar_height=COVER_STRIP_BAR_HEIGHT,
            space_after=0.0,
        ),
    ]


def _contents_block() -> Flowable:
    """The cover's contents list, built from the same specs as the headers."""
    cells: List[List[Any]] = []
    column_count = 3
    rows_needed = -(-len(SECTION_SPECS) // column_count)
    columns: List[List[Tuple[int, str]]] = [
        list(enumerate(SECTION_SPECS, start=1))[i * rows_needed : (i + 1) * rows_needed]
        for i in range(column_count)
    ]
    for row in range(rows_needed):
        line: List[Any] = []
        for column in columns:
            if row < len(column):
                number, (title, _) = column[row]
                line.append(
                    RP(
                        f'<font color="{ACCENT_HEX}" face="{FONT_SANS_BOLD}">{number}</font>'
                        f"&nbsp;&nbsp;{escape(title)}",
                        "Small",
                    )
                )
            else:
                line.append("")
        cells.append(line)
    width = CONTENT_WIDTH / float(column_count)
    table = make_table(
        cells,
        [width] * column_count,
        header=False,
        extra_styles=[
            ("LINEBELOW", (0, 0), (-1, -1), 0, PAPER),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ],
        space_after=0,
    )
    return _titled_block("In this report", table)


def _titled_block(label: str, *blocks: Flowable) -> Flowable:
    """
    A micro, letterspaced label above one or more blocks, kept with them.

    The blocks are passed in flat rather than pre-grouped: a keep-together
    nested inside another one reports a sentinel height during measurement,
    which the outer group reads as "does not fit" and answers with a page break.
    """
    heading = P(label.upper(), "Micro")
    heading.spaceAfter = 3.0
    return KeepHeadingWith([heading, *blocks])


def _probability_bars(node: Dict[str, Any]) -> List[Flowable]:
    """Class probabilities as labelled bars, highest first. May be empty."""
    classification = _as_dict(node.get("classification"))
    probabilities = _as_dict(classification.get("probabilities"))
    if not probabilities:
        return []
    predicted = _s(classification.get("predicted_label"))
    rows = [
        BarRow(
            label=_s(label, "?"),
            value=_as_float(value) or 0.0,
            text=fmt_prob(value, "not recorded"),
            color=ACCENT if _s(label) == predicted else tint(ACCENT, 0.55),
            emphasis=_s(label) == predicted,
        )
        for label, value in sorted(
            probabilities.items(), key=lambda item: _as_float(item[1]) or 0.0, reverse=True
        )
    ]
    return [HBarChart(rows, label_ratio=0.42, value_width=60.0)]


def _regression_table(node: Dict[str, Any], units: Dict[str, Any]) -> List[Flowable]:
    """Estimated values with their declared units. May be empty."""
    values = _as_dict(_as_dict(node.get("regression")).get("values"))
    if not values:
        return []
    rows = [header_row(["Output", "Value", "Unit"], numeric=[1])]
    for key, value in values.items():
        rows.append(
            [
                P(key, "TableCellMono"),
                P(fmt_number(value), "TableCellNumeric"),
                P(_s(units.get(key), "not declared"), "TableCellMuted"),
            ]
        )
    widths = [CONTENT_WIDTH * 0.46, CONTENT_WIDTH * 0.22, CONTENT_WIDTH * 0.32]
    return [make_table(rows, widths)]


def _prediction_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body = _Body()
    root_node = _as_dict(ctx.task_root)

    labels = [_s(item) for item in _as_list(root_node.get("class_labels")) if _s(item)]
    outputs = [_s(item) for item in _as_list(root_node.get("regression_outputs")) if _s(item)]
    task_id = _s(ctx.task_spec.get("task_id"))
    spec_pairs: List[Tuple[str, str]] = [
        ("Task", _s(root_node.get("display_name"), _s(ctx.performance.get("target_condition"), NOT_AVAILABLE))),
        ("Mode", humanize(root_node.get("mode")) if root_node else NOT_AVAILABLE),
    ]
    if _s(ctx.performance.get("target_condition")):
        spec_pairs.append(("Target label", _s(ctx.performance.get("target_condition"))))
    if _s(ctx.performance.get("control_condition")):
        spec_pairs.append(("Comparator label", _s(ctx.performance.get("control_condition"))))
    if labels:
        spec_pairs.append(("Class labels", ", ".join(labels)))
    if outputs:
        spec_pairs.append(("Regression outputs", ", ".join(outputs)))
    if len(ctx.node_rows) > 1:
        spec_pairs.append(("Prediction nodes", f"{len(ctx.node_rows):,}"))
    if task_id:
        spec_pairs.append(("Task id", task_id))
    body.add(spec_table(spec_pairs, columns=2))

    if ctx.node_rows or _s(ctx.primary_output) != NOT_AVAILABLE:
        confidence_bits = [f"Confidence {ctx.confidence_level.lower()}"]
        if ctx.root_confidence is not None:
            confidence_bits.append(f"score {fmt_prob(ctx.root_confidence)}")
        probability = _as_float(ctx.prediction_result.get("probability"))
        if probability is not None:
            confidence_bits.append(f"reported probability {fmt_prob(probability)}")
        body.add(
            _titled_block(
                "Primary output",
                P(ctx.primary_output, "BigValue"),
                P(", ".join(confidence_bits), "SmallMuted"),
            )
        )
        body.add(Spacer(1, GAP_TIGHT))
    else:
        body.add(missing_note("No prediction node was recorded for this run."))

    summary = _s(ctx.patient_report.get("clinical_summary"))
    body.block(
        "Clinical summary",
        [P(summary, "Body")] if summary else [],
        absent_as="a clinical summary",
    )

    root_row = ctx.node_rows[0] if ctx.node_rows else None
    if root_row is not None:
        bars = _probability_bars(root_row.payload)
        if bars:
            body.block("Class probabilities", bars)
        else:
            body.block(
                "Estimated values",
                _regression_table(root_row.payload, _as_dict(root_node.get("unit_by_output"))),
            )

    if len(ctx.node_rows) > 1:
        rows = [header_row(["Node", "Mode", "Output", "Confidence"])]
        for node in ctx.node_rows:
            # Only Latin-1 glyphs are safe in the built-in fonts, so the tree
            # marker is a dash rather than a box-drawing character.
            prefix = "" if node.depth == 0 else "- "
            label_style = _indented(_style(_styles(), "TableCellMono"), node.depth * 10.0)
            rows.append(
                [
                    Paragraph(escape(f"{prefix}{node.node_id}"), label_style),
                    P(humanize(node.mode), "TableCellMuted"),
                    P(node.output, "TableCell"),
                    P(
                        f"{node.confidence_level} {fmt_prob(node.confidence_score, '')}".strip(),
                        "TableCellMuted",
                    ),
                ]
            )
        widths = [
            CONTENT_WIDTH * 0.22,
            CONTENT_WIDTH * 0.20,
            CONTENT_WIDTH * 0.38,
            CONTENT_WIDTH * 0.20,
        ]
        # The per-node output already carries the winning label and its
        # probability, so a bar chart per child would repeat the table at length.
        body.block("Prediction tree", [make_table(rows, widths)])

    return _section(sections, SECTION_SPECS[0], body.close())


def _key_findings_table(findings: Sequence[Dict[str, Any]]) -> List[Flowable]:
    """Findings by domain, with the direction and the z-score that carried them."""
    if not findings:
        return []
    shown = list(findings)[:KEY_FINDING_CAP]
    rows = [header_row(["Domain", "Finding", "Direction", "z"], numeric=[3])]
    direction_styles: List[Tuple[Any, ...]] = []
    for index, item in enumerate(shown, start=1):
        direction = _s(item.get("direction"), "UNSPECIFIED").upper()
        color = _DIRECTION_COLORS.get(direction, MUTED)
        direction_styles.append(("TEXTCOLOR", (2, index), (2, index), color))
        rows.append(
            [
                P(_s(item.get("domain"), NOT_AVAILABLE), "TableCellLabel"),
                P(_s(item.get("finding"), NOT_AVAILABLE), "TableCell"),
                P(direction.replace("_", " ").title(), "TableCell"),
                P(fmt_z(_finding_z(item)), "TableCellNumeric"),
            ]
        )
    widths = [
        CONTENT_WIDTH * 0.22,
        CONTENT_WIDTH * 0.48,
        CONTENT_WIDTH * 0.19,
        CONTENT_WIDTH * 0.11,
    ]
    blocks: List[Flowable] = [make_table(rows, widths, extra_styles=direction_styles)]
    if len(findings) > len(shown):
        blocks.append(P(f"and {len(findings) - len(shown):,} further findings not shown", "Micro"))
    return blocks


def _evidence_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body = _Body()

    body.block("Key findings", _key_findings_table(ctx.key_findings), absent_as="key findings")
    body.block(
        "Reasoning chain",
        numbered_list(ctx.reasoning_chain, limit=REASONING_STEP_CAP),
        absent_as="a reasoning chain",
    )

    body.block(
        "Supporting evidence",
        _evidence_columns(ctx.evidence_for, ctx.evidence_against),
        absent_as="supporting evidence for or against",
    )
    return _section(sections, SECTION_SPECS[1], body.close())


def _evidence_columns(for_items: Sequence[str], against_items: Sequence[str]) -> List[Flowable]:
    """Evidence for and against side by side. Empty when neither was recorded."""
    if not for_items and not against_items:
        return []
    left: List[Flowable] = [P("For the prediction", "Micro")]
    left.extend(bullet_list(for_items) or [missing_note("None recorded.")])
    right: List[Flowable] = [P("Against the prediction", "Micro")]
    right.extend(bullet_list(against_items) or [missing_note("None recorded.")])
    half = CONTENT_WIDTH / 2.0
    return [
        make_table(
            [[left, right]],
            [half, half],
            header=False,
            extra_styles=[
                ("LINEBELOW", (0, 0), (-1, -1), 0, PAPER),
                ("RIGHTPADDING", (0, 0), (0, -1), 14),
                ("LEFTPADDING", (1, 0), (1, -1), 14),
            ],
        )
    ]


def _feature_ledger_blocks(coverage: Dict[str, Any]) -> List[Flowable]:
    """The feature ledger, its one-line reconciliation and any failure note."""
    if not coverage:
        return []
    pairs = [
        ("Features in the ledger", fmt_int(coverage.get("all_feature_count"))),
        ("Represented", fmt_int(coverage.get("represented_feature_count"))),
        ("Processed by tools", fmt_int(coverage.get("processed_feature_count"))),
        ("Passed through raw", fmt_int(coverage.get("unprocessed_raw_feature_count"))),
        ("Missing", fmt_int(coverage.get("missing_feature_count"))),
    ]
    blocks: List[Flowable] = [spec_table(pairs, columns=2, space_after=GAP_TIGHT)]

    missing_count = _as_int(coverage.get("missing_feature_count"))
    if missing_count is None:
        blocks.append(missing_note("The run did not record a missing-feature count."))
    elif missing_count > 0:
        blocks.append(
            P(f"{missing_count:,} features are missing from the prediction.", "Small")
        )
        names = [_s(name) for name in _as_list(coverage.get("missing_features")) if _s(name)]
        if names:
            blocks.append(Spacer(1, GAP_TIGHT))
            blocks.extend(bullet_list(names, limit=MISSING_FEATURE_CAP))
        else:
            blocks.append(missing_note("The identifiers of the missing features were not recorded."))

    if coverage.get("invariant_ok") is False:
        blocks.append(Spacer(1, GAP_TIGHT))
        blocks.append(
            Callout(
                "The coverage invariant failed for this run: the feature ledger does not "
                "reconcile with what reached the predictor. Treat the coverage figures "
                "above as unverified.",
                ALERT,
                title="Coverage invariant not satisfied",
            )
        )
    return blocks


def _uncertainty_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body = _Body()
    body.block(
        "Uncertainty factors",
        bullet_list(ctx.uncertainty_factors),
        absent_as="uncertainty factors",
    )
    # Per-domain coverage is charted in the next section, so the ledger here
    # stays a reconciliation of the feature count and nothing more.
    body.block(
        "Feature ledger",
        _feature_ledger_blocks(ctx.coverage_summary),
        absent_as="a coverage summary",
    )
    return _section(sections, SECTION_SPECS[2], body.close())


def _coverage_bar_rows(
    domain_coverage: Sequence[Tuple[str, Dict[str, Any]]]
) -> List[BarRow]:
    """Per-domain coverage as bar rows. The cover and section 4 share these."""
    rows: List[BarRow] = []
    for domain, stats in domain_coverage:
        percentage = _as_float(stats.get("coverage_percentage"))
        present = _as_int(stats.get("present_leaves"))
        total = _as_int(stats.get("total_leaves"))
        leaves = f"{present:,}/{total:,}" if present is not None and total is not None else "n/a"
        rows.append(
            BarRow(
                label=domain,
                value=(percentage or 0.0) / 100.0,
                text=f"{fmt_pct(percentage, '-'):>7}  {leaves}",
                color=ACCENT,
            )
        )
    return rows


def _coverage_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body = _Body()
    if ctx.domain_coverage:
        body.add(
            HBarChart(
                _coverage_bar_rows(ctx.domain_coverage),
                label_ratio=0.32,
                value_width=104.0,
                space_after=GAP_BLOCK,
            )
        )
        caption = (
            "Each bar is the share of that domain's ontology leaves that carried a value "
            "for this participant, with the leaf count alongside."
        )
        present = sum(_as_int(stats.get("present_leaves")) or 0 for _, stats in ctx.domain_coverage)
        total = sum(_as_int(stats.get("total_leaves")) or 0 for _, stats in ctx.domain_coverage)
        if total:
            caption += (
                f" Across every domain, {present:,} of {total:,} leaves "
                f"({present / total * 100.0:.1f}%) were present; the rest were absent from "
                "the source data."
            )
        body.add(P(caption, "Small"))
    else:
        body.add(
            missing_note(
                "Coverage cannot be charted: no data overview was supplied with this run."
            )
        )
    return _section(sections, SECTION_SPECS[3], body.close())


def _deviation_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body = _Body()
    leaves = ctx.deviation_leaves
    if not leaves:
        body.add(
            missing_note(
                "No hierarchical deviation map was supplied, so the deviation profile "
                "could not be computed."
            )
        )
        return _section(sections, SECTION_SPECS[4], body.close())

    rows = [header_row(["Domain", "Leaves", "Mean |z|", "Peak |z|"], numeric=[1, 2, 3])]
    for domain, count, mean_abs, peak in summarize_deviation_domains(leaves):
        rows.append(
            [
                P(domain, "TableCell"),
                P(f"{count:,}", "TableCellNumeric"),
                P(f"{mean_abs:.2f}", "TableCellNumeric"),
                P(f"{peak:.2f}", "TableCellNumeric"),
            ]
        )
    widths = [
        CONTENT_WIDTH * 0.46,
        CONTENT_WIDTH * 0.18,
        CONTENT_WIDTH * 0.18,
        CONTENT_WIDTH * 0.18,
    ]
    body.block("Domain summary", [make_table(rows, widths)])

    ranked = sorted(leaves, key=lambda leaf: abs(leaf.score), reverse=True)[:DEVIATION_CHART_ROWS]
    chart_rows = [
        DivRow(label=leaf.path, value=leaf.score, text=fmt_z(leaf.score)) for leaf in ranked
    ]
    caption = P(
        "Signed z-scores on a symmetric axis: left of the rule is below the normative "
        "mean, right of it is above.",
        "Small",
    )
    body.block(
        f"Furthest from the normative mean ({len(ranked)} of {len(leaves):,} leaves)",
        [
            DivergingBarChart(
                chart_rows, label_ratio=0.46, value_width=38.0, axis=True, space_after=GAP_BLOCK
            ),
            caption,
        ],
    )
    return _section(sections, SECTION_SPECS[4], body.close())


def _verdict_word(passed: bool) -> Paragraph:
    """A checklist result as a word, not a coloured pill."""
    style = ParagraphStyle(
        name=f"check_{'pass' if passed else 'fail'}",
        parent=_style(_styles(), "SpecValue"),
        fontName=FONT_SANS if passed else FONT_SANS_BOLD,
        textColor=MUTED if passed else ALERT,
    )
    return Paragraph("Passed" if passed else "Failed", style)


def _checklist_grid(checklist: Dict[str, Any], keys: Sequence[str]) -> Flowable:
    """Checklist results two to a line, so eight checks cost four lines."""
    pairs = [(humanize(key), bool(checklist.get(key))) for key in keys]
    rows: List[List[Any]] = []
    for start in range(0, len(pairs), 2):
        line: List[Any] = []
        for index in range(2):
            if start + index < len(pairs):
                label, passed = pairs[start + index]
                line.extend([P(label, "SpecKey"), _verdict_word(passed)])
            else:
                line.extend(["", ""])
        rows.append(line)
    pair_width = CONTENT_WIDTH / 2.0
    result_width = 52.0
    widths = [pair_width - result_width, result_width, pair_width - result_width, result_width]
    return make_table(
        rows,
        widths,
        header=False,
        extra_styles=[("RIGHTPADDING", (0, 0), (-1, -1), 12)],
    )


def _critic_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body = _Body()
    evaluation = ctx.evaluation
    verdict = ctx.verdict

    header_pairs: List[Tuple[str, str]] = [
        ("Verdict", verdict.title() if verdict else NOT_AVAILABLE),
        ("Confidence in verdict", fmt_prob(evaluation.get("confidence_in_verdict"))),
        ("Selected iteration", ctx.iteration_text),
    ]
    passed = _as_int(evaluation.get("checklist_passed"))
    total = _as_int(evaluation.get("checklist_total"))
    if passed is not None and total is not None:
        header_pairs.append(("Checklist", f"{passed} of {total} checks passed"))
    body.add(spec_table(header_pairs, columns=2))

    reason = _s(ctx.performance.get("selection_reason"))
    if reason:
        body.add(
            Callout(
                reason,
                ACCENT if verdict == "SATISFACTORY" else ALERT,
                title="Attempt selection",
            )
        )

    composite = _as_float(evaluation.get("composite_score"))
    breakdown = _as_dict(evaluation.get("score_breakdown"))
    score_blocks: List[Flowable] = []
    if composite is not None:
        score_blocks.append(ScoreMeter(composite, segments=10, label="composite", space_after=GAP_TIGHT))
    if breakdown:
        score_blocks.append(
            spec_table(
                [(humanize(key), fmt_number(value)) for key, value in breakdown.items()],
                columns=2,
            )
        )
    body.block("Composite score", score_blocks, absent_as="a composite critic score")

    checklist = _as_dict(evaluation.get("checklist"))
    checklist_blocks: List[Flowable] = []
    if checklist:
        active = [_s(key) for key in _as_list(checklist.get("active_checks")) if _s(key)]
        keys = active or [key for key, value in checklist.items() if isinstance(value, bool)]
        if keys:
            checklist_blocks.append(_checklist_grid(checklist, keys[:CHECKLIST_CAP]))
            # The header above states the full count, so a silent cut here would
            # leave the reader counting rows that do not add up.
            if len(keys) > CHECKLIST_CAP:
                checklist_blocks.append(
                    P(f"and {len(keys) - CHECKLIST_CAP:,} further checks not shown", "Micro")
                )
    elif passed is not None and total is not None:
        checklist_blocks.append(
            P(
                f"{passed} of {total} checks passed. The per-check detail was not recorded "
                "in the run artifacts.",
                "Small",
            )
        )
    body.block("Quality checklist", checklist_blocks, absent_as="a quality checklist")

    for title, key in (("Strengths", "strengths"), ("Weaknesses", "weaknesses")):
        entries = [_s(item) for item in _as_list(evaluation.get(key)) if _s(item)]
        body.block(title, bullet_list(entries), absent_as=title.lower())

    suggestions = [_as_dict(item) for item in _as_list(evaluation.get("improvement_suggestions"))]
    suggestions = [item for item in suggestions if item]
    suggestion_blocks: List[Flowable] = []
    if suggestions:
        rows = [header_row(["Priority", "Issue", "Suggestion"])]
        for item in suggestions:
            priority = _s(item.get("priority"), "UNSET").upper()
            rows.append(
                [
                    Chip(priority, ALERT if priority == "HIGH" else MUTED),
                    P(_s(item.get("issue"), NOT_AVAILABLE), "TableCell"),
                    P(_s(item.get("suggestion"), NOT_AVAILABLE), "TableCell"),
                ]
            )
        widths = [CONTENT_WIDTH * 0.16, CONTENT_WIDTH * 0.40, CONTENT_WIDTH * 0.44]
        suggestion_blocks.append(make_table(rows, widths))
    body.block("Improvement suggestions", suggestion_blocks, absent_as="improvement suggestions")

    return _section(sections, SECTION_SPECS[5], body.close())


#: Blocks of the narrative that restate, at length, what earlier sections
#: already give from the run's own record. The Communicator is asked for a
#: self-contained document, so it repeats the header and re-tabulates the
#: evidence; printing both costs several pages and tells the reader nothing new.
_NARRATIVE_DUPLICATES: Tuple[Tuple[str, str], ...] = (
    ("target header", "section 1"),
    ("technical summary", "sections 1 and 2"),
)


def _trim_narrative(markdown_text: str) -> Tuple[str, List[str]]:
    """
    Drop the narrative blocks that duplicate earlier sections, and name them.

    Returns the markdown still worth printing and a plain-language list of what
    was left out, which the section prints so nothing goes missing silently.
    """
    lines = markdown_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept: List[str] = []
    dropped: List[str] = []
    skipping = False
    fenced = False
    for line in lines:
        if _RE_FENCE.match(line):
            fenced = not fenced
        heading = None if fenced else _RE_HEADING.match(line)
        if heading is not None and len(heading.group(1)) <= 2:
            title = heading.group(2).strip()
            lowered = title.lower()
            skipping = False
            for needle, where in _NARRATIVE_DUPLICATES:
                if needle in lowered:
                    skipping = True
                    dropped.append(f"{title} (see {where})")
                    break
        if not skipping:
            kept.append(line)
    return "\n".join(kept), dropped


def _deep_phenotype_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body = _Body()
    if ctx.deep_phenotype_markdown.strip():
        trimmed, dropped = _trim_narrative(ctx.deep_phenotype_markdown)
        rendered = markdown_to_flowables(trimmed, _styles())
        # The narrative already opens on its own title, which would sit
        # directly under the section header saying the same thing twice.
        if rendered and _is_heading(rendered[0]):
            rendered = rendered[1:]
        if rendered:
            body.extend(rendered)
        elif dropped:
            # A narrative built only from the blocks the trimmer targets leaves
            # nothing to print. That is the trimmer's doing, not a parse failure.
            body.add(
                missing_note(
                    "Every part of this narrative repeats a section printed above, "
                    "so none of it is reprinted here."
                )
            )
        else:
            body.add(missing_note("The deep phenotype report was empty after parsing."))
        if dropped:
            body.add(Spacer(1, GAP_BLOCK))
            body.add(
                missing_note(
                    f"Not reprinted from the narrative: {sentence_list(dropped)}. "
                    "Those parts restate the prediction and the evidence, which this "
                    "report already gives from the run's own record. The narrative "
                    "itself is kept in full in deep_phenotype.md."
                )
            )
    else:
        body.add(missing_note("This run did not generate a deep phenotype narrative."))
        status = _as_dict(ctx.performance.get("deep_phenotype"))
        if status:
            body.add(
                P(
                    f"Generation flag: {'yes' if status.get('generated') else 'no'}. "
                    f"Trigger source: {_s(status.get('trigger_source'), 'not requested')}.",
                    "Micro",
                )
            )
    return _section(sections, SECTION_SPECS[6], body.close())


def _token_ledger_table(calls: Sequence[Any]) -> List[Flowable]:
    """Prompt and completion tokens per component, with a summed final row."""
    grouped = _group_token_calls(calls)
    if not grouped:
        return []
    rows = [header_row(["Component", "Calls", "Prompt", "Completion", "Total"], numeric=[1, 2, 3, 4])]
    for component, call_count, prompt, completion, total in grouped:
        rows.append(
            [
                P(humanize(component), "TableCell"),
                P(f"{call_count:,}", "TableCellNumeric"),
                P(f"{prompt:,}", "TableCellNumeric"),
                P(f"{completion:,}", "TableCellNumeric"),
                P(f"{total:,}", "TableCellNumeric"),
            ]
        )
    totals = [sum(row[index] for row in grouped) for index in (1, 2, 3, 4)]
    rows.append(
        [
            P("All components", "TableCellBold"),
            *[P(f"{value:,}", "TableCellNumericBold") for value in totals],
        ]
    )
    widths = [
        CONTENT_WIDTH * 0.32,
        CONTENT_WIDTH * 0.13,
        CONTENT_WIDTH * 0.18,
        CONTENT_WIDTH * 0.18,
        CONTENT_WIDTH * 0.19,
    ]
    return [make_table(rows, widths, extra_styles=totals_row_style(len(rows) - 1))]


def _cost_table(cost: Dict[str, Any]) -> List[Flowable]:
    """Priced token lines per model. Empty when no ledger was supplied."""
    lines = [_as_dict(line) for line in _as_list(cost.get("lines"))]
    lines = [line for line in lines if line]
    if not lines:
        return []
    rows = [header_row(["Model", "Prompt", "Completion", "Total tokens", "USD"], numeric=[1, 2, 3, 4])]
    for line in lines:
        rows.append(
            [
                P(_s(line.get("model"), NOT_AVAILABLE), "TableCellMono"),
                P(fmt_int(line.get("prompt_tokens"), "-"), "TableCellNumeric"),
                P(fmt_int(line.get("completion_tokens"), "-"), "TableCellNumeric"),
                P(fmt_int(line.get("total_tokens"), "-"), "TableCellNumeric"),
                P(fmt_usd(line.get("usd")), "TableCellNumeric"),
            ]
        )
    rows.append(
        [
            P("Total", "TableCellBold"),
            P("", "TableCellNumeric"),
            P("", "TableCellNumeric"),
            P(fmt_int(cost.get("total_tokens"), "-"), "TableCellNumericBold"),
            P(fmt_usd(cost.get("usd")), "TableCellNumericBold"),
        ]
    )
    widths = [
        CONTENT_WIDTH * 0.34,
        CONTENT_WIDTH * 0.15,
        CONTENT_WIDTH * 0.16,
        CONTENT_WIDTH * 0.18,
        CONTENT_WIDTH * 0.17,
    ]
    blocks: List[Flowable] = [
        make_table(rows, widths, extra_styles=totals_row_style(len(rows) - 1))
    ]
    if _as_float(cost.get("usd")) is None:
        blocks.append(
            P(
                "At least one model in the ledger has no published price, so the total "
                "cost cannot be closed out.",
                "Small",
            )
        )
    return blocks


def _execution_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body = _Body()
    plan = _as_dict(ctx.performance.get("plan_summary"))
    plan_pairs: List[Tuple[str, str]] = [
        ("Duration", fmt_duration(ctx.performance.get("total_duration_seconds"))),
        ("Iterations", ctx.iteration_text),
    ]
    if plan:
        domains = [_s(item) for item in _as_list(plan.get("priority_domains")) if _s(item)]
        plan_pairs.extend(
            [
                ("Plan steps", fmt_int(plan.get("total_steps"))),
                ("Plan id", _s(plan.get("plan_id"), NOT_AVAILABLE)),
            ]
        )
        if domains:
            plan_pairs.append(("Priority domains", ", ".join(domains)))
    body.add(spec_table(plan_pairs, columns=2))

    ledger = _token_ledger_table(_as_list(ctx.token_usage.get("calls")))
    recorded_total = _as_int(ctx.token_usage.get("total_tokens"))
    if not ledger and recorded_total is not None:
        # A total without the per-call breakdown is still worth stating, and
        # saying why the table is not there keeps the absence explicit.
        ledger = [
            P(
                f"{recorded_total:,} tokens were recorded in total, but the per-call "
                "ledger was not saved with this run.",
                "Small",
            )
        ]
    body.block("Token ledger by component", ledger, absent_as="a token ledger")

    body.block("Cost", _cost_table(ctx.cost), absent_as="a priced cost ledger")
    return _section(sections, SECTION_SPECS[7], body.close())


def _provenance_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    """
    Everything needed to say where this report came from and run it again.

    Deliberately a handful of rows. The full dataflow audit belongs in the JSON
    artifacts the run already writes, not transcribed across three pages here.

    Rows split on whether they can be attributed to the run. What the artifacts
    and the run record hold is stated plainly; settings that only a live record
    could vouch for are printed under a heading that says they are the service's
    current ones, so nothing here reads as a fact about the run that is not.
    """
    body = _Body()
    meta = ctx.provenance

    pairs: List[Tuple[str, str]] = [
        ("Engine version", _ENGINE_VERSION or "not recorded"),
        ("Participant", ctx.participant_id or "not recorded"),
    ]
    run_id = _s(meta.get("run_id"))
    if run_id:
        pairs.append(("Run id", run_id))
    pairs.append(("Executed", fmt_timestamp(ctx.performance.get("execution_timestamp"))))
    finished = _s(meta.get("finished_at"))
    if finished:
        pairs.append(("Finished", fmt_timestamp(finished)))
    pairs.append(("Report generated", fmt_timestamp(datetime.now().isoformat())))
    pairs.append(("Iterations", ctx.iteration_text))
    tokens = ctx.total_tokens
    if tokens is not None:
        pairs.append(("Tokens spent", f"{tokens:,}"))

    settings_pairs: List[Tuple[str, str]] = []
    reasoning = _s(meta.get("reasoning_effort"))
    if reasoning:
        settings_pairs.append(("Reasoning effort", humanize(reasoning)))
    budget = _as_int(meta.get("token_budget"))
    if budget:
        settings_pairs.append(("Token budget", f"{budget:,} tokens"))
    backend = _s(meta.get("backend"))
    if backend:
        settings_pairs.append(("Backend", backend))

    configured = _configured_model_pairs(ctx)
    billed = _billed_model_pairs(ctx)
    if ctx.settings_from_run:
        body.add(spec_table(pairs + settings_pairs, columns=2))
        body.block("Models", _mono_spec(configured + billed), absent_as="the models used")
    else:
        body.add(spec_table(pairs, columns=2))
        body.block("Models", _mono_spec(billed), absent_as="the models used")
        if settings_pairs or configured:
            body.block(
                "Current service settings",
                [
                    P(
                        "The settings this run was launched with were not kept with its "
                        "artifacts. What follows is how the service is configured now, "
                        "which is not necessarily what governed the run above.",
                        "Small",
                    ),
                    spec_table(settings_pairs, columns=2) if settings_pairs else None,
                    *_mono_spec(configured),
                ],
            )

    instructions = [
        (humanize(key), _s(value))
        for key, value in _as_dict(ctx.performance.get("agent_instructions")).items()
        if _s(value)
    ]
    if instructions:
        rows = [header_row(["Agent", "Instruction"])]
        rows.extend([P(key, "TableCell"), P(value, "TableCell")] for key, value in instructions)
        body.block(
            "Custom agent instructions",
            [make_table(rows, [CONTENT_WIDTH * 0.22, CONTENT_WIDTH * 0.78])],
        )

    explainability = _as_dict(ctx.performance.get("explainability"))
    if explainability.get("enabled"):
        methods = [_s(item) for item in _as_list(explainability.get("methods_requested")) if _s(item)]
        body.block(
            "Explainability",
            [
                spec_table(
                    [
                        ("Status", _s(explainability.get("status"), NOT_AVAILABLE)),
                        ("Methods requested", ", ".join(methods) if methods else "none"),
                    ],
                    columns=2,
                )
            ],
        )

    lines = ctx.reproduce_command_lines()
    if lines:
        lead = (
            "From the repository root, against the same participant directory and "
            "the same task, with the run's own model and budget settings:"
            if ctx.settings_from_run
            else "From the repository root, against the same participant directory and "
            "the same task. The model and the reasoning effort this run was launched "
            "with are not recorded in its artifacts, so the command names neither and "
            "the service defaults apply:"
        )
        # The command is short and unsplittable, so letting the frame break
        # between its heading and its panel strands it alone on a final page.
        body.block(
            "",
            [
                KeepHeadingWith(
                    [P("How to reproduce", "H2"), P(lead, "Small"), _code_block(lines, _styles())]
                )
            ],
        )

    return _section(sections, SECTION_SPECS[8], body.close())


def _configured_model_pairs(ctx: "_RunContext") -> List[Tuple[str, str]]:
    """Which model each role was given, according to a configuration."""
    meta = ctx.provenance
    default_model = _s(meta.get("default_model"))
    roles = {
        _s(role): _s(name) or default_model
        for role, name in _as_dict(meta.get("role_models")).items()
        if _s(role)
    }
    distinct = {name for name in roles.values() if name}
    # One line when every agent shared a model, a row per role when they did not.
    if roles and len(distinct) == 1 and default_model:
        return [("Configured for every role", default_model)]
    if roles:
        return [
            (humanize(role), name or "not configured") for role, name in sorted(roles.items())
        ]
    return [("Configured model", default_model)] if default_model else []


def _billed_model_pairs(ctx: "_RunContext") -> List[Tuple[str, str]]:
    """Which models the run's own token ledger charged for."""
    billed = ctx.models_used()
    return [("Billed by the token ledger", ", ".join(billed))] if billed else []


def _mono_spec(pairs: Sequence[Tuple[str, str]]) -> List[Flowable]:
    """Model names in a one-column monospaced grid, or nothing to print."""
    return [spec_table(list(pairs), columns=1, mono=True)] if pairs else []


# ============================================================================
# Run context
# ============================================================================


class _RunContext:
    """Normalized view of the run artifacts, resilient to anything missing."""

    def __init__(
        self,
        performance_report: Any,
        patient_report: Any,
        deep_phenotype_markdown: Any,
        data_overview: Any,
        deviation_map: Any,
        cost: Any,
        branding: Any,
        provenance: Any = None,
    ) -> None:
        self.performance = _as_dict(performance_report)
        self.patient_report = _as_dict(patient_report)
        self.deep_phenotype_markdown = (
            deep_phenotype_markdown if isinstance(deep_phenotype_markdown, str) else ""
        )
        self.data_overview = _as_dict(data_overview)
        self.deviation_map = _as_dict(deviation_map)
        self.cost = _as_dict(cost)
        self.branding = {k: _s(v) for k, v in _as_dict(branding).items()}
        self.provenance = _as_dict(provenance)
        # Whether the settings in ``provenance`` are this run's own or the
        # service's current ones. Absent means unattributable, which is the
        # safe reading: nothing is claimed for the run that it did not record.
        self.settings_from_run = bool(self.provenance.get("settings_from_run"))

        self.participant_id = _s(
            self.performance.get("participant_id")
        ) or _s(self.patient_report.get("participant_id"), "")

        self.task_spec = _as_dict(self.performance.get("prediction_task_spec")) or _as_dict(
            _as_dict(self.patient_report.get("prediction")).get("prediction_task_spec")
        )
        self.task_root = _as_dict(self.task_spec.get("root"))

        self.prediction_result = _as_dict(self.performance.get("prediction_result"))
        root_prediction = _as_dict(self.prediction_result.get("root_prediction")) or _as_dict(
            _as_dict(self.patient_report.get("prediction")).get("root_prediction")
        )
        self.node_rows = flatten_nodes(root_prediction)

        self.coverage_summary = _as_dict(self.performance.get("coverage_summary")) or _as_dict(
            _as_dict(self.patient_report.get("execution")).get("coverage_summary")
        )
        self.token_usage = _as_dict(self.performance.get("token_usage"))
        self.deviation_leaves = walk_deviation_map(self.deviation_map)

        coverage = _as_dict(self.data_overview.get("domain_coverage"))
        self.domain_coverage: List[Tuple[str, Dict[str, Any]]] = sorted(
            ((_s(name, "?"), _as_dict(stats)) for name, stats in coverage.items()),
            key=lambda row: _as_float(row[1].get("coverage_percentage")) or 0.0,
            reverse=True,
        )

        self.evaluation = self._merge_evaluation()
        self.verdict = _s(
            self.performance.get("critic_verdict") or self.evaluation.get("verdict"), ""
        ).upper()

        self.key_findings = _collect_key_findings(self.node_rows, self.patient_report)
        self.reasoning_chain = _collect_from_nodes(self.node_rows, "reasoning_chain") or [
            _s(item) for item in _as_list(self.patient_report.get("reasoning")) if _s(item)
        ]
        self.evidence_for = _collect_from_nodes(self.node_rows, "supporting_evidence_for")
        self.evidence_against = _collect_from_nodes(self.node_rows, "supporting_evidence_against")
        self.uncertainty_factors = _collect_from_nodes(self.node_rows, "uncertainty_factors")

    def _merge_evaluation(self) -> Dict[str, Any]:
        """The critic block can live in either artifact; take the union."""
        merged: Dict[str, Any] = {}
        for candidate in (
            _as_dict(self.patient_report.get("evaluation")),
            _as_dict(self.performance.get("evaluation")),
            _as_dict(self.performance.get("critic_evaluation")),
        ):
            for key, value in candidate.items():
                if value is not None and (key not in merged or merged[key] in (None, "", [], {})):
                    merged[key] = value
        return merged

    # -- derived display values -------------------------------------------

    @property
    def primary_output(self) -> str:
        report_output = _s(_as_dict(self.patient_report.get("prediction")).get("primary_output"))
        if report_output:
            return report_output
        if self.node_rows:
            return self.node_rows[0].output
        return _s(self.prediction_result.get("classification"), NOT_AVAILABLE)

    @property
    def confidence_level(self) -> str:
        return _s(
            self.prediction_result.get("confidence")
            or (self.node_rows[0].confidence_level if self.node_rows else ""),
            NOT_AVAILABLE,
        )

    @property
    def root_confidence(self) -> Optional[float]:
        value = _as_float(self.prediction_result.get("root_confidence"))
        if value is None and self.node_rows:
            value = self.node_rows[0].confidence_score
        return value

    @property
    def iteration_text(self) -> str:
        selected = _as_int(self.performance.get("selected_iteration"))
        total = _as_int(self.performance.get("iterations"))
        if selected is None and total is None:
            return NOT_AVAILABLE
        return f"{selected if selected is not None else '?'} of {total if total is not None else '?'}"

    @property
    def total_tokens(self) -> Optional[int]:
        value = _as_int(self.token_usage.get("total_tokens"))
        if value is None:
            value = _as_int(_as_dict(self.patient_report.get("execution")).get("tokens_used"))
        if value is None:
            value = _as_int(self.cost.get("total_tokens"))
        return value

    @property
    def task_line(self) -> str:
        mode = humanize(self.task_root.get("mode")) if self.task_root else ""
        mode = "" if mode == NOT_AVAILABLE else mode
        name = _s(self.task_root.get("display_name")) or _s(
            self.performance.get("target_condition")
        )
        labels = [_s(item) for item in _as_list(self.task_root.get("class_labels")) if _s(item)]
        outputs = [
            _s(item) for item in _as_list(self.task_root.get("regression_outputs")) if _s(item)
        ]
        detail = ""
        if labels:
            detail = f"labels: {', '.join(labels)}"
        elif outputs:
            detail = f"outputs: {', '.join(outputs)}"
        node_count = len(self.node_rows)
        if node_count == 0:
            tree = "no prediction node"
        elif node_count == 1:
            tree = "single node"
        else:
            tree = f"{node_count} nodes"
        if mode and name:
            headline = f"{mode} of {name}"
        else:
            headline = mode or (f"Task: {name}" if name else "Task not recorded")
        parts = [headline, tree]
        if detail:
            parts.append(detail)
        return META_SEPARATOR.join(parts)

    def models_used(self) -> List[str]:
        seen: List[str] = []
        for line in _as_list(self.cost.get("lines")):
            model = _s(_as_dict(line).get("model"))
            if model and model not in seen:
                seen.append(model)
        return seen

    def prediction_family(self) -> str:
        """The ``--prediction_type`` value that produced this task."""
        if len(self.node_rows) > 1 or _as_list(self.task_root.get("children")):
            return "hierarchical"
        return {
            "binary_classification": "binary",
            "multiclass_classification": "multiclass",
            "univariate_regression": "regression_univariate",
            "multivariate_regression": "regression_multivariate",
        }.get(_s(self.task_root.get("mode")), "")

    def reproduce_command_lines(self) -> List[str]:
        """
        The command that would run this task again, wrapped so it can be copied.

        Task, labels and iteration count come from the artifacts. The model and
        the reasoning effort come from a configuration, which is only this run's
        own while the run is still live, so they are written out when
        ``settings_from_run`` says they can be attributed and left off otherwise:
        what the line names, the run really used.
        """
        directory = _s(self.provenance.get("participant_dir"))
        if not directory:
            return []
        parts: List[str] = []
        family = self.prediction_family()
        if family:
            parts.append(f"--prediction_type {family}")
        if family == "hierarchical":
            parts.append("--task_spec_file <the task specification used for this run>")
        else:
            target = _s(self.performance.get("target_condition"))
            control = _s(self.performance.get("control_condition"))
            if target:
                parts.append(f"--target_label {target}")
            if control:
                parts.append(f"--control_label {control}")
        if self.settings_from_run:
            model = _s(self.provenance.get("default_model"))
            if model:
                parts.append(f"--model {model}")
            reasoning = _s(self.provenance.get("reasoning_effort"))
            if reasoning:
                parts.append(f"--reasoning_effort {reasoning}")
        iterations = _as_int(self.performance.get("iterations"))
        if iterations:
            parts.append(f"--iterations {iterations}")
        return shell_command_lines("python main.py", self.participant_path, parts)

    @property
    def participant_path(self) -> str:
        """The participant directory as the reader would type it from the repo root."""
        directory = _s(self.provenance.get("participant_dir"))
        try:
            return str(Path(directory).resolve().relative_to(_REPO_ROOT))
        except (OSError, ValueError):
            return directory

    def provenance_line(self) -> str:
        models = self.models_used()
        parts = [
            f"Engine {_ENGINE_VERSION}" if _ENGINE_VERSION else "Engine version not recorded",
            f"Models: {', '.join(models)}" if models else "Models: not recorded",
            f"Duration: {fmt_duration(self.performance.get('total_duration_seconds'))}",
            f"Executed: {fmt_timestamp(self.performance.get('execution_timestamp'))}",
        ]
        return META_SEPARATOR.join(parts)

    def kpi_items(self) -> List[KpiItem]:
        verdict = self.verdict or NOT_AVAILABLE
        cost_value = _as_float(self.cost.get("usd"))
        tokens = self.total_tokens
        return [
            KpiItem(
                label="Primary output",
                value=self.primary_output,
                note=humanize(self.task_root.get("mode")) if self.task_root else "",
            ),
            KpiItem(
                label="Confidence",
                value=fmt_prob(self.root_confidence),
                note="" if self.confidence_level == NOT_AVAILABLE else self.confidence_level.lower(),
            ),
            # The only tile that can change colour, and only to say that the
            # automated review was not satisfied.
            KpiItem(
                label="Critic verdict",
                value=verdict.title() if verdict != NOT_AVAILABLE else NOT_AVAILABLE,
                note="automated review",
                accent=None if verdict in ("SATISFACTORY", NOT_AVAILABLE) else ALERT,
            ),
            KpiItem(
                label="Iterations",
                value=self.iteration_text,
                note="selected of total",
            ),
            KpiItem(
                label="Total tokens",
                value=fmt_int(tokens, "Not recorded"),
                note="prompt and completion",
            ),
            # Not an estimate: the ledger prices the tokens the run actually
            # spent, so the tile says so, and says so plainly when it cannot.
            KpiItem(
                label="Run cost",
                value=fmt_usd(cost_value, "Not priced"),
                note="from the token ledger" if cost_value is not None else "pricing unavailable",
            ),
        ]

    def furniture(self) -> Furniture:
        title = self.branding.get("title") or "COMPASS Run Report"
        subtitle = self.branding.get("subtitle") or self.task_line
        footer = self.branding.get("footer") or DISCLAIMER
        return Furniture(
            title=title,
            subtitle=subtitle,
            participant_id=self.participant_id,
            task_line=self.task_line,
            timestamp=f"Generated {fmt_timestamp(datetime.now().isoformat())}",
            footer=footer,
        )


# ============================================================================
# Public API
# ============================================================================


def render_run_pdf(
    *,
    output_path: Path,
    performance_report: Dict[str, Any],
    patient_report: Optional[Dict[str, Any]] = None,
    deep_phenotype_markdown: str = "",
    data_overview: Optional[Dict[str, Any]] = None,
    deviation_map: Optional[Dict[str, Any]] = None,
    cost: Optional[Dict[str, Any]] = None,
    branding: Optional[Dict[str, str]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Render one COMPASS run into a standardized, print-quality PDF.

    Every argument except ``output_path`` and ``performance_report`` is optional,
    and each section degrades to an explicit "not recorded" statement rather than
    disappearing, so a reader can always tell absence from omission.

    Args:
        output_path: Where to write the PDF. Parent directories are created.
        performance_report: Contents of ``performance_report_<pid>.json``.
        patient_report: Contents of ``report_<pid>.json``, if available.
        deep_phenotype_markdown: Contents of ``deep_phenotype.md``, may be empty.
        data_overview: Contents of ``data_overview.json``, if available.
        deviation_map: Contents of ``hierarchical_deviation_map.json``.
        cost: Cost ledger shaped like ``{"usd", "lines", "total_tokens"}``.
        branding: Optional ``title`` / ``subtitle`` / ``footer`` overrides.
        provenance: What the artifacts cannot know about themselves: the run id,
            the participant directory, the configured models per role, the
            reasoning effort, the token budget and the run timestamps. Every
            key is optional and an absent one simply drops its row. The
            ``settings_from_run`` flag says whether the settings among those
            belong to this run or are merely the caller's current ones; without
            it they are printed as the latter, never attributed to the run.

    Returns:
        ``output_path``, after the file has been written.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    ctx = _RunContext(
        performance_report,
        patient_report,
        deep_phenotype_markdown,
        data_overview,
        deviation_map,
        cost,
        branding,
        provenance,
    )
    sections = _Sections()

    # The body template takes over from page two; the cover owns page one only.
    # Switching templates before the cover content, rather than after it, means
    # that even an unforeseen overflow lands on a body page instead of a second
    # page wearing the cover's accent band.
    story: List[Flowable] = [NextPageTemplate("Body")]
    story.extend(_cover_flowables(ctx))
    story.append(PageBreak())
    story.extend(_summary_flowables(ctx))
    story.append(PageBreak())

    builders: Sequence[Callable[["_RunContext", _Sections], List[Flowable]]] = (
        _prediction_section,
        _evidence_section,
        _uncertainty_section,
        _coverage_section,
        _deviation_section,
        _critic_section,
        _deep_phenotype_section,
        _execution_section,
        _provenance_section,
    )
    for builder, spec in zip(builders, SECTION_SPECS):
        try:
            story.extend(builder(ctx, sections))
        except Exception as error:  # A broken section must not lose the report.
            notice = Callout(
                f"This section could not be rendered from the run artifacts: {error}",
                ALERT,
                title="Section unavailable",
            )
            try:
                story.extend(_section(sections, spec, [notice]))
            except Exception:
                story.extend([Spacer(1, GAP_SECTION), notice])

    doc = RunDocTemplate(str(target), ctx.furniture())
    doc.build(story, canvasmaker=NumberedCanvas)
    return target
