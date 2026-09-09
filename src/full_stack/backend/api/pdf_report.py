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

INK = colors.HexColor("#0F172A")
MUTED = colors.HexColor("#64748B")
HAIRLINE = colors.HexColor("#E2E8F0")
PAGE_TINT = colors.HexColor("#F8FAFC")
ACCENT = colors.HexColor("#4F46E5")
POSITIVE = colors.HexColor("#059669")
CAUTION = colors.HexColor("#D97706")
CRITICAL = colors.HexColor("#DC2626")
INFO = colors.HexColor("#0EA5E9")
PAPER = colors.HexColor("#FFFFFF")

#: Type scale: (font size, leading). Helvetica and Courier ship with every
#: viewer, so the document renders identically without embedded fonts.
FONT_SANS = "Helvetica"
FONT_SANS_BOLD = "Helvetica-Bold"
FONT_SANS_OBLIQUE = "Helvetica-Oblique"
FONT_MONO = "Courier"
FONT_MONO_BOLD = "Courier-Bold"

SIZE_DISPLAY, LEAD_DISPLAY = 26.0, 32.0
SIZE_H1, LEAD_H1 = 16.0, 22.0
SIZE_H2, LEAD_H2 = 12.5, 17.0
SIZE_H3, LEAD_H3 = 10.5, 14.0
SIZE_BODY, LEAD_BODY = 9.5, 14.0
SIZE_SMALL, LEAD_SMALL = 8.5, 12.0
SIZE_MICRO, LEAD_MICRO = 7.5, 10.0

PAGE_SIZE = A4
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_SIDE = 42.0
MARGIN_TOP = 54.0
MARGIN_BOTTOM = 46.0
CONTENT_WIDTH = PAGE_WIDTH - (2 * MARGIN_SIDE)

#: Vertical rhythm. Every gap in the document is a multiple of these.
RHYTHM = 6.0
GAP_TIGHT = RHYTHM
GAP_BLOCK = RHYTHM * 2
GAP_SECTION = RHYTHM * 3

COVER_BAND_HEIGHT = 118.0
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

DISCLAIMER = (
    "COMPASS is a research prototype and is not a certified medical device. "
    "Outputs require review by qualified domain experts."
)
NOT_AVAILABLE = "Not available"
#: Section titles and standfirsts, in document order. The cover contents
#: list and the section headers both read from here so they cannot drift.
SECTION_SPECS: Tuple[Tuple[str, str], ...] = (
    ("Prediction", "Task specification and primary output"),
    ("Evidence", "What the prediction rests on"),
    ("Uncertainty and missing information", "What the run could not see"),
    ("Evidence coverage", "Ontology leaves present per domain"),
    ("Deviation profile", "Normative deviation across the ontology"),
    ("Critic evaluation", "Automated quality review"),
    ("Deep phenotype report", "Narrative synthesis from the Communicator"),
    ("Execution and cost", "How the run was carried out"),
    ("Appendix: run configuration", "Provenance for reproduction"),
)

#: Cap on how many missing feature identifiers are printed before eliding.
MISSING_FEATURE_CAP = 40
#: Cap on how many leaf paths the deviation chart shows.
DEVIATION_CHART_ROWS = 18
#: Depth guard for the generic deviation-map walk.
MAX_TREE_DEPTH = 12

try:  # The engine version is nice to have, not worth an import failure.
    from ..config.settings import COMPASS_VERSION as _ENGINE_VERSION
except Exception:  # pragma: no cover - defensive, settings pulls heavy optionals
    _ENGINE_VERSION = ""


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
    add(
        "H2",
        fontName=FONT_SANS_BOLD,
        fontSize=SIZE_H2,
        leading=LEAD_H2,
        spaceBefore=GAP_BLOCK,
        spaceAfter=GAP_TIGHT * 0.5,
        keepWithNext=1,
    )
    add(
        "H3",
        fontName=FONT_SANS_BOLD,
        fontSize=SIZE_H3,
        leading=LEAD_H3,
        spaceBefore=GAP_TIGHT,
        spaceAfter=2,
        keepWithNext=1,
    )
    add(
        "H4",
        fontName=FONT_SANS_BOLD,
        fontSize=SIZE_SMALL,
        leading=LEAD_SMALL,
        textColor=MUTED,
        spaceBefore=GAP_TIGHT,
        spaceAfter=2,
        keepWithNext=1,
    )
    add("Body", spaceAfter=GAP_TIGHT)
    add("BodyTight", spaceAfter=2)
    add("BodyMuted", textColor=MUTED, spaceAfter=GAP_TIGHT)
    add("Lead", fontSize=SIZE_H3, leading=LEAD_H3 + 2, spaceAfter=GAP_TIGHT)
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
    add("BigValue", fontName=FONT_SANS_BOLD, fontSize=22, leading=26)
    add("TableHeader", fontName=FONT_SANS_BOLD, fontSize=SIZE_MICRO, leading=LEAD_MICRO, textColor=MUTED)
    add("TableCell", fontSize=SIZE_SMALL, leading=LEAD_SMALL)
    add("TableCellMuted", fontSize=SIZE_SMALL, leading=LEAD_SMALL, textColor=MUTED)
    add("TableCellMono", fontName=FONT_MONO, fontSize=SIZE_MICRO + 0.5, leading=LEAD_SMALL, wordWrap="CJK")
    add("TableCellRight", fontSize=SIZE_SMALL, leading=LEAD_SMALL, alignment=TA_RIGHT)
    add(
        "TableCellNumeric",
        fontName=FONT_MONO,
        fontSize=SIZE_MICRO + 0.5,
        leading=LEAD_SMALL,
        alignment=TA_RIGHT,
    )
    add("Bullet", spaceAfter=2, leftIndent=13, bulletIndent=2)
    add("BulletNested", spaceAfter=2, leftIndent=26, bulletIndent=15, textColor=colors.HexColor("#334155"))
    add("Numbered", spaceAfter=2, leftIndent=18, bulletIndent=2)
    add("CalloutBody", fontSize=SIZE_SMALL, leading=LEAD_SMALL + 0.5)
    add("Code", fontName=FONT_MONO, fontSize=SIZE_SMALL, leading=LEAD_SMALL + 1, wordWrap="CJK")
    add("MdH1", fontName=FONT_SANS_BOLD, fontSize=SIZE_H1, leading=LEAD_H1, spaceBefore=GAP_BLOCK, spaceAfter=GAP_TIGHT, keepWithNext=1)
    add("MdH2", fontName=FONT_SANS_BOLD, fontSize=SIZE_H2, leading=LEAD_H2, spaceBefore=GAP_BLOCK, spaceAfter=3, keepWithNext=1)
    add("MdH3", fontName=FONT_SANS_BOLD, fontSize=SIZE_H3, leading=LEAD_H3, spaceBefore=GAP_TIGHT, spaceAfter=2, keepWithNext=1)
    add(
        "MdH4",
        fontName=FONT_SANS_BOLD,
        fontSize=SIZE_BODY,
        leading=LEAD_BODY,
        textColor=MUTED,
        spaceBefore=GAP_TIGHT,
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
    """Numbered section heading: accent circle, title, optional subtitle, rule."""

    RADIUS = 9.0
    GAP_AFTER_CIRCLE = 10.0

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
        text_left = (self.RADIUS * 2) + self.GAP_AFTER_CIRCLE
        self._subtitle_height = 0.0
        self._subtitle_para = None
        if self.subtitle:
            self._subtitle_para = P(self.subtitle, "SmallMuted")
            _, self._subtitle_height = self._subtitle_para.wrap(
                max(availWidth - text_left, 40.0), availHeight
            )
        self.height = LEAD_H1 + self._subtitle_height + GAP_TIGHT + 1.0
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        text_left = (self.RADIUS * 2) + self.GAP_AFTER_CIRCLE
        rule_y = 1.0
        title_baseline = rule_y + GAP_TIGHT + self._subtitle_height + 4.5

        canvas.setFillColor(ACCENT)
        canvas.circle(self.RADIUS, title_baseline + (SIZE_H1 * 0.34), self.RADIUS, stroke=0, fill=1)
        if self.number:
            canvas.setFillColor(PAPER)
            canvas.setFont(FONT_SANS_BOLD, 9.5)
            canvas.drawCentredString(
                self.RADIUS, title_baseline + (SIZE_H1 * 0.34) - 3.4, self.number
            )

        canvas.setFillColor(INK)
        title, size = _fit_text(
            self.title, FONT_SANS_BOLD, SIZE_H1, self.width - text_left, min_size=12.0
        )
        canvas.setFont(FONT_SANS_BOLD, size)
        canvas.drawString(text_left, title_baseline, title)

        if self._subtitle_para is not None:
            self._subtitle_para.drawOn(self.canv, text_left, rule_y + GAP_TIGHT)

        canvas.setStrokeColor(HAIRLINE)
        canvas.setLineWidth(0.8)
        canvas.line(0, rule_y, self.width, rule_y)


class Callout(Flowable):
    """A tinted, rounded note box with a coloured spine."""

    PAD_X = 10.0
    PAD_Y = 8.0
    SPINE = 3.0
    MAX_CHARS = 1600

    def __init__(
        self,
        text: str,
        tone: colors.Color = INFO,
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
        canvas.setFillColor(tint(self.tone, 0.92))
        canvas.setStrokeColor(tint(self.tone, 0.72))
        canvas.setLineWidth(0.6)
        canvas.roundRect(0, box_y, self.width, box_height, 4, stroke=1, fill=1)
        canvas.setFillColor(self.tone)
        canvas.roundRect(0, box_y, self.SPINE + 2, box_height, 2, stroke=0, fill=1)
        canvas.setFillColor(tint(self.tone, 0.92))
        canvas.rect(self.SPINE, box_y + 0.5, 3, box_height - 1, stroke=0, fill=1)

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
            canvas.setStrokeColor(self.color)
            text_color = PAPER
        else:
            canvas.setFillColor(tint(self.color, 0.88))
            canvas.setStrokeColor(tint(self.color, 0.62))
            text_color = self.color
        canvas.setLineWidth(0.5)
        canvas.roundRect(0, 1.0, self.width, self.height - 2.0, 3, stroke=1, fill=1)
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
        top = self.height - self.space_after
        for index, item in enumerate(self.items):
            row, col = divmod(index, self.cols)
            x = col * (tile_w + self.gap)
            y = top - (row + 1) * self.tile_height - row * self.gap
            self._draw_tile(canvas, item, x, y, tile_w)

    def _draw_tile(
        self, canvas: Any, item: KpiItem, x: float, y: float, tile_w: float
    ) -> None:
        accent = item.accent or ACCENT
        canvas.setFillColor(PAGE_TINT)
        canvas.setStrokeColor(HAIRLINE)
        canvas.setLineWidth(0.7)
        canvas.roundRect(x, y, tile_w, self.tile_height, 5, stroke=1, fill=1)
        canvas.setFillColor(accent)
        canvas.roundRect(x, y, 3.0, self.tile_height, 1.5, stroke=0, fill=1)

        pad = 10.0
        inner = tile_w - pad - 8.0
        label, _ = _fit_text(
            item.label.upper(), FONT_SANS_BOLD, SIZE_MICRO - 0.5, inner - 6.0
        )
        _draw_tracked(
            canvas,
            x + pad,
            y + self.tile_height - 15.0,
            label,
            FONT_SANS_BOLD,
            SIZE_MICRO - 0.5,
            MUTED,
            0.9,
        )

        value_baseline = y + (18.0 if item.note else 13.0)
        canvas.setFillColor(item.accent or INK)
        value, size = _fit_text(item.value or NOT_AVAILABLE, FONT_SANS_BOLD, SIZE_H1, inner, min_size=8.0)
        canvas.setFont(FONT_SANS_BOLD, size)
        canvas.drawString(x + pad, value_baseline, value)

        if item.note:
            note, note_size = _fit_text(item.note, FONT_SANS, SIZE_MICRO, inner, min_size=6.0)
            canvas.setFillColor(MUTED)
            canvas.setFont(FONT_SANS, note_size)
            canvas.drawString(x + pad, y + 8.0, note)


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
        top = self.height - self.space_after
        for index, row in enumerate(self.rows):
            y = top - (index + 1) * self.row_height - index * self.gap
            mid = y + (self.row_height / 2.0)
            bar_y = mid - (self.bar_height / 2.0)

            label, size = _fit_text(row.label, FONT_SANS, SIZE_SMALL, label_w, min_size=6.5)
            canvas.setFillColor(INK if row.emphasis else colors.HexColor("#334155"))
            canvas.setFont(FONT_SANS_BOLD if row.emphasis else FONT_SANS, size)
            canvas.drawString(0, mid - (size * 0.35), label)

            canvas.setFillColor(HAIRLINE)
            canvas.roundRect(track_x, bar_y, track_w, self.bar_height, self.bar_height / 2.0, stroke=0, fill=1)

            ratio = min(max(_as_float(row.value) or 0.0, 0.0), 1.0)
            filled = track_w * ratio
            if filled > 0.4:
                canvas.setFillColor(row.color or ACCENT)
                canvas.roundRect(
                    track_x, bar_y, max(filled, self.bar_height), self.bar_height,
                    self.bar_height / 2.0, stroke=0, fill=1,
                )

            text = row.text or f"{ratio * 100:.1f}%"
            value, value_size = _fit_text(text, FONT_MONO, SIZE_MICRO + 0.5, self.value_width, min_size=6.0)
            canvas.setFillColor(MUTED)
            canvas.setFont(FONT_MONO, value_size)
            canvas.drawRightString(self.width, mid - (value_size * 0.35), value)


@dataclass
class DivRow:
    """One row of a diverging bar chart: a signed value against a shared scale."""

    label: str
    value: float
    text: str = ""


class DivergingBarChart(Flowable):
    """Signed bars around a centred zero rule, scaled to the largest magnitude."""

    splittable = True


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
        common = dict(
            row_height=self.row_height, gap=self.gap, label_ratio=self.label_ratio,
            value_width=self.value_width, bar_height=self.bar_height, domain=self.domain,
        )
        head = DivergingBarChart(self.rows[:fits], space_after=self.gap, **common)
        tail = DivergingBarChart(self.rows[fits:], space_after=self.space_after, **common)
        return [head, tail]

    @staticmethod
    def _bar_color(value: float, domain: float) -> colors.Color:
        if value < 0:
            return INFO
        magnitude = abs(value) / domain if domain else 0.0
        return CRITICAL if magnitude >= 0.66 else CAUTION

    def draw(self) -> None:
        if not self.rows:
            return
        canvas = self.canv
        label_w = max(self.width * self.label_ratio, 70.0)
        chart_x = label_w + 8.0
        chart_w = max(self.width - chart_x - self.value_width - 6.0, 30.0)
        center = chart_x + (chart_w / 2.0)
        half = chart_w / 2.0
        top = self.height - self.space_after
        bottom = top - self._content_height()

        canvas.setStrokeColor(HAIRLINE)
        canvas.setLineWidth(0.7)
        canvas.line(center, bottom - 2.0, center, top + 2.0)

        for index, row in enumerate(self.rows):
            y = top - (index + 1) * self.row_height - index * self.gap
            mid = y + (self.row_height / 2.0)
            bar_y = mid - (self.bar_height / 2.0)
            value = _as_float(row.value) or 0.0
            length = min(abs(value) / self.domain, 1.0) * half if self.domain else 0.0

            label, size = _fit_text(row.label, FONT_SANS, SIZE_MICRO + 0.5, label_w, min_size=5.5)
            canvas.setFillColor(colors.HexColor("#334155"))
            canvas.setFont(FONT_SANS, size)
            canvas.drawString(0, mid - (size * 0.35), label)

            if length > 0.3:
                canvas.setFillColor(self._bar_color(value, self.domain))
                x = center - length if value < 0 else center
                canvas.rect(x, bar_y, max(length, 0.8), self.bar_height, stroke=0, fill=1)

            canvas.setFillColor(MUTED)
            canvas.setFont(FONT_MONO, SIZE_MICRO)
            canvas.drawRightString(self.width, mid - (SIZE_MICRO * 0.35), row.text or fmt_z(value))


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
        value = self.value01 or 0.0
        if value >= 0.8:
            return POSITIVE
        if value >= 0.6:
            return CAUTION
        return CRITICAL

    def draw(self) -> None:
        canvas = self.canv
        value_width = 58.0
        track_w = max(self.width - value_width - 8.0, 40.0)
        gap = 3.0
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
                0.9,
            )

        filled = 0 if self.value01 is None else int(round(self.value01 * self.segments))
        color = self._color()
        for index in range(self.segments):
            x = index * (seg_w + gap)
            canvas.setFillColor(color if index < filled else HAIRLINE)
            canvas.roundRect(x, y, seg_w, self.bar_height, 1.5, stroke=0, fill=1)

        canvas.setFillColor(INK if self.value01 is not None else MUTED)
        canvas.setFont(FONT_SANS_BOLD, SIZE_H3)
        text = "n/r" if self.value01 is None else f"{self.value01:.2f}"
        canvas.drawRightString(self.width, y + 2.0, text)


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

_BASE_TABLE_STYLE: List[Tuple[Any, ...]] = [
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LINEBELOW", (0, 0), (-1, -2), 0.4, HAIRLINE),
]


def make_table(
    data: List[List[Any]],
    col_widths: Sequence[float],
    header: bool = True,
    extra_styles: Optional[Sequence[Tuple[Any, ...]]] = None,
    space_after: float = GAP_BLOCK,
) -> Flowable:
    """A table with the document's shared look: hairlines only, tinted header."""
    if not data:
        return Spacer(1, 0)
    style: List[Tuple[Any, ...]] = list(_BASE_TABLE_STYLE)
    if header:
        style.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PAGE_TINT),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, 0), 5),
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


def header_row(labels: Sequence[str]) -> List[Paragraph]:
    return [P(label.upper(), "TableHeader") for label in labels]


def kv_table(pairs: Sequence[Tuple[str, Any]], key_ratio: float = 0.34) -> Flowable:
    """A two-column key/value table with monospaced values."""
    rows = [header_row(["Field", "Value"])]
    for key, value in pairs:
        rows.append([P(key, "TableCell"), P(_s(value, NOT_AVAILABLE), "TableCellMono")])
    key_w = CONTENT_WIDTH * key_ratio
    return make_table(rows, [key_w, CONTENT_WIDTH - key_w])


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


def numbered_list(items: Sequence[Any]) -> List[Flowable]:
    flowables: List[Flowable] = []
    for index, entry in enumerate([e for e in items if _s(e)], start=1):
        flowables.append(
            Paragraph(escape(_s(entry)), _style(_styles(), "Numbered"), bulletText=f"{index}.")
        )
    return flowables


def missing_note(text: str) -> Flowable:
    """The single, consistent way this document reports absent information."""
    return RP(
        f'<font color="#64748B">{escape(text)}</font>',
        "Small",
    )


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
    staged = _RE_LINK.sub(r'<link href="\2" color="#4F46E5">\1</link>', staged)
    for index, span in enumerate(spans):
        replacement = (
            f'<font face="{FONT_MONO}" size="{SIZE_SMALL}" color="#4F46E5">{span}</font>'
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
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -1), 0, PAGE_TINT),
        ],
        space_after=GAP_BLOCK,
    )


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

_DIRECTION_COLORS: Dict[str, colors.Color] = {
    "ABNORMAL_HIGH": CRITICAL,
    "ABNORMAL_LOW": INFO,
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


def _flatten_mapping(
    payload: Any, prefix: str = "", depth: int = 0, out: Optional[List[Tuple[str, str]]] = None
) -> List[Tuple[str, str]]:
    """Flatten a nested mapping into dotted key/value pairs for the appendix."""
    rows = out if out is not None else []
    if depth > MAX_TREE_DEPTH:
        return rows
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else _s(key, "?")
            _flatten_mapping(value, path, depth + 1, rows)
        if not payload and prefix:
            rows.append((prefix, "{}"))
        return rows
    if isinstance(payload, (list, tuple)):
        if not payload:
            rows.append((prefix, "[]"))
        elif all(not isinstance(item, (dict, list, tuple)) for item in payload):
            rows.append((prefix, ", ".join(_s(item, "-") for item in payload)))
        else:
            for index, item in enumerate(payload):
                _flatten_mapping(item, f"{prefix}[{index}]", depth + 1, rows)
        return rows
    if isinstance(payload, bool):
        rows.append((prefix, "true" if payload else "false"))
    elif payload is None:
        rows.append((prefix, "null"))
    elif isinstance(payload, float):
        rows.append((prefix, fmt_number(payload)))
    elif isinstance(payload, int):
        rows.append((prefix, f"{payload:,}"))
    else:
        rows.append((prefix, _s(payload, "-")[:180]))
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
        self.drawRightString(
            PAGE_WIDTH - MARGIN_SIDE, PAGE_HEIGHT - 42.0, f"Page {page} of {total}"
        )
        self.restoreState()


def _furniture(doc: Any) -> Furniture:
    value = getattr(doc, "furniture", None)
    return value if isinstance(value, Furniture) else Furniture()


def draw_cover_page(canvas: Any, doc: Any) -> None:
    """
    Full-bleed accent band, wordmark, title and generation timestamp.

    The band deliberately stops there: the task line belongs to the participant
    block in the frame below, and printing it in both places read as a mistake.
    ``Furniture.subtitle`` survives as document metadata only.
    """
    info = _furniture(doc)
    canvas.saveState()
    band_bottom = PAGE_HEIGHT - COVER_BAND_HEIGHT
    canvas.setFillColor(ACCENT)
    canvas.rect(0, band_bottom, PAGE_WIDTH, COVER_BAND_HEIGHT, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#312E81"))
    canvas.rect(0, band_bottom, PAGE_WIDTH, 4.0, stroke=0, fill=1)

    _draw_tracked(
        canvas,
        MARGIN_SIDE,
        PAGE_HEIGHT - 34.0,
        "COMPASS ENGINE",
        FONT_SANS_BOLD,
        SIZE_MICRO,
        tint(ACCENT, 0.72),
        2.0,
    )

    title, size = _fit_text(
        info.title, FONT_SANS_BOLD, SIZE_DISPLAY, CONTENT_WIDTH, min_size=15.0
    )
    canvas.setFillColor(PAPER)
    canvas.setFont(FONT_SANS_BOLD, size)
    # Sits low in the band now that no subtitle follows it, so the wordmark and
    # the title stay optically centred between the band's edges.
    canvas.drawString(MARGIN_SIDE, PAGE_HEIGHT - 78.0, title)

    if info.timestamp:
        canvas.setFillColor(tint(ACCENT, 0.72))
        canvas.setFont(FONT_SANS, SIZE_MICRO)
        canvas.drawRightString(PAGE_WIDTH - MARGIN_SIDE, PAGE_HEIGHT - 34.0, info.timestamp)

    canvas.setFillColor(MUTED)
    canvas.setFont(FONT_SANS, SIZE_MICRO)
    canvas.drawString(MARGIN_SIDE, 27.0, f"Engine {_ENGINE_VERSION or 'version not recorded'}")
    canvas.restoreState()


def draw_body_page(canvas: Any, doc: Any) -> None:
    """Thin accent rule, participant id and the footer disclaimer."""
    info = _furniture(doc)
    canvas.saveState()
    rule_y = PAGE_HEIGHT - 46.0
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(1.2)
    canvas.line(MARGIN_SIDE, rule_y, PAGE_WIDTH - MARGIN_SIDE, rule_y)

    canvas.setFillColor(MUTED)
    canvas.setFont(FONT_MONO, SIZE_MICRO)
    identifier, _ = _fit_text(
        info.participant_id or "participant not identified",
        FONT_MONO,
        SIZE_MICRO,
        CONTENT_WIDTH - 120.0,
    )
    canvas.drawString(MARGIN_SIDE, PAGE_HEIGHT - 42.0, identifier)

    canvas.setStrokeColor(HAIRLINE)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN_SIDE, 38.0, PAGE_WIDTH - MARGIN_SIDE, 38.0)
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT_SANS, SIZE_MICRO - 0.5)
    footer, _ = _fit_text(info.footer or DISCLAIMER, FONT_SANS, SIZE_MICRO - 0.5, CONTENT_WIDTH)
    canvas.drawString(MARGIN_SIDE, 26.0, footer)
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


_HEADING_STYLES = frozenset(
    {"H1", "H2", "H3", "H4", "MdH1", "MdH2", "MdH3", "MdH4"}
)


def _is_heading(flowable: Any) -> bool:
    style = getattr(flowable, "style", None)
    return bool(style is not None and getattr(style, "name", "") in _HEADING_STYLES)


def _section(
    sections: _Sections, spec: Tuple[str, str], body: Sequence[Flowable]
) -> List[Flowable]:
    """A section header glued to its first block so headings never orphan."""
    title, _ = spec
    header = sections.header(spec)
    content = [flowable for flowable in body if flowable is not None]
    if not content:
        content = [missing_note(f"{title}: no information was recorded for this run.")]
    header.spaceAfter = GAP_TIGHT
    lead: List[Flowable] = [header, content[0]]
    rest = content[1:]
    # A section that opens on a subheading needs the block under it too, or the
    # subheading would be regrouped on its own and strand the section header.
    if rest and _is_heading(content[0]):
        lead.append(rest[0])
        rest = rest[1:]
    flowables: List[Flowable] = [Spacer(1, GAP_SECTION)]
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


def _cover_flowables(ctx: "_RunContext") -> List[Flowable]:
    """
    Participant strip, KPI grid, provenance line, contents and disclaimer, then
    whatever the leftover room can usefully hold.

    The cover is one frame of a known height, so the block is measured before it
    is emitted. A run that has source data to show closes with a glance block; a
    run that has none is centred rather than left hanging above a void.
    """
    core: List[Flowable] = [Spacer(1, GAP_BLOCK)]
    core.append(
        RP(
            f'<font face="{FONT_MONO}" size="{SIZE_H2}" color="#0F172A">'
            f"{escape(ctx.participant_id or 'participant id not recorded')}</font>",
            "Body",
        )
    )
    core.append(P(ctx.task_line, "BodyMuted"))
    core.append(Spacer(1, GAP_BLOCK))
    core.append(KpiGrid(ctx.kpi_items(), cols=3, tile_height=62.0))
    core.append(Spacer(1, GAP_BLOCK + 2))
    core.append(HRule(space_before=0, space_after=GAP_TIGHT))
    core.append(P(ctx.provenance_line(), "Micro"))
    core.append(Spacer(1, GAP_SECTION + 4))
    core.append(_contents_block())
    core.append(Spacer(1, GAP_SECTION))
    core.append(Callout(DISCLAIMER, CAUTION, title="Pre-clinical use only", space_after=0))

    measured = _stack_height(core)
    if measured is None:
        return core
    free = COVER_FRAME_HEIGHT - measured
    glance = _cover_glance(ctx, free)
    if glance:
        return core + glance
    if free > COVER_GLANCE_MIN_SPACE:
        return [Spacer(1, free / 2.0)] + core
    return core


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
    heading = P(f"Most deviating leaves ({len(ranked)} of {len(leaves):,})", "Micro")
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
                        f'<font color="#4F46E5" face="{FONT_SANS_BOLD}">{number}</font>'
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
            ("LEFTPADDING", (0, 0), (0, -1), 0),
        ],
        space_after=0,
    )
    return _titled_block("In this report", table)


def _titled_block(label: str, block: Flowable) -> Flowable:
    """A micro, letterspaced label above a block, kept together with it."""
    heading = P(label.upper(), "Micro")
    heading.spaceAfter = 3.0
    return KeepHeadingWith([heading, block])


def _prediction_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body: List[Flowable] = []
    root_node = _as_dict(ctx.task_root)

    spec_pairs: List[Tuple[str, str]] = [
        ("Task mode", humanize(root_node.get("mode")) if root_node else NOT_AVAILABLE),
        ("Task label", _s(root_node.get("display_name"), NOT_AVAILABLE)),
        ("Task id", _s(ctx.task_spec.get("task_id"), "not assigned")),
        ("Target label", _s(ctx.performance.get("target_condition"), NOT_AVAILABLE)),
        ("Comparator label", _s(ctx.performance.get("control_condition"), "not applicable")),
    ]
    labels = [_s(item) for item in _as_list(root_node.get("class_labels")) if _s(item)]
    outputs = [_s(item) for item in _as_list(root_node.get("regression_outputs")) if _s(item)]
    spec_pairs.append(("Class labels", ", ".join(labels) if labels else "none declared"))
    spec_pairs.append(("Regression outputs", ", ".join(outputs) if outputs else "none declared"))
    body.append(kv_table(spec_pairs))

    body.append(P("Primary output", "H2"))
    if ctx.node_rows:
        body.append(P(ctx.primary_output, "BigValue"))
        confidence_bits = [
            f"Confidence level: {ctx.confidence_level}",
            f"Root confidence: {fmt_prob(ctx.root_confidence)}",
        ]
        probability = _as_float(ctx.prediction_result.get("probability"))
        if probability is not None:
            confidence_bits.append(f"Reported probability: {fmt_prob(probability)}")
        body.append(P("   |   ".join(confidence_bits), "SmallMuted"))
        body.append(Spacer(1, GAP_BLOCK))
    else:
        body.append(missing_note("No prediction node was recorded for this run."))

    summary = _s(ctx.patient_report.get("clinical_summary"))
    if summary:
        body.append(P("Clinical summary", "H3"))
        body.append(P(summary, "Body"))
    else:
        body.append(P("Clinical summary", "H3"))
        body.append(missing_note("The run did not record a clinical summary."))

    root_row = ctx.node_rows[0] if ctx.node_rows else None
    if root_row is not None:
        classification = _as_dict(root_row.payload.get("classification"))
        regression = _as_dict(root_row.payload.get("regression"))
        if classification:
            probabilities = _as_dict(classification.get("probabilities"))
            if probabilities:
                body.append(P("Class probabilities", "H3"))
                predicted = _s(classification.get("predicted_label"))
                rows: List[BarRow] = []
                for label, value in sorted(
                    probabilities.items(), key=lambda item: _as_float(item[1]) or 0.0, reverse=True
                ):
                    score = _as_float(value)
                    rows.append(
                        BarRow(
                            label=_s(label, "?"),
                            value=score or 0.0,
                            text=fmt_prob(score, "not recorded"),
                            color=ACCENT if _s(label) == predicted else tint(ACCENT, 0.55),
                            emphasis=_s(label) == predicted,
                        )
                    )
                body.append(HBarChart(rows, label_ratio=0.42, value_width=60.0))
            else:
                body.append(missing_note("Class probabilities were not recorded."))
        elif regression:
            values = _as_dict(regression.get("values"))
            body.append(P("Estimated values", "H3"))
            if values:
                units = _as_dict(root_node.get("unit_by_output"))
                rows = [header_row(["Output", "Value", "Unit"])]
                for key, value in values.items():
                    rows.append(
                        [
                            P(key, "TableCellMono"),
                            P(fmt_number(value), "TableCellNumeric"),
                            P(_s(units.get(key), "not declared"), "TableCellMuted"),
                        ]
                    )
                widths = [CONTENT_WIDTH * 0.46, CONTENT_WIDTH * 0.22, CONTENT_WIDTH * 0.32]
                body.append(make_table(rows, widths))
            else:
                body.append(missing_note("No regression values were recorded."))

    if len(ctx.node_rows) > 1:
        body.append(P("Hierarchical prediction tree", "H2"))
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
            CONTENT_WIDTH * 0.20,
            CONTENT_WIDTH * 0.22,
            CONTENT_WIDTH * 0.38,
            CONTENT_WIDTH * 0.20,
        ]
        body.append(make_table(rows, widths))

        for node in ctx.node_rows[1:]:
            child_classification = _as_dict(node.payload.get("classification"))
            probabilities = _as_dict(child_classification.get("probabilities"))
            if not probabilities:
                continue
            body.append(P(f"{node.node_id}: class probabilities", "H4"))
            predicted = _s(child_classification.get("predicted_label"))
            bars = [
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
            body.append(HBarChart(bars, label_ratio=0.42, value_width=60.0, space_after=GAP_TIGHT))

    return _section(sections, SECTION_SPECS[0], body)


def _evidence_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body: List[Flowable] = []

    body.append(P("Key findings", "H2"))
    if ctx.key_findings:
        rows = [header_row(["Domain", "Finding", "Direction", "z"])]
        direction_styles: List[Tuple[Any, ...]] = []
        for index, item in enumerate(ctx.key_findings, start=1):
            direction = _s(item.get("direction"), "UNSPECIFIED").upper()
            color = _DIRECTION_COLORS.get(direction, MUTED)
            direction_styles.append(("TEXTCOLOR", (2, index), (2, index), color))
            rows.append(
                [
                    P(_s(item.get("domain"), NOT_AVAILABLE), "TableCellMuted"),
                    P(_s(item.get("finding"), NOT_AVAILABLE), "TableCell"),
                    Paragraph(
                        escape(direction.replace("_", " ").title()),
                        ParagraphStyle(
                            name=f"dir{index}",
                            parent=_style(_styles(), "TableCell"),
                            fontName=FONT_SANS_BOLD,
                            fontSize=SIZE_MICRO + 0.5,
                            textColor=color,
                        ),
                    ),
                    P(fmt_z(_finding_z(item)), "TableCellNumeric"),
                ]
            )
        widths = [
            CONTENT_WIDTH * 0.20,
            CONTENT_WIDTH * 0.50,
            CONTENT_WIDTH * 0.20,
            CONTENT_WIDTH * 0.10,
        ]
        body.append(make_table(rows, widths, extra_styles=direction_styles))
    else:
        body.append(missing_note("No key findings were recorded by the predictor."))

    body.append(P("Reasoning chain", "H2"))
    if ctx.reasoning_chain:
        body.extend(numbered_list(ctx.reasoning_chain))
        body.append(Spacer(1, GAP_TIGHT))
    else:
        body.append(missing_note("No reasoning chain was recorded by the predictor."))

    body.append(P("Supporting evidence", "H2"))
    if ctx.evidence_for or ctx.evidence_against:
        left: List[Flowable] = [P("For the prediction", "H4")]
        left.extend(
            bullet_list(ctx.evidence_for)
            or [missing_note("None recorded.")]
        )
        right: List[Flowable] = [P("Against the prediction", "H4")]
        right.extend(
            bullet_list(ctx.evidence_against)
            or [missing_note("None recorded.")]
        )
        half = CONTENT_WIDTH / 2.0
        body.append(
            make_table(
                [[left, right]],
                [half, half],
                header=False,
                extra_styles=[
                    ("LINEBELOW", (0, 0), (-1, -1), 0, PAPER),
                    ("LEFTPADDING", (0, 0), (0, -1), 0),
                    ("RIGHTPADDING", (0, 0), (0, -1), 10),
                    ("LEFTPADDING", (1, 0), (1, -1), 10),
                ],
            )
        )
    else:
        body.append(missing_note("No supporting evidence was recorded for or against."))

    return _section(sections, SECTION_SPECS[1], body)


def _uncertainty_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body: List[Flowable] = []
    coverage = ctx.coverage_summary

    body.append(P("Uncertainty factors", "H2"))
    if ctx.uncertainty_factors:
        body.extend(bullet_list(ctx.uncertainty_factors))
        body.append(Spacer(1, GAP_TIGHT))
    else:
        body.append(missing_note("No uncertainty factors were recorded by the predictor."))

    body.append(P("Feature ledger", "H2"))
    if coverage:
        missing_count = _as_int(coverage.get("missing_feature_count"))
        pairs = [
            ("Features in the ledger", fmt_int(coverage.get("all_feature_count"))),
            ("Represented in the prediction", fmt_int(coverage.get("represented_feature_count"))),
            ("Processed by tools", fmt_int(coverage.get("processed_feature_count"))),
            ("Passed through unprocessed", fmt_int(coverage.get("unprocessed_raw_feature_count"))),
            ("Missing from the prediction", fmt_int(coverage.get("missing_feature_count"))),
        ]
        body.append(kv_table(pairs, key_ratio=0.44))
        if missing_count is None:
            body.append(missing_note("The run did not record a missing-feature count."))
        elif missing_count == 0:
            body.append(
                P(
                    "No features are missing: every feature in the ledger is represented in the prediction.",
                    "Small",
                )
            )
        else:
            body.append(
                P(f"{missing_count:,} features are missing from the prediction.", "Small")
            )
            names = [_s(name) for name in _as_list(coverage.get("missing_features")) if _s(name)]
            if names:
                body.append(Spacer(1, GAP_TIGHT))
                body.extend(bullet_list(names, limit=MISSING_FEATURE_CAP))
            else:
                body.append(
                    missing_note("The identifiers of the missing features were not recorded.")
                )
        if coverage.get("invariant_ok") is False:
            body.append(Spacer(1, GAP_TIGHT))
            body.append(
                Callout(
                    "The coverage invariant failed for this run: the feature ledger does not "
                    "reconcile with what reached the predictor. Treat the coverage figures "
                    "above as unverified.",
                    CAUTION,
                    title="Coverage invariant not satisfied",
                )
            )
    else:
        body.append(missing_note("No coverage summary was recorded for this run."))

    body.append(P("Domain coverage of the source data", "H2"))
    if ctx.domain_coverage:
        rows = [header_row(["Domain", "Present", "Total", "Coverage", "Missing"])]
        for domain, stats in ctx.domain_coverage:
            rows.append(
                [
                    P(domain, "TableCell"),
                    P(fmt_int(stats.get("present_leaves"), "-"), "TableCellNumeric"),
                    P(fmt_int(stats.get("total_leaves"), "-"), "TableCellNumeric"),
                    P(fmt_pct(stats.get("coverage_percentage"), "-"), "TableCellNumeric"),
                    P(fmt_int(stats.get("missing_count"), "-"), "TableCellNumeric"),
                ]
            )
        widths = [
            CONTENT_WIDTH * 0.36,
            CONTENT_WIDTH * 0.16,
            CONTENT_WIDTH * 0.16,
            CONTENT_WIDTH * 0.16,
            CONTENT_WIDTH * 0.16,
        ]
        body.append(make_table(rows, widths))
    else:
        body.append(
            missing_note(
                "No data overview was supplied, so per-domain coverage could not be reported."
            )
        )

    return _section(sections, SECTION_SPECS[2], body)


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
                color=_coverage_color(percentage),
            )
        )
    return rows


def _coverage_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body: List[Flowable] = []
    if ctx.domain_coverage:
        body.append(
            HBarChart(_coverage_bar_rows(ctx.domain_coverage), label_ratio=0.32, value_width=104.0)
        )
        body.append(
            P(
                "The bar shows the share of the ontology leaves that carry a value for this "
                "participant. Everything else was absent from the source data.",
                "Micro",
            )
        )
    else:
        body.append(
            missing_note(
                "Evidence coverage cannot be charted: no data overview was supplied with this run."
            )
        )
    return _section(sections, SECTION_SPECS[3], body)


def _coverage_color(percentage: Optional[float]) -> colors.Color:
    if percentage is None:
        return MUTED
    if percentage >= 50.0:
        return POSITIVE
    if percentage >= 20.0:
        return ACCENT
    return CAUTION


def _deviation_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body: List[Flowable] = []
    leaves = ctx.deviation_leaves
    if not leaves:
        body.append(
            missing_note(
                "No hierarchical deviation map was supplied, so the deviation profile "
                "could not be computed."
            )
        )
        return _section(sections, SECTION_SPECS[4], body)

    body.append(P("Domain summary", "H2"))
    rows = [header_row(["Domain", "Leaves", "Mean |z|", "Peak |z|"])]
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
    body.append(make_table(rows, widths))

    ranked = sorted(leaves, key=lambda leaf: abs(leaf.score), reverse=True)[:DEVIATION_CHART_ROWS]
    body.append(
        P(
            f"Most deviating leaves ({len(ranked)} of {len(leaves):,})",
            "H2",
        )
    )
    chart_rows = [
        DivRow(label=leaf.path, value=leaf.score, text=fmt_z(leaf.score)) for leaf in ranked
    ]
    body.append(DivergingBarChart(chart_rows, label_ratio=0.46, value_width=38.0))
    body.append(
        P(
            "Bars are signed z-scores on a symmetric axis. Left of the rule is below the "
            "normative mean, right of it is above.",
            "Micro",
        )
    )
    return _section(sections, SECTION_SPECS[4], body)


def _critic_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body: List[Flowable] = []
    evaluation = ctx.evaluation
    verdict = ctx.verdict
    tone = POSITIVE if verdict == "SATISFACTORY" else CAUTION

    header_pairs: List[Tuple[str, str]] = [
        ("Verdict", verdict or NOT_AVAILABLE),
        ("Confidence in verdict", fmt_prob(evaluation.get("confidence_in_verdict"))),
        ("Selected iteration", ctx.iteration_text),
    ]
    passed = _as_int(evaluation.get("checklist_passed"))
    total = _as_int(evaluation.get("checklist_total"))
    if passed is not None and total is not None:
        header_pairs.append(("Checklist", f"{passed} of {total} checks passed"))
    body.append(kv_table(header_pairs, key_ratio=0.34))

    reason = _s(ctx.performance.get("selection_reason"))
    if reason:
        body.append(Callout(reason, tone, title="Attempt selection", space_after=GAP_BLOCK))

    body.append(P("Composite score", "H2"))
    composite = _as_float(evaluation.get("composite_score"))
    if composite is None:
        body.append(
            missing_note("The run did not record a composite critic score for this attempt.")
        )
    else:
        body.append(ScoreMeter(composite, segments=10, label="composite"))

    breakdown = _as_dict(evaluation.get("score_breakdown"))
    body.append(P("Score breakdown", "H3"))
    if breakdown:
        rows = [header_row(["Component", "Score"])]
        for key, value in breakdown.items():
            rows.append([P(humanize(key), "TableCell"), P(fmt_number(value), "TableCellNumeric")])
        body.append(make_table(rows, [CONTENT_WIDTH * 0.68, CONTENT_WIDTH * 0.32]))
    else:
        body.append(missing_note("No per-component score breakdown was recorded."))

    body.append(P("Quality checklist", "H2"))
    checklist = _as_dict(evaluation.get("checklist"))
    if checklist:
        active = [_s(key) for key in _as_list(checklist.get("active_checks")) if _s(key)]
        keys = active or [
            key for key, value in checklist.items() if isinstance(value, bool)
        ]
        rows = [header_row(["Check", "Result"])]
        for key in keys:
            passed_check = bool(checklist.get(key))
            rows.append(
                [
                    P(humanize(key), "TableCell"),
                    Chip("PASS" if passed_check else "FAIL", POSITIVE if passed_check else CRITICAL),
                ]
            )
        body.append(make_table(rows, [CONTENT_WIDTH * 0.74, CONTENT_WIDTH * 0.26]))
        if not active:
            body.append(
                P(
                    "The run did not declare an active-check subset, so every recorded check "
                    "is listed.",
                    "Micro",
                )
            )
    elif passed is not None and total is not None:
        body.append(
            P(
                f"{passed} of {total} checks passed. The per-check detail was not recorded "
                "in the run artifacts.",
                "Small",
            )
        )
    else:
        body.append(missing_note("No quality checklist was recorded for this run."))

    for title, key, tone_for in (
        ("Strengths", "strengths", POSITIVE),
        ("Weaknesses", "weaknesses", CAUTION),
    ):
        entries = [_s(item) for item in _as_list(evaluation.get(key)) if _s(item)]
        body.append(P(title, "H3"))
        if entries:
            body.extend(bullet_list(entries))
            body.append(Spacer(1, GAP_TIGHT))
        else:
            body.append(missing_note(f"No {title.lower()} were recorded."))

    body.append(P("Improvement suggestions", "H3"))
    suggestions = [_as_dict(item) for item in _as_list(evaluation.get("improvement_suggestions"))]
    suggestions = [item for item in suggestions if item]
    if suggestions:
        rows = [header_row(["Priority", "Issue", "Suggestion"])]
        priority_tones = {"HIGH": CRITICAL, "MEDIUM": CAUTION, "LOW": INFO}
        for item in suggestions:
            priority = _s(item.get("priority"), "UNSET").upper()
            rows.append(
                [
                    Chip(priority, priority_tones.get(priority, MUTED)),
                    P(_s(item.get("issue"), NOT_AVAILABLE), "TableCell"),
                    P(_s(item.get("suggestion"), NOT_AVAILABLE), "TableCell"),
                ]
            )
        widths = [CONTENT_WIDTH * 0.16, CONTENT_WIDTH * 0.40, CONTENT_WIDTH * 0.44]
        body.append(make_table(rows, widths))
    else:
        body.append(
            missing_note(
                "No improvement suggestions were recorded, which is expected for a "
                "satisfactory first attempt."
                if ctx.verdict == "SATISFACTORY"
                else "No improvement suggestions were recorded."
            )
        )

    return _section(sections, SECTION_SPECS[5], body)


def _deep_phenotype_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body: List[Flowable] = []
    if ctx.deep_phenotype_markdown.strip():
        body.extend(markdown_to_flowables(ctx.deep_phenotype_markdown, _styles()))
        if not body:
            body.append(missing_note("The deep phenotype report was empty after parsing."))
    else:
        body.append(missing_note("This run did not generate a deep phenotype report."))
        status = _as_dict(ctx.performance.get("deep_phenotype"))
        if status:
            trigger = _s(status.get("trigger_source"), "not requested")
            body.append(
                P(
                    f"Generation flag: {'yes' if status.get('generated') else 'no'}. "
                    f"Trigger source: {trigger}.",
                    "Micro",
                )
            )
    return _section(sections, SECTION_SPECS[6], body)


def _execution_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body: List[Flowable] = []
    plan = _as_dict(ctx.performance.get("plan_summary"))
    body.append(P("Plan", "H2"))
    if plan:
        domains = [_s(item) for item in _as_list(plan.get("priority_domains")) if _s(item)]
        body.append(
            kv_table(
                [
                    ("Plan id", _s(plan.get("plan_id"), NOT_AVAILABLE)),
                    ("Total steps", fmt_int(plan.get("total_steps"))),
                    ("Priority domains", ", ".join(domains) if domains else "none recorded"),
                    ("Iterations", ctx.iteration_text),
                    ("Duration", fmt_duration(ctx.performance.get("total_duration_seconds"))),
                ],
                key_ratio=0.34,
            )
        )
    else:
        body.append(missing_note("No plan summary was recorded for this run."))

    body.append(P("Token ledger by component", "H2"))
    grouped = _group_token_calls(_as_list(ctx.token_usage.get("calls")))
    if grouped:
        rows = [header_row(["Component", "Calls", "Prompt", "Completion", "Total"])]
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
                P("All components", "TableCell"),
                P(f"{totals[0]:,}", "TableCellNumeric"),
                P(f"{totals[1]:,}", "TableCellNumeric"),
                P(f"{totals[2]:,}", "TableCellNumeric"),
                P(f"{totals[3]:,}", "TableCellNumeric"),
            ]
        )
        last = len(rows) - 1
        body.append(
            make_table(
                rows,
                [
                    CONTENT_WIDTH * 0.32,
                    CONTENT_WIDTH * 0.13,
                    CONTENT_WIDTH * 0.18,
                    CONTENT_WIDTH * 0.18,
                    CONTENT_WIDTH * 0.19,
                ],
                extra_styles=[
                    ("BACKGROUND", (0, last), (-1, last), PAGE_TINT),
                    ("FONT", (0, last), (-1, last), FONT_SANS_BOLD),
                    ("LINEABOVE", (0, last), (-1, last), 0.8, colors.HexColor("#CBD5E1")),
                ],
            )
        )
    else:
        recorded_total = _as_int(ctx.token_usage.get("total_tokens"))
        if recorded_total is not None:
            body.append(
                P(
                    f"{recorded_total:,} tokens were recorded in total, but the per-call "
                    "ledger was not saved with this run.",
                    "Small",
                )
            )
        else:
            body.append(missing_note("No token usage was recorded for this run."))

    body.append(P("Cost", "H2"))
    cost = ctx.cost
    lines = [_as_dict(line) for line in _as_list(cost.get("lines"))]
    lines = [line for line in lines if line]
    if lines:
        rows = [header_row(["Model", "Prompt", "Completion", "Total tokens", "USD"])]
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
                P("Total", "TableCell"),
                P("", "TableCellNumeric"),
                P("", "TableCellNumeric"),
                P(fmt_int(cost.get("total_tokens"), "-"), "TableCellNumeric"),
                P(fmt_usd(cost.get("usd")), "TableCellNumeric"),
            ]
        )
        last = len(rows) - 1
        body.append(
            make_table(
                rows,
                [
                    CONTENT_WIDTH * 0.34,
                    CONTENT_WIDTH * 0.15,
                    CONTENT_WIDTH * 0.16,
                    CONTENT_WIDTH * 0.18,
                    CONTENT_WIDTH * 0.17,
                ],
                extra_styles=[
                    ("BACKGROUND", (0, last), (-1, last), PAGE_TINT),
                    ("FONT", (0, last), (-1, last), FONT_SANS_BOLD),
                    ("LINEABOVE", (0, last), (-1, last), 0.8, colors.HexColor("#CBD5E1")),
                ],
            )
        )
        if _as_float(cost.get("usd")) is None:
            body.append(
                P(
                    "Pricing unavailable: at least one model in the ledger has no published "
                    "price, so the total cost cannot be closed out.",
                    "Small",
                )
            )
    else:
        body.append(
            missing_note(
                "Pricing unavailable: no cost ledger was supplied with this run."
            )
        )
    return _section(sections, SECTION_SPECS[7], body)


def _appendix_section(ctx: "_RunContext", sections: _Sections) -> List[Flowable]:
    body: List[Flowable] = []

    body.append(P("Agent instructions", "H2"))
    instructions = _as_dict(ctx.performance.get("agent_instructions"))
    populated = [(key, _s(value)) for key, value in instructions.items() if _s(value)]
    if populated:
        rows = [header_row(["Agent", "Instruction"])]
        for key, value in populated:
            rows.append([P(humanize(key), "TableCell"), P(value, "TableCell")])
        body.append(make_table(rows, [CONTENT_WIDTH * 0.24, CONTENT_WIDTH * 0.76]))
    elif instructions:
        body.append(
            P(
                "No custom agent instructions were supplied: every slot in the run "
                "configuration was left at its default.",
                "Small",
            )
        )
    else:
        body.append(missing_note("No agent instruction block was recorded for this run."))

    body.append(P("Dataflow summary", "H2"))
    dataflow = _as_dict(ctx.performance.get("dataflow_summary"))
    if dataflow:
        pairs = _flatten_mapping(dataflow)
        rows = [header_row(["Key", "Value"])]
        for key, value in pairs:
            rows.append([P(key, "TableCellMono"), P(value, "TableCellMono")])
        body.append(make_table(rows, [CONTENT_WIDTH * 0.52, CONTENT_WIDTH * 0.48]))
    else:
        body.append(missing_note("No dataflow summary was recorded for this run."))

    body.append(P("Explainability", "H2"))
    explainability = _as_dict(ctx.performance.get("explainability"))
    xai = _as_dict(ctx.performance.get("xai_report"))
    if explainability or xai:
        methods = [_s(item) for item in _as_list(explainability.get("methods_requested")) if _s(item)]
        body.append(
            kv_table(
                [
                    ("Enabled", "yes" if explainability.get("enabled") else "no"),
                    ("Status", _s(explainability.get("status"), NOT_AVAILABLE)),
                    ("Reason", _s(explainability.get("reason"), "none recorded")),
                    ("Methods requested", ", ".join(methods) if methods else "none"),
                    ("XAI report generated", "yes" if xai.get("generated") else "no"),
                    ("XAI report path", _s(xai.get("path"), "not written")),
                ],
                key_ratio=0.34,
            )
        )
    else:
        body.append(missing_note("No explainability status was recorded for this run."))

    return _section(sections, SECTION_SPECS[8], body)


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
        mode = humanize(self.task_root.get("mode")) if self.task_root else NOT_AVAILABLE
        name = _s(self.task_root.get("display_name")) or _s(
            self.performance.get("target_condition"), "task not recorded"
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
        parts = [f"Task: {name}", f"Mode: {mode}", tree]
        if detail:
            parts.append(detail)
        return "   |   ".join(parts)

    def models_used(self) -> List[str]:
        seen: List[str] = []
        for line in _as_list(self.cost.get("lines")):
            model = _s(_as_dict(line).get("model"))
            if model and model not in seen:
                seen.append(model)
        return seen

    def provenance_line(self) -> str:
        models = self.models_used()
        parts = [
            f"Engine {_ENGINE_VERSION}" if _ENGINE_VERSION else "Engine version not recorded",
            f"Models: {', '.join(models)}" if models else "Models: not recorded",
            f"Duration: {fmt_duration(self.performance.get('total_duration_seconds'))}",
            f"Executed: {fmt_timestamp(self.performance.get('execution_timestamp'))}",
        ]
        return "   |   ".join(parts)

    def kpi_items(self) -> List[KpiItem]:
        verdict = self.verdict or NOT_AVAILABLE
        verdict_color = POSITIVE if verdict == "SATISFACTORY" else CAUTION
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
                note="" if self.confidence_level == NOT_AVAILABLE else self.confidence_level,
            ),
            KpiItem(
                label="Critic verdict",
                value=verdict.title() if verdict != NOT_AVAILABLE else NOT_AVAILABLE,
                note="automated review",
                accent=verdict_color if verdict != NOT_AVAILABLE else MUTED,
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
    )
    sections = _Sections()

    # The body template takes over from page two; the cover owns page one only.
    # Switching templates before the cover content, rather than after it, means
    # that even an unforeseen overflow lands on a body page instead of a second
    # page wearing the cover's accent band.
    story: List[Flowable] = [NextPageTemplate("Body")]
    story.extend(_cover_flowables(ctx))
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
        _appendix_section,
    )
    for builder, spec in zip(builders, SECTION_SPECS):
        try:
            story.extend(builder(ctx, sections))
        except Exception as error:  # A broken section must not lose the report.
            notice = Callout(
                f"This section could not be rendered from the run artifacts: {error}",
                CRITICAL,
                title="Section unavailable",
            )
            try:
                story.extend(_section(sections, spec, [notice]))
            except Exception:
                story.extend([Spacer(1, GAP_SECTION), notice])

    doc = RunDocTemplate(str(target), ctx.furniture())
    doc.build(story, canvasmaker=NumberedCanvas)
    return target
