"""Deterministic resolution of time phrases -> (start, end) date windows."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from config import ANCHOR_DATE

MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                           "september", "october", "november", "december"])}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"


@dataclass
class Window:
    start: date
    end: date
    label: str

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def previous(self) -> "Window":
        """The immediately preceding window of the same length (default comparison)."""
        end = self.start - timedelta(days=1)
        return Window(end - timedelta(days=self.days - 1), end, f"previous {self.days} days")

    def to_dict(self):
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "label": self.label}


def _month_window(month: int, year: int, label: str) -> Window:
    start = date(year, month, 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return Window(start, nxt - timedelta(days=1), label)


def _parse_day(text: str, default_year: int) -> date | None:
    text = text.strip().lower()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))
    m = re.fullmatch(rf"{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?", text)
    if m:
        return date(int(m[3] or default_year), MONTHS[m[1][:3]], int(m[2]))
    m = re.fullmatch(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MON}(?:,?\s*(\d{{4}}))?", text)
    if m:
        return date(int(m[3] or default_year), MONTHS[m[2][:3]], int(m[1]))
    return None


_MON_NC = _MON.replace("(", "(?:", 1)
_DAY = rf"(\d{{4}}-\d{{2}}-\d{{2}}|{_MON_NC}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MON_NC}(?:,?\s*\d{{4}})?)"


def parse_windows(question: str, anchor: date = ANCHOR_DATE) -> list[Window]:
    """Return all windows mentioned, in order of appearance (first = focus window)."""
    q = question.lower()
    found: list[tuple[int, Window]] = []
    monday = anchor - timedelta(days=anchor.weekday())

    for m in re.finditer(rf"{_DAY}\s*(?:to|until|till|through|-|–|and)\s*{_DAY}", q):
        a, b = _parse_day(m[1], anchor.year), _parse_day(m[2], anchor.year)
        if a and b:
            found.append((m.start(), Window(min(a, b), max(a, b), f"{min(a, b)} to {max(a, b)}")))
    for m in re.finditer(rf"week (?:of|starting|beginning)\s+{_DAY}", q):
        d = _parse_day(m[1], anchor.year)
        if d:
            s = d - timedelta(days=d.weekday())
            found.append((m.start(), Window(s, s + timedelta(days=6), f"week of {s}")))
    rel = [
        (r"\blast week\b|\bprevious week\b|\bpast week\b", lambda: Window(monday - timedelta(days=7), monday - timedelta(days=1), "last week")),
        (r"\bthis week\b", lambda: Window(monday, anchor, "this week")),
        (r"\bweek before last\b", lambda: Window(monday - timedelta(days=14), monday - timedelta(days=8), "week before last")),
        (r"\byesterday\b", lambda: Window(anchor - timedelta(days=1), anchor - timedelta(days=1), "yesterday")),
        (r"\blast month\b|\bprevious month\b", lambda: _month_window(*((anchor.month - 2) % 12 + 1, anchor.year - (anchor.month == 1)), "last month")),
        (r"\bthis month\b", lambda: Window(anchor.replace(day=1), anchor, "this month")),
        (r"\blast quarter\b|\bpast quarter\b|\blast 3 months\b|\blast three months\b", lambda: Window(anchor - timedelta(days=91), anchor - timedelta(days=1), "last ~3 months")),
    ]
    for pat, fn in rel:
        for m in re.finditer(pat, q):
            found.append((m.start(), fn()))
    for m in re.finditer(r"\b(?:last|past|previous)\s+(\d+)\s+(day|week|month)s?\b", q):
        n, unit = int(m[1]), m[2]
        days = n * {"day": 1, "week": 7, "month": 30}[unit]
        found.append((m.start(), Window(anchor - timedelta(days=days), anchor - timedelta(days=1), f"last {n} {unit}s")))
    taken = [(p, p + 40) for p, _ in found]
    for m in re.finditer(rf"\b(?:in|for|during|of|vs\.?|versus|and|compared to|to)?\s*{_MON}\b(?!\s+\d)(?:\s+(\d{{4}}))?", q):
        if any(a <= m.start() < b for a, b in taken):
            continue
        if re.match(r"\s*may\b", m.group(0)) and not re.search(r"\b(in|for|during)\s+may\b", q):
            continue
        mon = MONTHS[m[1][:3]]
        yr = int(m[2]) if m[2] else (anchor.year if mon <= anchor.month else anchor.year - 1)
        found.append((m.start(), _month_window(mon, yr, f"{m[1].capitalize()} {yr}")))
    for m in re.finditer(rf"\bon\s+{_DAY}", q):
        d = _parse_day(m[1], anchor.year)
        if d:
            found.append((m.start(), Window(d, d, str(d))))

    found.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _, w in found:
        if (w.start, w.end) not in seen:
            seen.add((w.start, w.end)); out.append(w)
    return out
