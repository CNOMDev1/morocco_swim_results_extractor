#!/usr/bin/env python3
"""Scraper dédié aux pages résultats HTML de la FRMN.

- Parcourt les <li> de #block-frmn-content > div > div
- Garde seulement les <li> ayant un lien vers une page .html/.htm
- Ouvre chaque page HTML de résultats
- Extrait les blocs du type:
  * "100 m NAGE LIBRE Dames Classement"
  * "Finale A" (ou Séries)
  * tableau Place/Nom/Nation/...
- Exporte un JSON structuré par page HTML dans un dossier
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://frmnatation.com"
DEFAULT_COMPETITIONS_URL = f"{BASE_URL}/index.php/competitions-resultats"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

AGE_CATEGORIES = {"SENIORS", "JUNIORS", "CADETS", "MINIMES", "BENJAMINS", "POUSSINS"}
ROUND_TITLES = {
    "séries",
    "series",
    "finale",
    "finale a",
    "finale b",
    "finale c",
    "demi-finale",
    "demi-finale a",
    "demi-finale b",
}

STROKE_FROM_LABEL: list[tuple[str, str]] = [
    ("nage libre", "FR"),
    ("4 nages", "4N"),
    ("dos", "DOS"),
    ("brasse", "BR"),
    ("papillon", "PAP"),
]

RESULT_HEADERS = {
    "place": "Place",
    "nom et prénom": "Nom et prénom",
    "nom et prenom": "Nom et prénom",
    "nation": "Nation",
    "naissance": "Naissance",
    "club": "Club",
    "temps": "Temps",
    "points": "Points",
    "temps de passage": "Temps de passage",
}


def fetch_html(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=90)
    response.raise_for_status()
    return response.text


def drupal_content_roots(soup: BeautifulSoup) -> list[Tag]:
    roots = soup.select("#block-frmn-content div[property='schema:text']")
    if not roots:
        roots = soup.select("#block-frmn-content")
    if not roots:
        return [soup]
    return roots


def collect_result_html_links(page_html: str, page_url: str) -> list[tuple[str, str]]:
    """Retourne les liens .html/.htm trouvés dans les <li> du bloc demandé."""
    soup = BeautifulSoup(page_html, "html.parser")
    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    li_nodes = soup.select("#block-frmn-content > div > div li")
    if not li_nodes:
        for root in drupal_content_roots(soup):
            li_nodes.extend(root.select("li"))

    for li in li_nodes:
        a = li.find("a", href=True)
        if not a:
            continue
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        path = unquote(urlparse(href).path)
        if not re.search(r"\.html?$", path, re.IGNORECASE):
            continue
        abs_url = urljoin(page_url, href)
        if abs_url in seen:
            continue
        seen.add(abs_url)
        label = " ".join(a.get_text(separator=" ", strip=True).split()) or abs_url
        out.append((abs_url, label))

    return out


def clean_text(node: Tag) -> str:
    return " ".join(node.get_text(separator=" ", strip=True).split())


def is_event_title(text: str) -> bool:
    if not text or "Classement" not in text:
        return False
    compact = " ".join(text.split())
    if len(compact) > 120:
        return False
    return bool(
        re.match(
            r"^(?:\d+\s*m|4\s*x\s*\d+\s*m)\s+.+\s+Classement$",
            compact,
            re.IGNORECASE,
        )
    )


def parse_table(table: Tag) -> tuple[list[str], list[dict[str, str]]]:
    rows = table.find_all("tr")
    if not rows:
        return [], []

    header_cells = rows[0].find_all(["th", "td"])
    headers = [clean_text(c) for c in header_cells]

    data: list[dict[str, str]] = []
    for row in rows[1:]:
        cells = [clean_text(c) for c in row.find_all(["td", "th"])]
        if not cells:
            continue
        if len(cells) < len(headers):
            cells.extend([""] * (len(headers) - len(cells)))
        if len(cells) > len(headers):
            cells = cells[: len(headers)]
        data.append(
            {(headers[i] if i < len(headers) else f"col_{i+1}"): cells[i] for i in range(len(cells))}
        )

    return headers, data


def is_program_paragraph(text: str) -> bool:
    return " ".join(text.split()).lower() == "programme de la compétition"


def is_program_table(headers: list[str]) -> bool:
    normalized = [" ".join(h.split()).lower() for h in headers]
    return normalized[:3] == ["epreuves au programme", "dames", "messieurs"]


def detect_category_from_text(text: str) -> str:
    compact = " ".join(text.split()).upper()
    return compact if compact in AGE_CATEGORIES else ""


def is_round_title(text: str) -> bool:
    compact = " ".join(text.split())
    lowered = compact.lower()
    if lowered in ROUND_TITLES:
        return True
    return bool(re.match(r"^finale\s+[a-z0-9]+$", lowered, re.IGNORECASE))


def parse_swim_time_seconds(swim_time: str) -> float | None:
    if not swim_time or not isinstance(swim_time, str):
        return None
    s = swim_time.strip()
    if not s or s.lower() in {"frf n.d.", "n.d.", "-"}:
        return None
    try:
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 2:
                minutes, seconds = parts
                return int(minutes) * 60 + float(seconds.replace(",", "."))
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds.replace(",", "."))
        return float(s.replace(",", "."))
    except ValueError:
        return None


def format_swim_time(swim_time: str, seconds: float | None) -> str:
    if swim_time and swim_time.strip():
        return swim_time.strip()
    if seconds is None:
        return ""
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}:{m:02d}:{s:05.2f}"
    if seconds >= 60:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}:{s:05.2f}"
    return f"00:{seconds:05.2f}"


def compute_speed(distance: int | None, swim_time_seconds: float | None) -> float | None:
    if distance is None or swim_time_seconds is None:
        return None
    if distance <= 0 or swim_time_seconds <= 0:
        return None
    return round(distance / swim_time_seconds, 4)


def parse_meet_date(text: str) -> tuple[str, int | None]:
    match = re.search(r"(\d{2})/(\d{2})/(\d{4})\s*$", text.strip())
    if not match:
        return "", None
    day, month, year = match.groups()
    try:
        dt = datetime(int(year), int(month), int(day))
    except ValueError:
        return "", None
    return dt.strftime("%Y-%m-%d"), int(year)


def parse_pool_info(pool_hint: str) -> tuple[str, int]:
    lowered = pool_hint.lower()
    if "grand" in lowered or "50" in lowered:
        return "LCM", 50
    return "SCM", 25


def parse_meet_header(text: str) -> dict[str, Any]:
    swim_date, swim_year = parse_meet_date(text)
    parts = [part.strip() for part in text.split(" - ") if part.strip()]
    location = ""
    meet = text.strip()
    course, pool_length = "SCM", 25

    if len(parts) >= 2 and re.fullmatch(r"\d{2}/\d{2}/\d{4}", parts[-1]):
        if len(parts) >= 3:
            course, pool_length = parse_pool_info(parts[-2])
        if len(parts) >= 4:
            location = parts[-3]
            meet = " - ".join(parts[:-3])
        elif len(parts) == 3:
            meet = parts[0]

    return {
        "SwimDate": swim_date,
        "SwimYear": swim_year,
        "Meet": meet,
        "location": location,
        "Country": "MAR",
        "Course": course,
        "PoolLength": pool_length,
    }


def parse_event_title(title: str, course: str, pool_length: int) -> dict[str, Any] | None:
    compact = " ".join(title.split())
    match = re.match(
        r"^(?:(\d+)\s*x\s*(\d+)\s*m|(\d+)\s*m)\s+(.+?)\s+(Dames|Messieurs)\s+Classement$",
        compact,
        re.IGNORECASE,
    )
    if not match:
        return None

    relay_legs, relay_distance, single_distance, stroke_label, gender_label = match.groups()
    if relay_legs and relay_distance:
        legs = int(relay_legs)
        leg_distance = int(relay_distance)
        distance = legs * leg_distance
        stroke = "REL"
    else:
        distance = int(single_distance)
        stroke = stroke_from_label(stroke_label)

    gender = "F" if gender_label.lower() == "dames" else "M"
    event = f"{distance} {stroke} {course}"
    return {
        "title": compact,
        "Event": event,
        "Distance": distance,
        "Stroke": stroke,
        "Course": course,
        "PoolLength": pool_length,
        "Gender": gender,
    }


def stroke_from_label(label: str) -> str:
    lowered = label.lower()
    if re.search(r"\b4\s*x\b", lowered):
        return "REL"
    for token, code in STROKE_FROM_LABEL:
        if token in lowered:
            return code
    return ""


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        mapped = RESULT_HEADERS.get(" ".join(key.split()).lower())
        if mapped:
            normalized[mapped] = value
    return normalized


def infer_status(place: str, swim_time: str) -> str:
    place_u = place.strip().upper()
    time_l = swim_time.strip().lower()
    if place_u.startswith("N.C"):
        return "NC"
    if "disqual" in time_l or time_l.startswith("dsq"):
        return "DSQ"
    if "abandon" in time_l:
        return "DNF"
    if "frf n.d" in time_l or time_l in {"n.d.", "-"}:
        return "DNS"
    return "OK"


def parse_rank(place: str) -> int | None:
    match = re.match(r"(\d+)\.", place.strip())
    if match:
        return int(match.group(1))
    return None


def parse_performance(
    row: dict[str, str],
    gender: str,
    swim_year: int | None,
    distance: int | None,
) -> dict[str, Any] | None:
    normalized = normalize_row(row)
    name = normalized.get("Nom et prénom", "").strip()
    club = normalized.get("Club", "").strip()
    if not name and not club:
        return None

    place = normalized.get("Place", "")
    swim_time_raw = normalized.get("Temps", "").strip()
    status = infer_status(place, swim_time_raw)
    rank = parse_rank(place)
    swim_secs = parse_swim_time_seconds(swim_time_raw)
    swim_time = format_swim_time(swim_time_raw, swim_secs)
    speed = compute_speed(distance, swim_secs)

    birth_raw = normalized.get("Naissance", "").strip()
    year_of_birth: int | None = None
    if birth_raw.isdigit():
        year_of_birth = int(birth_raw)

    age: int | None = None
    if swim_year is not None and year_of_birth is not None:
        age = swim_year - year_of_birth

    return {
        "Rank": rank,
        "club": club,
        "SwimTime": swim_time,
        "SwimTimeSeconds": swim_secs,
        "Status": status,
        "Speed": speed,
        "swimmer": {
            "Name": name,
            "Gender": gender,
            "Year_of_birth": year_of_birth,
            "Age": age,
            "Nationality": normalized.get("Nation", "").strip(),
        },
    }


def build_tour(category: str, round_name: str) -> str:
    parts = [part for part in (category, round_name) if part]
    return " ".join(parts)


def parse_results_page(page_html: str) -> dict[str, Any]:
    soup = BeautifulSoup(page_html, "html.parser")
    meet_info: dict[str, Any] = {
        "SwimDate": "",
        "SwimYear": None,
        "Meet": "",
        "location": "",
        "Country": "MAR",
        "Course": "SCM",
        "PoolLength": 25,
    }
    epreuves: list[dict[str, Any]] = []
    current_event: dict[str, Any] | None = None
    current_category = ""
    current_round = ""
    header_seen = False

    for node in soup.find_all(["p", "table"]):
        if node.name == "p":
            text = clean_text(node)
            if not text or is_program_paragraph(text):
                continue

            if not header_seen and re.search(r"\d{2}/\d{2}/\d{4}\s*$", text):
                meet_info.update(parse_meet_header(text))
                header_seen = True
                continue

            if is_event_title(text):
                current_event = parse_event_title(
                    text,
                    meet_info["Course"],
                    meet_info["PoolLength"],
                )
                continue

            detected = detect_category_from_text(text)
            if detected:
                current_category = detected
                continue

            if is_round_title(text):
                current_round = " ".join(text.split())
                continue

            continue

        headers, rows = parse_table(node)
        if not headers or is_program_table(headers):
            continue
        if current_event is None:
            continue

        performances: list[dict[str, Any]] = []
        for row in rows:
            perf = parse_performance(
                row,
                current_event["Gender"],
                meet_info.get("SwimYear"),
                current_event.get("Distance"),
            )
            if perf:
                performances.append(perf)

        if not performances:
            continue

        epreuves.append(
            {
                "Event": current_event["Event"],
                "Distance": current_event["Distance"],
                "Stroke": current_event["Stroke"],
                "Course": current_event["Course"],
                "PoolLength": current_event["PoolLength"],
                "tour": build_tour(current_category, current_round),
                "performances": performances,
            }
        )

    return {
        "SwimDate": meet_info["SwimDate"],
        "SwimYear": meet_info["SwimYear"],
        "Meet": meet_info["Meet"],
        "location": meet_info["location"],
        "Country": meet_info["Country"],
        "epreuves": epreuves,
    }


def slugify(text: str) -> str:
    base = unquote(text or "").strip().lower()
    base = re.sub(r"\s+", "_", base)
    base = re.sub(r"[^a-z0-9_-]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "result"


def filename_for_html_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    stem = Path(path).stem
    return f"{slugify(stem)}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Scraper des pages HTML de résultats FRMN")
    parser.add_argument(
        "--competitions-url",
        default=DEFAULT_COMPETITIONS_URL,
        help="URL de la page compétitions",
    )
    parser.add_argument(
        "--page-url",
        default="",
        help="URL directe d'une page HTML de résultats (optionnel)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/html_results",
        type=Path,
        help="Dossier de sortie (un fichier JSON par page HTML)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limiter le nombre de pages HTML à traiter (0 = toutes)",
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    targets: list[tuple[str, str]] = []

    try:
        if args.page_url:
            targets = [(args.page_url, args.page_url)]
        else:
            html = fetch_html(session, args.competitions_url)
            targets = collect_result_html_links(html, args.competitions_url)
    except requests.RequestException as exc:
        print(f"Erreur HTTP: {exc}", file=sys.stderr)
        return 1

    if not targets:
        print("Aucun lien HTML trouvé.")
        return 1

    if args.limit and args.limit > 0:
        targets = targets[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for idx, (url, _label) in enumerate(targets, start=1):
        out_file = args.out_dir / filename_for_html_url(url)
        try:
            page_html = fetch_html(session, url)
            page_payload = parse_results_page(page_html)
            out_file.write_text(
                json.dumps(page_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written += 1
            n_epreuves = len(page_payload.get("epreuves", []))
            n_perf = sum(len(e.get("performances", [])) for e in page_payload.get("epreuves", []))
            print(
                f"[ok] ({idx}/{len(targets)}) {url} -> "
                f"{n_epreuves} épreuve(s), {n_perf} performance(s) -> {out_file}"
            )
        except requests.RequestException as exc:
            print(f"[échec] {url}\n       {exc}")
            error_payload = {
                "SwimDate": "",
                "SwimYear": None,
                "Meet": "",
                "location": "",
                "Country": "",
                "epreuves": [],
                "error": str(exc),
                "url": url,
            }
            out_file.write_text(
                json.dumps(error_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(f"Terminé: {written}/{len(targets)} fichier(s) JSON écrits dans {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
