#!/usr/bin/env python3
"""Scraper dédié aux pages résultats HTML de la FRMN.

- Parcourt les <li> de #block-frmn-content > div > div
- Garde seulement les <li> ayant un lien vers une page .html/.htm
- Ouvre chaque page HTML de résultats
- Extrait les blocs du type:
  * "100 m NAGE LIBRE Dames Classement"
  * "Finale A" (ou Séries)
  * tableau Place/Nom/Nation/...
- Exporte un JSON par page HTML dans un dossier
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://frmnatation.com"
DEFAULT_COMPETITIONS_URL = f"{BASE_URL}/index.php/competitions-resultats"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def fetch_html(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=90)
    response.raise_for_status()
    return response.text


def drupal_content_roots(soup: BeautifulSoup) -> list[Tag]:
    roots = soup.select("#block-frmn-content div[property='schema:text']")
    if not roots:
        roots = soup.select("#block-frmn-content")
    if not roots:
        return [soup]  # fallback robuste
    return roots


def collect_result_html_links(page_html: str, page_url: str) -> list[tuple[str, str]]:
    """Retourne les liens .html/.htm trouvés dans les <li> du bloc demandé."""
    soup = BeautifulSoup(page_html, "html.parser")
    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    # Ciblage demandé: <li> dans #block-frmn-content > div > div
    li_nodes = soup.select("#block-frmn-content > div > div li")
    if not li_nodes:
        # Fallback robuste
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
    """Filtre les vrais titres d'épreuves (évite les gros blocs de texte bruités)."""
    if not text or "Classement" not in text:
        return False
    compact = " ".join(text.split())
    # Exemples visés:
    # - 100 m NAGE LIBRE Dames Classement
    # - 4 x 100 m NAGE LIBRE Messieurs Classement
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
        data.append({headers[i] if i < len(headers) else f"col_{i+1}": cells[i] for i in range(len(cells))})

    return headers, data


def is_program_paragraph(text: str) -> bool:
    return " ".join(text.split()).lower() == "programme de la compétition"


def is_program_table(headers: list[str]) -> bool:
    normalized = [" ".join(h.split()).lower() for h in headers]
    return normalized[:3] == ["epreuves au programme", "dames", "messieurs"]


def detect_category_from_text(text: str) -> str:
    """Détecte la catégorie d'âge/serie à partir d'un paragraphe."""
    compact = " ".join(text.split()).upper()
    allowed = {"SENIORS", "JUNIORS", "CADETS", "MINIMES", "BENJAMINS", "POUSSINS"}
    return compact if compact in allowed else ""


def extract_page_content(page_html: str) -> dict:
    """Extrait le contenu de la page en excluant le bloc Programme."""
    soup = BeautifulSoup(page_html, "html.parser")
    paragraphs: list[str] = []
    tables: list[dict] = []
    current_category = ""

    for node in soup.find_all(["p", "table"]):
        if node.name == "p":
            text = clean_text(node)
            if not text:
                continue
            if is_program_paragraph(text):
                continue
            detected = detect_category_from_text(text)
            if detected:
                current_category = detected
            paragraphs.append(text)
            continue

        headers, rows = parse_table(node)
        if not headers:
            continue
        if is_program_table(headers):
            continue
        tables.append(
            {
                "category": current_category,
                "headers": headers,
                "rows": rows,
            }
        )

    return {
        "paragraphs": paragraphs,
        "tables": tables,
        "paragraph_count": len(paragraphs),
        "table_count": len(tables),
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

    for idx, (url, label) in enumerate(targets, start=1):
        try:
            page_html = fetch_html(session, url)
            content = extract_page_content(page_html)
            page_payload = {
                "url": url,
                "label": label,
                "paragraph_count": content["paragraph_count"],
                "table_count": content["table_count"],
                "paragraphs": content["paragraphs"],
                "tables": content["tables"],
            }
            out_file = args.out_dir / filename_for_html_url(url)
            out_file.write_text(
                json.dumps(page_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written += 1
            print(
                f"[ok] ({idx}/{len(targets)}) {url} -> {content['table_count']} table(s) -> {out_file}"
            )
        except requests.RequestException as exc:
            print(f"[échec] {url}\n       {exc}")
            page_payload = {
                "url": url,
                "label": label,
                "error": str(exc),
                "section_count": 0,
                "sections": [],
            }
            out_file = args.out_dir / filename_for_html_url(url)
            out_file.write_text(
                json.dumps(page_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(f"Terminé: {written}/{len(targets)} fichier(s) JSON écrits dans {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
