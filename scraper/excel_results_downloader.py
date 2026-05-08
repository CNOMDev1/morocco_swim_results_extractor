"""Télécharge les fichiers Excel listés dans les <li> de la page compétitions FRMN."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

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


def collect_excel_links(page_html: str, page_url: str) -> list[tuple[str, str]]:
    """Récupère les liens .xlsx/.xls depuis les <li> du bloc demandé."""
    soup = BeautifulSoup(page_html, "html.parser")
    seen: set[str] = set()
    links: list[tuple[str, str]] = []

    li_nodes = soup.select("#block-frmn-content > div > div li")
    if not li_nodes:
        li_nodes = soup.select("#block-frmn-content li")

    for li in li_nodes:
        a = li.find("a", href=True)
        if not a:
            continue
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue

        path = unquote(urlparse(href).path)
        if not re.search(r"\.xlsx?$", path, re.IGNORECASE):
            continue

        abs_url = urljoin(page_url, href)
        if abs_url in seen:
            continue
        seen.add(abs_url)

        label = " ".join(a.get_text(separator=" ", strip=True).split()) or abs_url
        links.append((abs_url, label))

    return links


def safe_filename_from_url(url: str, index: int) -> str:
    path = unquote(urlparse(url).path)
    name = Path(path).name
    if not name:
        name = f"excel_{index:03d}.xlsx"
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def download_file(session: requests.Session, url: str, dest: Path) -> None:
    with session.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description="Télécharger les résultats Excel FRMN")
    parser.add_argument(
        "--url",
        default=DEFAULT_COMPETITIONS_URL,
        help=f"URL compétitions (défaut: {DEFAULT_COMPETITIONS_URL})",
    )
    parser.add_argument(
        "--out-dir",
        default="data/excel_results",
        type=Path,
        help="Dossier de sortie des fichiers Excel",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Lister les liens Excel sans télécharger",
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    try:
        page_html = fetch_html(session, args.url)
    except requests.RequestException as exc:
        print(f"Erreur HTTP (page compétitions): {exc}", file=sys.stderr)
        return 1

    links = collect_excel_links(page_html, args.url)
    if not links:
        print("Aucun lien Excel trouvé.")
        return 1

    print(f"{len(links)} fichier(s) Excel trouvé(s).")

    if args.dry_run:
        for url, label in links:
            print(f"- {label}\n  {url}")
        return 0

    used_names: dict[str, int] = {}
    ok, fail = 0, 0

    for i, (url, label) in enumerate(links, start=1):
        base_name = safe_filename_from_url(url, i)
        n = used_names.get(base_name, 0)
        used_names[base_name] = n + 1
        if n:
            stem = Path(base_name).stem
            suffix = Path(base_name).suffix or ".xlsx"
            file_name = f"{stem}_{n+1}{suffix}"
        else:
            file_name = base_name

        out_path = args.out_dir / file_name

        try:
            download_file(session, url, out_path)
            ok += 1
            print(f"[ok] ({i}/{len(links)}) {label} -> {out_path}")
        except requests.RequestException as exc:
            fail += 1
            print(f"[échec] {url}\n       {exc}", file=sys.stderr)

    print(f"Terminé: {ok} téléchargé(s), {fail} échec(s).")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
