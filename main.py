#!/usr/bin/env python3
"""Orchestrateur: lance le scraping PDF, Excel et HTML en une seule commande."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import requests

from scraper.pdf_results_scraper import (
    DEFAULT_PAGE,
    USER_AGENT,
    collect_pdf_urls,
    download_file as download_pdf_file,
    fetch_html,
    filename_for_url,
)

DATA_DIR = Path("data")
PDF_RESULTS_DIR = DATA_DIR / "pdfs_results"
EXCEL_RESULTS_DIR = DATA_DIR / "excel_results"
HTML_RESULTS_DIR = DATA_DIR / "html_results"
ACTUALITES_RESULTS_DIR = DATA_DIR / "pdfs_results_actualites"


def run_pdf_scraper() -> int:
    """Télécharge les PDF depuis la page compétitions."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    try:
        html = fetch_html(session, DEFAULT_PAGE)
    except requests.RequestException as exc:
        print(f"[PDF][échec] Erreur HTTP: {exc}")
        return 1

    pairs = collect_pdf_urls(html, DEFAULT_PAGE)
    if not pairs:
        print("[PDF] Aucun lien PDF trouvé.")
        return 1

    out_dir = PDF_RESULTS_DIR
    used_names: dict[str, int] = {}
    ok, fail = 0, 0

    print(f"[PDF] {len(pairs)} fichier(s) trouvé(s).")
    for i, (url, label) in enumerate(pairs, start=1):
        base_name = filename_for_url(url, i)
        n = used_names.get(base_name, 0)
        used_names[base_name] = n + 1
        out_name = base_name if n == 0 else f"{base_name.rsplit('.', 1)[0]}_{n+1}.pdf"
        dest = out_dir / out_name
        try:
            download_pdf_file(session, url, dest)
            ok += 1
            print(f"[PDF][ok] ({i}/{len(pairs)}) {label} -> {dest}")
        except requests.RequestException as exc:
            fail += 1
            print(f"[PDF][échec] {url}\n             {exc}")

    print(f"[PDF] Terminé: {ok} téléchargé(s), {fail} échec(s).")
    # Certains liens FRMN sont cassés (404): on ne bloque pas tout le workflow pour ça.
    return 0


def run_script(script_name: str, args: list[str]) -> int:
    """Exécute un script Python local et renvoie son exit code."""
    cmd = [sys.executable, script_name, *args]
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1) Scraping PDF ===")
    pdf_code = run_pdf_scraper()

    print("\n=== 2) Scraping Excel ===")
    excel_code = run_script(
        "scraper/excel_results_downloader.py", ["--out-dir", str(EXCEL_RESULTS_DIR)]
    )

    print("\n=== 3) Scraping HTML ===")
    html_code = run_script(
        "scraper/html_results_scraper.py", ["--out-dir", str(HTML_RESULTS_DIR)]
    )

    print("\n=== 4) Scraping Actualites (PDF) ===")
    actualites_code = run_script(
        "scraper/actualites_results_scraper.py",
        ["--out-dir", str(ACTUALITES_RESULTS_DIR)],
    )

    final_code = (
        0
        if (pdf_code == 0 and excel_code == 0 and html_code == 0 and actualites_code == 0)
        else 1
    )
    print(
        "\nRésumé: "
        f"PDF={pdf_code} | Excel={excel_code} | HTML={html_code} | "
        f"Actualites={actualites_code} | Exit={final_code}"
    )
    return final_code


if __name__ == "__main__":
    raise SystemExit(main())
