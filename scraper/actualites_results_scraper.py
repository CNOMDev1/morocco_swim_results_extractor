#!/usr/bin/env python3
"""Scrape les actualites FRMN et telecharge les PDF de resultats.

Workflow:
1) Ouvre la page "Actualites & Communiques"
2) Parcourt tous les blocs .row.item-actu de chaque page de pagination
3) Garde les actualites contenant "resultat"/"resultats"
4) Ouvre chaque actualite retenue et extrait les liens PDF
5) Telecharge les PDF trouves
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://frmnatation.com"
DEFAULT_ACTUALITES_URL = f"{BASE_URL}/index.php/actualites"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def log(message: str) -> None:
    print(message, flush=True)


def fetch_html(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=90)
    response.raise_for_status()
    return response.text


def clean_text(raw: str) -> str:
    return " ".join((raw or "").split())


def extract_year_from_date_text(date_text: str) -> str:
    """Extrait l'annee depuis un format de date type jj/mm/aaaa."""
    m = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", clean_text(date_text))
    if m:
        return m.group(3)
    return "inconnue"


def normalize_for_match(raw: str) -> str:
    text = clean_text(raw).lower()
    # Supprime les accents de facon generique: "resultats" matche aussi "résultats".
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def contains_resultats_keyword(text: str) -> bool:
    normalized = normalize_for_match(text)
    return bool(re.search(r"\bresultat(?:s)?\b", normalized))


def extract_item_text(item: Tag) -> str:
    title = clean_text(item.select_one("h4 a").get_text(" ", strip=True) if item.select_one("h4 a") else "")
    date = clean_text(item.select_one("span").get_text(" ", strip=True) if item.select_one("span") else "")
    desc = clean_text(item.select_one("p").get_text(" ", strip=True) if item.select_one("p") else "")
    return " | ".join(x for x in [title, date, desc] if x)


def parse_actualites_items(page_html: str, page_url: str) -> list[dict[str, str]]:
    """Extrait les .row.item-actu visibles sur la page actuelle."""
    soup = BeautifulSoup(page_html, "html.parser")
    items: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    item_nodes = soup.select("#block-frmn-content .row.item-actu")
    for node in item_nodes:
        title_link = node.select_one("h4 a[href]")
        if not title_link:
            continue

        href = (title_link.get("href") or "").strip()
        if not href:
            continue

        article_url = urljoin(page_url, href)
        if article_url in seen_urls:
            continue
        seen_urls.add(article_url)

        title = clean_text(title_link.get_text(" ", strip=True))
        date_text = clean_text(node.select_one("span").get_text(" ", strip=True) if node.select_one("span") else "")
        desc_text = clean_text(node.select_one("p").get_text(" ", strip=True) if node.select_one("p") else "")
        blob = extract_item_text(node)

        items.append(
            {
                "title": title or article_url,
                "date": date_text,
                "year": extract_year_from_date_text(date_text),
                "description": desc_text,
                "article_url": article_url,
                "search_blob": blob,
            }
        )

    return items


def find_next_page_url(page_html: str, page_url: str) -> str:
    """Trouve la page suivante via la pagination FRMN, sinon vide."""
    soup = BeautifulSoup(page_html, "html.parser")
    pager = soup.select_one("#block-frmn-content .pagination-actu.fix-w.pager")
    if not pager:
        return ""

    # Cas standard Drupal: lien "next"
    next_anchor = pager.select_one("li.pager__item--next a[href], li.pager__item.next a[href]")
    if next_anchor:
        return urljoin(page_url, (next_anchor.get("href") or "").strip())

    # Fallback: a partir de l'item actif ?page=N, prendre N+1 si present.
    active = pager.select_one("li.pager__item.active a[href]")
    if not active:
        return ""

    active_href = (active.get("href") or "").strip()
    m = re.search(r"[?&]page=(\d+)", active_href)
    if not m:
        return ""
    current_idx = int(m.group(1))
    target_page = current_idx + 1

    for a in pager.select("a[href]"):
        href = (a.get("href") or "").strip()
        m2 = re.search(r"[?&]page=(\d+)", href)
        if m2 and int(m2.group(1)) == target_page:
            return urljoin(page_url, href)

    return ""


def collect_resultats_articles(
    session: requests.Session,
    actualites_url: str,
    limit_pages: int = 0,
) -> list[dict[str, str]]:
    """Parcourt toute la pagination et garde les actualites liees aux resultats."""
    to_visit = actualites_url
    visited: set[str] = set()
    page_count = 0
    selected: list[dict[str, str]] = []
    seen_articles: set[str] = set()

    while to_visit and to_visit not in visited:
        if limit_pages > 0 and page_count >= limit_pages:
            break

        visited.add(to_visit)
        page_count += 1
        html = fetch_html(session, to_visit)
        log(f"[pagination] page {page_count}: {to_visit}")

        page_items = parse_actualites_items(html, to_visit)
        log(f"[pagination] {len(page_items)} actualite(s) trouvee(s) sur cette page")
        for item in page_items:
            if item["article_url"] in seen_articles:
                continue
            seen_articles.add(item["article_url"])
            if contains_resultats_keyword(item["search_blob"]):
                selected.append(item)
                log(f"[match item-actu] {item['title']}")

        to_visit = find_next_page_url(html, to_visit)

    log(
        f"[pagination] fin parcours: {page_count} page(s), "
        f"{len(selected)} actualite(s) candidate(s)"
    )
    return selected


def collect_all_articles(
    session: requests.Session,
    actualites_url: str,
    limit_pages: int = 0,
) -> list[dict[str, str]]:
    """Recupere toutes les actualites, sans filtre mot-cle."""
    to_visit = actualites_url
    visited: set[str] = set()
    page_count = 0
    all_items: list[dict[str, str]] = []
    seen_articles: set[str] = set()

    while to_visit and to_visit not in visited:
        if limit_pages > 0 and page_count >= limit_pages:
            break
        visited.add(to_visit)
        page_count += 1
        html = fetch_html(session, to_visit)
        log(f"[fallback] page {page_count}: {to_visit}")
        page_items = parse_actualites_items(html, to_visit)
        log(f"[fallback] {len(page_items)} actualite(s) trouvee(s) sur cette page")
        for item in page_items:
            if item["article_url"] in seen_articles:
                continue
            seen_articles.add(item["article_url"])
            all_items.append(item)
        to_visit = find_next_page_url(html, to_visit)

    log(f"[fallback] total actualites candidates: {len(all_items)}")
    return all_items


def collect_pdf_links_from_article(article_html: str, article_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(article_html, "html.parser")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    for a in soup.select("#block-frmn-content a[href]"):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        path = unquote(urlparse(href).path)
        if not re.search(r"\.pdf$", path, re.IGNORECASE):
            continue
        abs_url = urljoin(article_url, href)
        if abs_url in seen:
            continue
        seen.add(abs_url)
        label = clean_text(a.get_text(" ", strip=True)) or Path(path).name or abs_url
        out.append((abs_url, label))

    return out


def safe_filename_from_url(url: str, fallback: str) -> str:
    path = unquote(urlparse(url).path)
    name = Path(path).name or fallback
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def download_file(session: requests.Session, url: str, destination: Path) -> None:
    with session.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scraper les actualites FRMN et telecharger les PDF de resultats"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_ACTUALITES_URL,
        help=f"URL de la page actualites (defaut: {DEFAULT_ACTUALITES_URL})",
    )
    parser.add_argument(
        "--out-dir",
        default="data/pdfs_results_actualites",
        type=Path,
        help="Dossier de sortie des PDF",
    )
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=0,
        help="Limiter le nombre de pages de pagination (0 = toutes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Lister les articles/URLs PDF sans telecharger",
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    try:
        articles = collect_resultats_articles(
            session=session,
            actualites_url=args.url,
            limit_pages=args.limit_pages,
        )
    except requests.RequestException as exc:
        print(f"Erreur HTTP pendant le parcours des actualites: {exc}", file=sys.stderr)
        return 1

    if not articles:
        log(
            "Aucune actualite contenant 'resultat(s)' dans item-actu. "
            "Passage en mode fallback: analyse de toutes les actualites."
        )
        try:
            articles = collect_all_articles(
                session=session,
                actualites_url=args.url,
                limit_pages=args.limit_pages,
            )
        except requests.RequestException as exc:
            print(f"Erreur HTTP pendant le fallback: {exc}", file=sys.stderr)
            return 1
        if not articles:
            log("Aucune actualite a analyser.")
            return 1

    log(f"{len(articles)} actualite(s) a analyser.")

    all_pairs: list[tuple[str, str, str, str]] = []  # (title, year, pdf_url, pdf_label)
    for item in articles:
        article_url = item["article_url"]
        article_year = item.get("year", "inconnue") or "inconnue"
        log(f"[analyse article] {item['title']}")
        try:
            article_html = fetch_html(session, article_url)
            pdf_links = collect_pdf_links_from_article(article_html, article_url)
            if pdf_links:
                # Si fallback actif, on ne garde que les PDF lies aux resultats.
                if not contains_resultats_keyword(item["search_blob"]):
                    pdf_links = [
                        (url, label)
                        for (url, label) in pdf_links
                        if contains_resultats_keyword(label)
                    ]
                for pdf_url, pdf_label in pdf_links:
                    all_pairs.append((item["title"], article_year, pdf_url, pdf_label))
                log(f"[article] {item['title']} -> {len(pdf_links)} PDF")
            else:
                log(f"[article] {item['title']} -> 0 PDF")
        except requests.RequestException as exc:
            print(f"[echec article] {article_url}\n  {exc}", file=sys.stderr)

    if not all_pairs:
        log("Aucun lien PDF n'a ete trouve dans les actualites filtrees.")
        return 1

    log(f"{len(all_pairs)} lien(s) PDF trouve(s) au total.")
    if args.dry_run:
        for title, year, pdf_url, pdf_label in all_pairs:
            log(f"- {title} ({year})\n  [{pdf_label}] {pdf_url}")
        return 0

    used_names: dict[str, int] = {}
    ok, fail = 0, 0
    for i, (title, year, pdf_url, pdf_label) in enumerate(all_pairs, start=1):
        base_name = safe_filename_from_url(pdf_url, fallback=f"resultat_{i:03d}.pdf")
        n = used_names.get(base_name, 0)
        used_names[base_name] = n + 1
        if n:
            stem = Path(base_name).stem
            file_name = f"{stem}_{n+1}.pdf"
        else:
            file_name = base_name
        out_path = args.out_dir / year / file_name

        try:
            download_file(session, pdf_url, out_path)
            ok += 1
            log(
                f"[ok] ({i}/{len(all_pairs)}) {title} ({year}) | "
                f"{pdf_label} -> {out_path}"
            )
        except requests.RequestException as exc:
            fail += 1
            print(f"[echec] {pdf_url}\n  {exc}", file=sys.stderr)

    log(f"Termine: {ok} telecharge(s), {fail} echec(s).")
    return 0 if ok > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
