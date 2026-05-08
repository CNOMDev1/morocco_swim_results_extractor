"""Logique de scraping et téléchargement des PDF FRMN (Compétitions & Résultats)."""

from __future__ import annotations
import re
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://frmnatation.com"
DEFAULT_PAGE = f"{BASE_URL}/index.php/competitions-resultats"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _drupal_content_roots(soup: BeautifulSoup) -> list:
    """Sélectionne la/les zone(s) de contenu utile de la page (bloc Drupal principal)."""
    roots = soup.select("#block-frmn-content div[property='schema:text']")
    if not roots:
        roots = soup.select("#block-frmn-content")
    if not roots:
        roots = [soup]
    return roots


def _href_is_document_or_html_page_link(href: str) -> bool:
    """
    True si le href mène à un fichier PDF (souvent sous /sites/default/files/)
    ou vers une page / ressource HTML.

    Détecte :
    - PDF : chemin se terminant par .pdf ;
    - Fichiers HTML : .html ou .htm dans le chemin ;
    - Pages Drupal internes : URLs du type /index.php/slug.
    """
    h = (href or "").strip()
    if not h or h.startswith("#"):
        return False
    low = h.lower()
    if low.startswith("mailto:") or low.startswith("javascript:"):
        return False

    path = unquote(urlparse(h).path.strip() or "")

    if re.search(r"\.pdf(?:\?|$)", path, re.IGNORECASE):
        return True
    if re.search(r"\.html?(?:\?|$)", path, re.IGNORECASE):
        return True

    return bool(re.match(r"^/index\.php/.", path))


def _href_is_pdf_under_sites_default_files(href: str) -> bool:
    """Filtrer strictement les liens PDF hébergés dans des fichiers du site (sous /sites/default/files/)."""
    h = (href or "").strip()
    if not h or h.startswith("#"):
        return False
    path = unquote(urlparse(h).path)
    if "/sites/default/files/" not in path:
        return False
    return bool(re.search(r"\.pdf(?:\?|$)", path, re.IGNORECASE))


def count_li_with_pdf_or_html_links(html: str, *, direct_child_a: bool = False) -> int:
    """
    sert à compter les balises <li> qui contiennent au moins un lien vers :
        un PDF
        ou une page HTML (.html/.htm)
        ou une page interne type /index.php/...
    """
    soup = BeautifulSoup(html, "html.parser")
    n = 0
    for root in _drupal_content_roots(soup):
        for li in root.find_all("li"):
            if direct_child_a:
                match = False
                for child in li.children:
                    if getattr(child, "name", None) == "a" and child.get("href"):
                        if _href_is_document_or_html_page_link(
                            child.get("href", "")
                        ):
                            match = True
                            break
                if match:
                    n += 1
            else:
                for a in li.find_all("a", href=True):
                    if _href_is_document_or_html_page_link(a.get("href", "")):
                        n += 1
                        break
    return n


def count_li_with_sites_default_files_pdf(html: str, *, direct_child_a: bool = False) -> int:
    """Compte uniquement les <li> dont un lien mène à un .pdf sous /sites/default/files/."""
    soup = BeautifulSoup(html, "html.parser")
    n = 0
    for root in _drupal_content_roots(soup):
        for li in root.find_all("li"):
            if direct_child_a:
                hits = []
                for child in li.children:
                    if getattr(child, "name", None) == "a" and child.get("href"):
                        hits.append(child.get("href", ""))
            else:
                hits = [a.get("href", "") for a in li.find_all("a", href=True)]
            if any(_href_is_pdf_under_sites_default_files(z) for z in hits):
                n += 1
    return n


def fetch_html(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def collect_pdf_urls(html: str, page_url: str) -> list[tuple[str, str]]:
    """
    Retourne une liste de (url_absolue, libellé) pour chaque lien PDF
    trouvé dans le corps de page (priorité au bloc principal Drupal).
    """
    soup = BeautifulSoup(html, "html.parser")
    roots = _drupal_content_roots(soup)

    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []

    for root in roots:
        for a in root.select("a[href]"):
            href = (a.get("href") or "").strip()
            if not href or href.startswith("#"):
                continue
            if not re.search(r"\.pdf(?:\?.*)?$", href, re.IGNORECASE):
                continue
            abs_url = urljoin(page_url, href)
            if abs_url in seen:
                continue
            seen.add(abs_url)
            label = " ".join(a.get_text(separator=" ", strip=True).split()) or abs_url
            ordered.append((abs_url, label))

    return ordered


def filename_for_url(url: str, index: int) -> str:
    path = unquote(urlparse(url).path)
    name = Path(path).name
    if not name or name == "/":
        name = f"document_{index:04d}.pdf"
    # Évite les noms vides ou trop bizarres
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name


def download_file(
    session: requests.Session, url: str, dest: Path, chunk: int = 1 << 15
) -> None:
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                if block:
                    f.write(block)
