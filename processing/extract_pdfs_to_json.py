#!/usr/bin/env python3
"""Extrait le texte de fichiers PDF avec PyMuPDF et l'enregistre en JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIRS = [
    PROJECT_ROOT / "data" / "pdfs_results",
    PROJECT_ROOT / "data" / "pdfs_results_actualites",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "json_from_pdfs"


def extract_pdf_content(pdf_path: Path) -> dict:
    """Extrait le contenu texte page par page d'un PDF."""
    pages: list[dict[str, str | int]] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            pages.append(
                {
                    "page_number": page_index,
                    "text": text.strip(),
                }
            )

    full_text = "\n\n".join(page["text"] for page in pages if page["text"])
    return {
        "source_pdf": str(pdf_path),
        "filename": pdf_path.name,
        "page_count": len(pages),
        "pages": pages,
        "full_text": full_text,
    }


def save_json(content: dict, output_path: Path) -> None:
    """Enregistre un dictionnaire dans un fichier JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(content, json_file, ensure_ascii=False, indent=2)


def collect_pdf_files(input_dir: Path, recursive: bool) -> list[Path]:
    """Récupère la liste des fichiers PDF à traiter pour un dossier."""
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(
        pdf_path
        for pdf_path in input_dir.glob(pattern)
        if pdf_path.is_file() and pdf_path.suffix.lower() == ".pdf"
    )


def build_output_path(
    pdf_path: Path,
    input_dir: Path,
    output_dir: Path,
    preserve_structure: bool,
) -> Path:
    """Construit le chemin de sortie JSON pour un PDF.

    Si ``preserve_structure`` est vrai, on reproduit la structure relative
    du PDF (incluant le nom du dossier d'entrée) sous ``output_dir`` pour
    éviter les collisions entre dossiers (ex: années identiques dans
    ``pdfs_results_actualites``).
    """
    json_name = f"{pdf_path.stem}.json"
    if not preserve_structure:
        return output_dir / json_name

    relative_pdf = pdf_path.relative_to(input_dir)
    return output_dir / input_dir.name / relative_pdf.parent / json_name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extraire le contenu de PDF et générer des fichiers JSON."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=[str(path) for path in DEFAULT_INPUT_DIRS],
        help=(
            "Un ou plusieurs dossiers qui contiennent les PDF "
            "(défaut: <racine_projet>/data/pdfs_results "
            "<racine_projet>/data/pdfs_results_actualites)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Dossier de sortie pour les JSON "
            "(défaut: <racine_projet>/data/json_from_pdfs)."
        ),
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help=(
            "Ne pas parcourir les sous-dossiers (par défaut, le parcours "
            "est récursif pour gérer pdfs_results_actualites/<année>/...)."
        ),
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help=(
            "Écrire tous les JSON à plat dans --output-dir. "
            "Par défaut on reproduit la structure des dossiers d'entrée "
            "pour éviter les collisions de noms."
        ),
    )
    parser.set_defaults(recursive=True)
    args = parser.parse_args()

    input_dirs = [Path(p) for p in args.input_dirs]
    output_dir = Path(args.output_dir)
    preserve_structure = not args.flat_output

    existing_input_dirs = [d for d in input_dirs if d.exists() and d.is_dir()]
    missing_input_dirs = [d for d in input_dirs if d not in existing_input_dirs]

    for missing_dir in missing_input_dirs:
        print(f"[warning] Dossier introuvable ignoré: {missing_dir}")

    if not existing_input_dirs:
        print("[erreur] Aucun dossier d'entrée valide.")
        return 1

    pdf_jobs: list[tuple[Path, Path]] = []
    seen_pdfs: set[Path] = set()
    for input_dir in existing_input_dirs:
        for pdf_path in collect_pdf_files(input_dir, args.recursive):
            resolved = pdf_path.resolve()
            if resolved in seen_pdfs:
                continue
            seen_pdfs.add(resolved)
            pdf_jobs.append((pdf_path, input_dir))

    if not pdf_jobs:
        joined_dirs = ", ".join(str(d) for d in existing_input_dirs)
        print(f"[info] Aucun PDF trouvé dans: {joined_dirs}")
        return 0

    success_count = 0
    error_count = 0

    print(f"[info] {len(pdf_jobs)} PDF trouvé(s).")
    for index, (pdf_path, input_dir) in enumerate(pdf_jobs, start=1):
        try:
            content = extract_pdf_content(pdf_path)
            output_path = build_output_path(
                pdf_path, input_dir, output_dir, preserve_structure
            )
            save_json(content, output_path)
            success_count += 1
            print(f"[ok] ({index}/{len(pdf_jobs)}) {pdf_path.name} -> {output_path}")
        except Exception as exc:
            error_count += 1
            print(f"[échec] {pdf_path}: {exc}")

    print(f"[résumé] Succès: {success_count} | Échecs: {error_count}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
