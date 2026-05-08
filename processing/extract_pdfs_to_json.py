#!/usr/bin/env python3
"""Extrait le texte de fichiers PDF avec PyMuPDF et l'enregistre en JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extraire le contenu de PDF et générer des fichiers JSON."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=["data/pdfs_results", "data/pdfs_results_actualites"],
        help=(
            "Un ou plusieurs dossiers qui contiennent les PDF "
            "(défaut: data/pdfs_results data/pdfs_results_actualites)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/json_from_pdfs",
        help="Dossier de sortie pour les JSON (défaut: data/json_from_pdfs).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Parcourir les sous-dossiers du dossier d'entrée.",
    )
    args = parser.parse_args()

    input_dirs = [Path(p) for p in args.input_dirs]
    output_dir = Path(args.output_dir)

    existing_input_dirs = [d for d in input_dirs if d.exists() and d.is_dir()]
    missing_input_dirs = [d for d in input_dirs if d not in existing_input_dirs]

    for missing_dir in missing_input_dirs:
        print(f"[warning] Dossier introuvable ignoré: {missing_dir}")

    if not existing_input_dirs:
        print("[erreur] Aucun dossier d'entrée valide.")
        return 1

    pdf_files: list[Path] = []
    for input_dir in existing_input_dirs:
        pdf_files.extend(collect_pdf_files(input_dir, args.recursive))
    pdf_files = sorted(set(pdf_files))

    if not pdf_files:
        joined_dirs = ", ".join(str(d) for d in existing_input_dirs)
        print(f"[info] Aucun PDF trouvé dans: {joined_dirs}")
        return 0

    success_count = 0
    error_count = 0

    print(f"[info] {len(pdf_files)} PDF trouvé(s).")
    for index, pdf_path in enumerate(pdf_files, start=1):
        try:
            content = extract_pdf_content(pdf_path)
            json_name = f"{pdf_path.stem}.json"
            output_path = output_dir / json_name
            save_json(content, output_path)
            success_count += 1
            print(f"[ok] ({index}/{len(pdf_files)}) {pdf_path.name} -> {output_path}")
        except Exception as exc:
            error_count += 1
            print(f"[échec] {pdf_path}: {exc}")

    print(f"[résumé] Succès: {success_count} | Échecs: {error_count}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
