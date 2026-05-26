#!/usr/bin/env python3
"""Extrait les tableaux de résultats d'un PDF via LlamaExtract (LlamaCloud)."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from llama_cloud import LlamaCloud
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "pdfs_results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "json_from_pdfs_llamaextract"

EXTRACTORS_DIR = Path(__file__).resolve().parent
ENV_PATH = EXTRACTORS_DIR / ".env"

VALID_CATEGORIES = {"", "BENJAMINS", "MINIMES", "CADETS", "JUNIORS", "SENIORS"}
NAMED_CATEGORIES = {"BENJAMINS", "MINIMES", "CADETS", "JUNIORS", "SENIORS"}

DEFAULT_HEADERS = [
    "Place",
    "Nom et prénom",
    "Nation",
    "Naissance",
    "Club",
    "Temps",
    "Points",
    "Temps de passage",
]

Category = Literal["", "BENJAMINS", "MINIMES", "CADETS", "JUNIORS", "SENIORS"]

SYSTEM_PROMPT = """Tu es un extracteur de données de compétitions de natation marocaine (FRMN).
Analyse le document PDF et extrais UNIQUEMENT les tableaux de résultats présents.

Schéma attendu :
{
  "tables": [
    {
      "category": "<vide|BENJAMINS|MINIMES|CADETS|JUNIORS|SENIORS>",
      "headers": ["Place", "Nom et prénom", "Nation", "Naissance", "Club", "Temps", "Points", "Temps de passage"],
      "rows": [
        {
          "Place": "",
          "Nom et prénom": "",
          "Nation": "",
          "Naissance": "",
          "Club": "",
          "Temps": "",
          "Points": "",
          "Temps de passage": ""
        }
      ]
    }
  ]
}

Règles absolues :
- Ne pas inventer de nageurs : extraire UNIQUEMENT ce qui est visible dans le PDF.
- Ne jamais halluciner : si un champ est absent ou illisible, mettre "".
- category = "" pour les classements « TOUTES CATEGORIES » / « Finale A » sans sous-catégorie.
- category = BENJAMINS, MINIMES, CADETS, JUNIORS ou SENIORS lorsque le bloc est sous ce titre.
- Une table par combinaison (épreuve + catégorie) : plusieurs tables si plusieurs catégories.
- headers doit toujours être exactement la liste des 8 colonnes ci-dessus.
- Chaque ligne de nageur = un objet dans rows avec les 8 clés exactes."""


class ResultRow(BaseModel):
    """Une ligne de résultat (colonnes fixes)."""
    model_config = ConfigDict(populate_by_name=True)
    Place: str = Field(default="", description="Classement: 1., 2., N.C., etc.")
    nom_prenom: str = Field(
        default="",
        alias="Nom et prénom",
        description="Nom et prénom du nageur.",
    )
    Nation: str = Field(default="", description="Code pays, ex: MAR, ALG, TUN.")
    Naissance: str = Field(default="", description="Année de naissance.")
    Club: str = Field(default="", description="Code ou nom du club.")
    Temps: str = Field(default="", description="Temps final de l'épreuve.")
    Points: str = Field(default="", description="Points FINA ou barème local.")
    temps_passage: str = Field(
        default="",
        alias="Temps de passage",
        description="Splits / temps de passage pour relais ou fond.",
    )


class ResultsTable(BaseModel):
    """Tableau de résultats pour une catégorie d'âge."""
    category: Category = Field(
        default="",
        description=(
            "Catégorie d'âge: BENJAMINS, MINIMES, CADETS, JUNIORS, SENIORS. "
            "Laisser vide (\"\") pour TOUTES CATEGORIES / classement général."
        ),
    )
    headers: list[str] = Field(
        default_factory=lambda: list(DEFAULT_HEADERS),
        description="En-têtes de colonnes (liste fixe de 8 colonnes).",
    )
    rows: list[ResultRow] = Field(
        default_factory=list,
        description="Lignes de résultats, une par nageur ou relais.",
    )


class SwimResultsDocument(BaseModel):
    """Document complet : tous les tableaux extraits du PDF."""

    tables: list[ResultsTable] = Field(
        description="Liste des tableaux de résultats trouvés dans le document."
    )


def load_api_key() -> str:
    """Charge LLAMA_CLOUD_API_KEY."""
    if ENV_PATH.is_file():
        load_dotenv(ENV_PATH, override=False)

    api_key = os.getenv("LLAMA_CLOUD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "LLAMA_CLOUD_API_KEY manquante. "
            "Ajoutez-la dans extractors/.env ou exportez-la: "
        )
    return api_key


def build_configuration(*, tier: str) -> dict:
    """Construit la configuration passé à l'API LlamaExtract (schéma JSON + consignes)."""
    return {
        "data_schema": SwimResultsDocument.model_json_schema(by_alias=True),
        "extraction_target": "per_doc",
        "tier": tier,
        "system_prompt": SYSTEM_PROMPT,
    }


def normalize_headers(raw_headers: Any, first_row: Any) -> list[str]:
    """Récupère une liste fiable de noms de colonnes à partir de ce que LlamaExtract renvoie"""
    if isinstance(raw_headers, list):
        headers = [str(header).strip() for header in raw_headers if str(header).strip()]
        if headers:
            return headers
    if isinstance(first_row, dict):
        return list(first_row.keys())
    return list(DEFAULT_HEADERS)


def normalize_rows(raw_rows: Any, headers: list[str]) -> list[dict[str, str]]:
    if not isinstance(raw_rows, list):
        return []
    rows_out: list[dict[str, str]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        rows_out.append(
            {
                header: ("" if row.get(header) is None else str(row.get(header, "")).strip())
                for header in headers
            }
        )
    return rows_out


def normalize_category(raw_category: Any) -> str:
    category = str(raw_category or "").strip().upper()
    if category in NAMED_CATEGORIES:
        return category
    return ""


def normalize_tables(raw: Any) -> list[dict[str, Any]]:
    """Valide et normalise les tableaux extraits par LlamaExtract."""
    tables_in = raw.get("tables", []) if isinstance(raw, dict) else []
    tables_out: list[dict[str, Any]] = []

    for table in (tables_in if isinstance(tables_in, list) else []):
        if not isinstance(table, dict):
            continue

        category = normalize_category(table.get("category"))
        if category not in VALID_CATEGORIES:
            continue

        rows_in = table.get("rows", [])
        first_row = rows_in[0] if isinstance(rows_in, list) and rows_in else None
        headers = normalize_headers(table.get("headers"), first_row)
        if not headers:
            headers = list(DEFAULT_HEADERS)

        rows = normalize_rows(rows_in, headers)
        if not rows:
            continue

        tables_out.append({"category": category, "headers": headers, "rows": rows})

    return tables_out


def build_output_payload(pdf_path: Path, tables: list[dict[str, Any]]) -> dict[str, Any]:
    """Format final aligné sur json_structures / html_results."""
    return {
        "source_file": pdf_path.name,
        "tables": tables,
    }


def collect_pdf_files(input_dir: Path) -> list[Path]:
    """Liste triée des PDF à la racine de input_dir."""
    if not input_dir.is_dir():
        return []
    return sorted(
        pdf_path
        for pdf_path in input_dir.glob("*.pdf")
        if pdf_path.is_file()
    )


def output_path_for_pdf(pdf_path: Path, output_dir: Path) -> Path:
    """Chemin JSON de sortie pour un PDF donné."""
    return output_dir / f"{pdf_path.stem}.json"


def is_already_processed(output_path: Path) -> bool:
    """True si un JSON valide avec au moins une table existe déjà."""
    if not output_path.is_file():
        return False
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    tables = payload.get("tables") if isinstance(payload, dict) else None
    return isinstance(tables, list) and len(tables) > 0


def extract_pdf_with_llama(
    pdf_path: Path,
    client: LlamaCloud,
    *,
    tier: str = "agentic",
    verbose: bool = False,
) -> dict[str, Any]:
    """Upload le PDF, lance LlamaExtract et retourne les tableaux structurés."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF introuvable: {pdf_path}")

    print(f"[info] Upload: {pdf_path.name}")
    uploaded = client.files.create(file=str(pdf_path), purpose="extract")

    print(f"[info] Extraction LlamaExtract (tier={tier}, format=tables)")
    job = client.extract.run(
        file_input=uploaded.id,
        configuration=build_configuration(tier=tier),
        verbose=verbose,
    )

    if job.status != "COMPLETED":
        error = job.error_message or f"Statut inattendu: {job.status}"
        raise RuntimeError(f"Extraction échouée: {error}")

    tables = normalize_tables(job.extract_result)
    if not tables:
        raise RuntimeError("Aucun tableau extrait (extract_result vide ou invalide).")

    return build_output_payload(pdf_path, tables)


def save_json(payload: dict, output_path: Path) -> None:
    """Enregistre le résultat en JSON UTF-8."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)


def count_rows(tables: list[dict[str, Any]]) -> int:
    return sum(len(table.get("rows", [])) for table in tables)


def process_one_pdf(
    pdf_path: Path,
    output_path: Path,
    client: LlamaCloud,
    *,
    tier: str,
    verbose: bool,
) -> tuple[str, int, int]:
    """Traite un PDF. Retourne (status, nb_tables, nb_lignes)."""
    payload = extract_pdf_with_llama(
        pdf_path,
        client,
        tier=tier,
        verbose=verbose,
    )
    save_json(payload, output_path)
    tables = payload["tables"]
    return "ok", len(tables), count_rows(tables)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extraire les tableaux de résultats des PDF via LlamaExtract "
            "(format source_file + tables)."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Dossier des PDF (défaut: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Dossier de sortie JSON (défaut: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--pdf",
        default="",
        help="Traiter un seul PDF (sinon tous les PDF de --input-dir).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Fichier JSON de sortie (uniquement avec --pdf).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ré-extraire même si le JSON existe déjà.",
    )
    parser.add_argument(
        "--tier",
        choices=("cost_effective", "agentic"),
        default="agentic",
        help="Niveau LlamaExtract (défaut: agentic, recommandé pour les tableaux).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affiche la progression du polling LlamaExtract.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        pdf_jobs = [(Path(args.pdf), Path(args.output) if args.output else None)]
    else:
        if not input_dir.is_dir():
            print(f"[erreur] Dossier introuvable: {input_dir}")
            return 1
        pdf_jobs = [(pdf_path, None) for pdf_path in collect_pdf_files(input_dir)]

    if not pdf_jobs:
        print(f"[info] Aucun PDF trouvé dans: {input_dir}")
        return 0

    try:
        client = LlamaCloud(api_key=load_api_key())
    except RuntimeError as exc:
        print(f"[erreur] {exc}")
        return 1

    skipped = 0
    success = 0
    failed = 0

    print(f"[info] {len(pdf_jobs)} PDF à examiner.")
    for index, (pdf_path, custom_output) in enumerate(pdf_jobs, start=1):
        if not pdf_path.is_file():
            print(f"[échec] ({index}/{len(pdf_jobs)}) PDF introuvable: {pdf_path}")
            failed += 1
            continue

        output_path = custom_output or output_path_for_pdf(pdf_path, output_dir)

        if not args.force and is_already_processed(output_path):
            skipped += 1
            print(f"[skip] ({index}/{len(pdf_jobs)}) Déjà traité: {pdf_path.name}")
            continue

        print(f"[traitement] ({index}/{len(pdf_jobs)}) {pdf_path.name}")
        try:
            status, table_count, row_count = process_one_pdf(
                pdf_path,
                output_path,
                client,
                tier=args.tier,
                verbose=args.verbose,
            )
            success += 1
            print(
                f"[{status}] {pdf_path.name} -> {output_path} "
                f"({table_count} table(s), {row_count} ligne(s))"
            )
        except Exception as exc:
            failed += 1
            print(f"[échec] ({index}/{len(pdf_jobs)}) {pdf_path.name}: {exc}")

    print(
        f"[résumé] Succès: {success} | Ignorés: {skipped} | Échecs: {failed} "
        f"| Total: {len(pdf_jobs)}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())