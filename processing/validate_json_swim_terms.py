#!/usr/bin/env python3
"""Vérifie que les JSON extraits des PDFs contiennent les nages et distances attendues."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON_DIR = PROJECT_ROOT / "data" / "json_from_pdfs" / "pdfs_results"

# Catégories de nage : libellé affiché → motifs recherchés (insensible à la casse)
STROKE_CATEGORIES: dict[str, list[str]] = {
    "Nage libre / Crawl": ["nage libre", "crawl"],
    "Dos crawlé": ["dos crawl", "dos crawlé", r"\bdos\b"],
    "Brasse": ["brasse"],
    "Papillon": ["papillon"],
    "4 nages": ["4 nages", "4 nage"],
    "Relais 4 nages": ["relais 4 nages", "relais 4 nage", r"\b4\s*x\b.*nages"],
}

DISTANCES_M = ["50", "100", "200", "400", "800", "1500"]


def extract_text(data: dict) -> str:
    """Récupère tout le texte exploitable d'un fichier JSON."""
    if isinstance(data.get("full_text"), str) and data["full_text"].strip():
        return data["full_text"]

    parts: list[str] = []
    for page in data.get("pages", []):
        if isinstance(page, dict) and isinstance(page.get("text"), str):
            parts.append(page["text"])
    return "\n\n".join(parts)


def normalize(text: str) -> str:
    return text.lower().replace("\u00a0", " ")


def find_strokes(text: str) -> dict[str, bool]:
    """Indique quelles catégories de nage sont présentes dans le texte."""
    normalized = normalize(text)
    found: dict[str, bool] = {}

    for label, patterns in STROKE_CATEGORIES.items():
        matched = False
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                matched = True
                break
        found[label] = matched

    return found


def find_distances(text: str) -> dict[str, bool]:
    """Indique quelles distances (en mètres) apparaissent dans le texte."""
    normalized = normalize(text)
    found: dict[str, bool] = {}

    for distance in DISTANCES_M:
        pattern = rf"(?:^|[\s,.;]){re.escape(distance)}\s*m(?:\b|[\s,.;]|$)"
        found[f"{distance} m"] = bool(re.search(pattern, normalized))

    return found


def validate_file(json_path: Path) -> dict:
    """Analyse un fichier JSON et retourne le rapport de validation."""
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    text = extract_text(data)
    strokes = find_strokes(text)
    distances = find_distances(text)

    missing_strokes = [label for label, ok in strokes.items() if not ok]
    missing_distances = [label for label, ok in distances.items() if not ok]

    return {
        "file": json_path.name,
        "text_length": len(text),
        "strokes": strokes,
        "distances": distances,
        "missing_strokes": missing_strokes,
        "missing_distances": missing_distances,
        "ok": not missing_strokes and not missing_distances,
    }


def print_report(results: list[dict], verbose: bool) -> None:
    total = len(results)
    complete = sum(1 for r in results if r["ok"])
    empty_text = sum(1 for r in results if r["text_length"] == 0)

    print(f"\n{'=' * 70}")
    print(f"Fichiers analysés : {total}")
    print(f"Texte vide        : {empty_text}")
    print(f"Tous termes OK    : {complete}")
    print(f"{'=' * 70}\n")

    for result in sorted(results, key=lambda r: r["file"].lower()):
        status = "OK" if result["ok"] else "MANQUANT"
        print(f"[{status}] {result['file']}")

        if verbose or not result["ok"]:
            present_strokes = [k for k, v in result["strokes"].items() if v]
            present_distances = [k for k, v in result["distances"].items() if v]

            if present_strokes:
                print(f"  Nages trouvées    : {', '.join(present_strokes)}")
            if present_distances:
                print(f"  Distances trouvées: {', '.join(present_distances)}")
            if result["missing_strokes"]:
                print(f"  Nages manquantes  : {', '.join(result['missing_strokes'])}")
            if result["missing_distances"]:
                print(f"  Distances manquantes: {', '.join(result['missing_distances'])}")
            if result["text_length"] == 0:
                print("  ⚠ Aucun texte extrait (full_text / pages vides)")
            print()


def aggregate_stats(results: list[dict]) -> None:
    """Résumé global : fréquence de chaque nage et distance dans le corpus."""
    stroke_counts = {label: 0 for label in STROKE_CATEGORIES}
    distance_counts = {f"{d} m": 0 for d in DISTANCES_M}

    for result in results:
        for label, ok in result["strokes"].items():
            if ok:
                stroke_counts[label] += 1
        for label, ok in result["distances"].items():
            if ok:
                distance_counts[label] += 1

    total = len(results) or 1
    print("\n--- Statistiques globales ---")
    print("Nages (fichiers contenant le terme) :")
    for label, count in stroke_counts.items():
        print(f"  {label:25} {count:4}/{total} ({100 * count / total:.1f}%)")

    print("\nDistances :")
    for label, count in distance_counts.items():
        print(f"  {label:10} {count:4}/{total} ({100 * count / total:.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Vérifie la présence des nages et distances dans les JSON extraits des PDFs."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_JSON_DIR,
        help=f"Dossier des fichiers JSON (défaut: {DEFAULT_JSON_DIR})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Afficher le détail pour tous les fichiers, pas seulement ceux incomplets",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Afficher les statistiques globales à la fin",
    )
    args = parser.parse_args()

    json_dir: Path = args.dir
    if not json_dir.is_dir():
        print(f"Erreur : dossier introuvable — {json_dir}", file=sys.stderr)
        return 1

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"Aucun fichier .json dans {json_dir}", file=sys.stderr)
        return 1

    results: list[dict] = []
    errors: list[str] = []

    for json_path in json_files:
        try:
            results.append(validate_file(json_path))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"{json_path.name}: {exc}")

    print_report(results, verbose=args.verbose)

    if args.stats:
        aggregate_stats(results)

    if errors:
        print("\nErreurs de lecture :")
        for err in errors:
            print(f"  - {err}")

    incomplete = sum(1 for r in results if not r["ok"])
    return 1 if incomplete or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
