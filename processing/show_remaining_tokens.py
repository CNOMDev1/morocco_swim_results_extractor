#!/usr/bin/env python3
"""Affiche les tokens utilises et restants pour Gemini."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.0-flash"
# Valeur par defaut configurable via --max-context-tokens.
DEFAULT_MAX_CONTEXT_TOKENS = 1_000_000


def count_tokens_gemini(api_key: str, model: str, text: str, timeout_s: int) -> int:
    url = f"{GEMINI_API_BASE}/{model}:countTokens?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": text}]}],
    }
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    return int(data.get("totalTokens", 0))


def collect_json_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    if isinstance(parsed, dict):
        full_text = parsed.get("full_text")
        if isinstance(full_text, str) and full_text.strip():
            return full_text

    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compte les tokens d'un texte/fichier avec Gemini et affiche "
            "combien il reste sur un budget de contexte."
        )
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Cle API Gemini (optionnel si GEMINI_API_KEY est defini).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modele Gemini (defaut: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--file",
        default="",
        help="Fichier a analyser (txt/json).",
    )
    parser.add_argument(
        "--dir",
        default="",
        help="Dossier de fichiers JSON a analyser (un resultat par fichier + total).",
    )
    parser.add_argument(
        "--glob",
        default="*.json",
        help="Pattern de recherche quand --dir est fourni (defaut: *.json).",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        help=(
            "Budget de tokens de contexte pour calculer le restant "
            f"(defaut: {DEFAULT_MAX_CONTEXT_TOKENS})."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout HTTP en secondes.",
    )
    parser.add_argument(
        "--pdfs-results-dir",
        default="",
        help=(
            "Dossier json_from_pdfs/pdfs_results a analyser "
            "(mode comparaison 2 dossiers)."
        ),
    )
    parser.add_argument(
        "--pdfs-results-actualites-dir",
        default="",
        help=(
            "Dossier json_from_pdfs/pdfs_results_actualites a analyser "
            "(mode comparaison 2 dossiers)."
        ),
    )
    return parser.parse_args()


def analyze_directory(
    api_key: str,
    model: str,
    dir_path: Path,
    glob_pattern: str,
    timeout: int,
) -> tuple[int, int]:
    files = sorted(p for p in dir_path.glob(glob_pattern) if p.is_file())
    if not files:
        print(f"[info] Aucun fichier trouve dans {dir_path} avec le pattern {glob_pattern}")
        return 0, 0

    total_used = 0
    print(f"[info] {len(files)} fichier(s) trouves dans {dir_path}.")
    for idx, file_path in enumerate(files, start=1):
        text = collect_json_text(file_path)
        used = count_tokens_gemini(api_key, model, text, timeout)
        total_used += used
        print(f"[{idx}/{len(files)}] {file_path.name}: utilises={used}")
    return len(files), total_used


def main() -> int:
    args = parse_args()
    api_key = (args.api_key or os.getenv("GEMINI_API_KEY", "")).strip()

    if not api_key:
        print("[erreur] Cle manquante: --api-key ou variable GEMINI_API_KEY.")
        return 1

    pair_mode = bool(args.pdfs_results_dir) or bool(args.pdfs_results_actualites_dir)
    single_mode_selected = int(bool(args.file)) + int(bool(args.dir))
    if pair_mode and single_mode_selected:
        print(
            "[erreur] Utilise soit --file/--dir, soit "
            "--pdfs-results-dir + --pdfs-results-actualites-dir."
        )
        return 1

    if pair_mode:
        if not (args.pdfs_results_dir and args.pdfs_results_actualites_dir):
            print(
                "[erreur] En mode 2 dossiers, fournis --pdfs-results-dir ET "
                "--pdfs-results-actualites-dir."
            )
            return 1

        dir_a = Path(args.pdfs_results_dir)
        dir_b = Path(args.pdfs_results_actualites_dir)
        if not dir_a.exists() or not dir_a.is_dir():
            print(f"[erreur] Dossier introuvable: {dir_a}")
            return 1
        if not dir_b.exists() or not dir_b.is_dir():
            print(f"[erreur] Dossier introuvable: {dir_b}")
            return 1

        print(f"Modele: {args.model}")
        count_a, used_a = analyze_directory(
            api_key=api_key,
            model=args.model,
            dir_path=dir_a,
            glob_pattern=args.glob,
            timeout=args.timeout,
        )
        count_b, used_b = analyze_directory(
            api_key=api_key,
            model=args.model,
            dir_path=dir_b,
            glob_pattern=args.glob,
            timeout=args.timeout,
        )

        remaining_a = args.max_context_tokens - used_a
        remaining_b = args.max_context_tokens - used_b
        grand_total = used_a + used_b
        grand_remaining = args.max_context_tokens - grand_total

        print("\n--- Resume par dossier ---")
        print(f"pdfs_results: fichiers={count_a} | tokens utilises={used_a} | restants={remaining_a}")
        print(
            "pdfs_results_actualites: "
            f"fichiers={count_b} | tokens utilises={used_b} | restants={remaining_b}"
        )
        print("\n--- Total global ---")
        print(f"Tokens utilises (2 dossiers): {grand_total}")
        print(f"Tokens restants (vs budget unique): {grand_remaining}")
        return 0

    if single_mode_selected != 1:
        print(
            "[erreur] Fournis soit --file, soit --dir, "
            "ou bien les 2 dossiers en mode comparaison."
        )
        return 1

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists() or not file_path.is_file():
            print(f"[erreur] Fichier introuvable: {file_path}")
            return 1

        text = collect_json_text(file_path)
        used = count_tokens_gemini(api_key, args.model, text, args.timeout)
        remaining = args.max_context_tokens - used
        print(f"Fichier: {file_path}")
        print(f"Modele: {args.model}")
        print(f"Tokens utilises: {used}")
        print(f"Tokens restants: {remaining}")
        return 0

    dir_path = Path(args.dir)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"[erreur] Dossier introuvable: {dir_path}")
        return 1

    _, total_used = analyze_directory(
        api_key=api_key,
        model=args.model,
        dir_path=dir_path,
        glob_pattern=args.glob,
        timeout=args.timeout,
    )

    total_remaining = args.max_context_tokens - total_used
    print("\n--- Total ---")
    print(f"Tokens utilises (somme): {total_used}")
    print(f"Tokens restants (vs budget): {total_remaining}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
