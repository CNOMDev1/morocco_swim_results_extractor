#!/usr/bin/env python3
"""Structure les JSON extraits de PDF via Gemini en tables normalisees."""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
import requests

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-1.5-flash"

CATEGORIES = ["BENJAMINS", "MINIMES", "CADETS", "JUNIORS", "SENIORS"]
HEADERS = [
    "Place",
    "Nom et prénom",
    "Nation",
    "Naissance",
    "Club",
    "Temps",
    "Points",
    "Temps de passage",
]
HEADER_SET = set(HEADERS)


def collect_json_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def load_text_for_llm(source: dict[str, Any], max_chars: int) -> str:
    full_text = source.get("full_text")
    if isinstance(full_text, str) and full_text.strip():
        return full_text[:max_chars]

    pages = source.get("pages", [])
    if isinstance(pages, list):
        chunks: list[str] = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            text = page.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
        if chunks:
            return "\n\n".join(chunks)[:max_chars]

    return ""


def build_prompt(raw_text: str) -> str:
    return f"""
Tu es un extracteur de resultats de natation.
Convertis ce texte brut en JSON strictement au format:
{{
  "tables": [
    {{
      "category": "BENJAMINS|MINIMES|CADETS|JUNIORS|SENIORS",
      "headers": ["Place","Nom et prénom","Nation","Naissance","Club","Temps","Points","Temps de passage"],
      "rows": [
        {{
          "Place": "...",
          "Nom et prénom": "...",
          "Nation": "...",
          "Naissance": "...",
          "Club": "...",
          "Temps": "...",
          "Points": "...",
          "Temps de passage": "..."
        }}
      ]
    }}
  ]
}}

Regles:
- Retourne uniquement un JSON valide (pas de markdown, pas de commentaires).
- Garde uniquement les categories: BENJAMINS, MINIMES, CADETS, JUNIORS, SENIORS.
- Ignore toutes les autres categories (POUSSINS, AVENIR, etc.).
- Chaque row doit contenir exactement les 8 champs demandes. Si une valeur manque, mets une chaine vide.
- Ne pas inventer de nageurs: extraire uniquement depuis le texte.

Texte a analyser:
\"\"\"
{raw_text}
\"\"\"
""".strip()


def call_gemini(
    api_key: str,
    model: str,
    prompt: str,
    timeout_s: int,
    max_retries: int,
    retry_delay_s: float,
) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "You extract structured competition data from OCR-like text. "
                            "Return only valid JSON."
                        )
                    },
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    }
    url = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"
    attempts = max_retries + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout_s,
            )
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            # Retry only on transient/rate-limit errors.
            if status_code not in (429, 500, 502, 503, 504) or attempt >= attempts:
                raise
            last_error = exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt >= attempts:
                raise
            last_error = exc

        sleep_for = retry_delay_s * (2 ** (attempt - 1))
        time.sleep(sleep_for)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Appel Gemini impossible.")


def strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def normalize_row(row: dict[str, Any]) -> dict[str, str]:
    normalized = {header: "" for header in HEADERS}
    for header in HEADERS:
        value = row.get(header, "")
        normalized[header] = str(value).strip() if value is not None else ""
    return normalized


def normalize_tables(payload: dict[str, Any]) -> dict[str, Any]:
    tables_in = payload.get("tables", [])
    tables_out: list[dict[str, Any]] = []

    if not isinstance(tables_in, list):
        return {"tables": tables_out}

    for table in tables_in:
        if not isinstance(table, dict):
            continue
        category = str(table.get("category", "")).strip().upper()
        if category not in CATEGORIES:
            continue

        rows_out: list[dict[str, str]] = []
        rows_in = table.get("rows", [])
        if isinstance(rows_in, list):
            for row in rows_in:
                if isinstance(row, dict):
                    # Drop malformed rows that do not include any expected values.
                    if not (set(row.keys()) & HEADER_SET):
                        continue
                    rows_out.append(normalize_row(row))

        tables_out.append(
            {
                "category": category,
                "headers": HEADERS,
                "rows": rows_out,
            }
        )

    return {"tables": tables_out}


def process_file(
    source_path: Path,
    output_path: Path,
    api_key: str,
    model: str,
    max_chars: int,
    timeout_s: int,
    max_retries: int,
    retry_delay_s: float,
) -> tuple[bool, str]:
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Lecture JSON impossible: {exc}"

    raw_text = load_text_for_llm(source, max_chars=max_chars)
    if not raw_text.strip():
        return False, "Aucun texte exploitable trouve dans full_text/pages."

    prompt = build_prompt(raw_text)
    try:
        response_text = call_gemini(
            api_key=api_key,
            model=model,
            prompt=prompt,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_delay_s=retry_delay_s,
        )
        parsed = json.loads(strip_markdown_fences(response_text))
    except Exception as exc:
        return False, f"Echec appel/parsing Gemini: {exc}"

    structured = normalize_tables(parsed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(structured, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return True, "OK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lit les JSON de data/json_from_pdfs et produit des JSON structures "
            "avec categories BENJAMINS/MINIMES/CADETS/JUNIORS/SENIORS via Gemini."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data/json_from_pdfs",
        help="Dossier source des JSON extraits de PDF.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/structured_results",
        help="Dossier de sortie des JSON structures.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modele Gemini (defaut: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Cle API Gemini (optionnel si GEMINI_API_KEY est defini).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Parcourir les sous-dossiers d'entree.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=30000,
        help="Nombre max de caracteres envoyes a Gemini par fichier.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Timeout HTTP en secondes pour l'appel Gemini.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Nombre max de fichiers a traiter (0 = tous).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Nombre de retries en cas de 429/erreurs temporaires.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Delai de base (secondes) pour le backoff exponentiel.",
    )
    parser.add_argument(
        "--sleep-between-files",
        type=float,
        default=1.0,
        help="Pause entre deux fichiers pour limiter le rate limit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    api_key = (args.api_key or os.getenv("GEMINI_API_KEY", "")).strip()

    if not api_key:
        print("[erreur] Cle manquante: --api-key ou variable GEMINI_API_KEY.")
        return 1

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[erreur] Dossier introuvable: {input_dir}")
        return 1

    json_files = collect_json_files(input_dir, recursive=args.recursive)
    if args.limit and args.limit > 0:
        json_files = json_files[: args.limit]

    if not json_files:
        print(f"[info] Aucun fichier JSON trouve dans: {input_dir}")
        return 0

    print(f"[info] {len(json_files)} fichier(s) a traiter.")
    ok_count = 0
    err_count = 0

    for index, source_path in enumerate(json_files, start=1):
        output_path = output_dir / source_path.name
        ok, status = process_file(
            source_path=source_path,
            output_path=output_path,
            api_key=api_key,
            model=args.model,
            max_chars=args.max_chars,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
            retry_delay_s=args.retry_delay,
        )
        if ok:
            ok_count += 1
            print(f"[ok] ({index}/{len(json_files)}) {source_path.name} -> {output_path}")
        else:
            err_count += 1
            print(f"[echec] ({index}/{len(json_files)}) {source_path.name}: {status}")

        if args.sleep_between_files > 0 and index < len(json_files):
            time.sleep(args.sleep_between_files)

    print(f"[resume] succes={ok_count} | echecs={err_count}")
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
