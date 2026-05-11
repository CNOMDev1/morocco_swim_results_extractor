#!/usr/bin/env python3
"""Structure les JSON OCR de natation avec Gemini Flash 2.0."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import google.generativeai as genai

DEFAULT_MODEL = "gemini-2.5-flash-lite"
DAILY_LIMIT = 1_000_000
DAILY_STOP_THRESHOLD = 950_000
REQUEST_SLEEP_SECONDS = 4
RATE_LIMIT_RETRY_SECONDS = 60
NETWORK_MAX_RETRIES = 3
NETWORK_BACKOFF_BASE_SECONDS = 2

VALID_CATEGORIES = {"BENJAMINS", "MINIMES", "CADETS", "JUNIORS", "SENIORS"}

SYSTEM_PROMPT = """Tu es un extracteur de données de compétitions de natation.
Analyse le texte fourni et retourne UNIQUEMENT un objet JSON valide, sans markdown, sans backticks, sans explication.

Schéma attendu :
{
  "tables": [
    {
      "category": "<BENJAMINS|MINIMES|CADETS|JUNIORS|SENIORS>",
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
- Ne pas inventer de nageurs : extraire uniquement depuis le texte.
- Ne jamais halluciner de valeurs : si un champ est absent ou illisible, mettre "".
- category doit être exactement l'une de : BENJAMINS, MINIMES, CADETS, JUNIORS, SENIORS.
- Retourner uniquement le JSON brut, rien d'autre."""

DEFAULT_PROGRESS = {
    "tokens_used_today": 0,
    "last_reset_date": "",
    "processed_files": [],
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Lit les fichiers JSON issus des PDF et produit des JSON structures "
            "avec Gemini Flash 2.0, en respectant un quota journalier."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(project_root / "data/json_from_pdfs/pdfs_results"),
        help="Dossier contenant les fichiers JSON source.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "json_structures"),
        help="Dossier de sortie des JSON structures.",
    )
    parser.add_argument(
        "--progress-file",
        default=str(script_dir / "progress.json"),
        help="Fichier de progression (quota + fichiers traites).",
    )
    parser.add_argument(
        "--errors-dir",
        default=str(script_dir / "errors"),
        help="Dossier pour stocker les reponses invalides/non parsables.",
    )
    parser.add_argument(
        "--log-file",
        default=str(script_dir / "processing.log"),
        help="Fichier de log de traitement.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modele Gemini (defaut: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--daily-stop-threshold",
        type=int,
        default=DAILY_STOP_THRESHOLD,
        help="Seuil de stop journalier (defaut: 950000).",
    )
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=REQUEST_SLEEP_SECONDS,
        help="Pause entre requetes (defaut: 4s).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Nombre max de fichiers a traiter (0 = tous).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Affiche les details de traitement (tentatives, retries, timings).",
    )
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("structure_json")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_progress(progress_file: Path) -> dict[str, Any]:
    if not progress_file.exists():
        return DEFAULT_PROGRESS.copy()
    try:
        data = json.loads(progress_file.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_PROGRESS.copy()

    if not isinstance(data, dict):
        return DEFAULT_PROGRESS.copy()

    merged = DEFAULT_PROGRESS.copy()
    merged.update(data)
    if not isinstance(merged.get("processed_files"), list):
        merged["processed_files"] = []
    return merged


def save_progress(progress_file: Path, progress: dict[str, Any]) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def maybe_reset_quota(progress: dict[str, Any]) -> None:
    today = date.today().isoformat()
    if progress.get("last_reset_date") != today:
        progress["last_reset_date"] = today
        progress["tokens_used_today"] = 0


def collect_input_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def extract_text_from_source(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    full_text = payload.get("full_text")
    if isinstance(full_text, str) and full_text.strip():
        return full_text

    pages = payload.get("pages")
    if isinstance(pages, list):
        chunks: list[str] = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            page_text = page.get("text")
            if isinstance(page_text, str) and page_text.strip():
                chunks.append(page_text)
        if chunks:
            return "\n\n".join(chunks)
    return ""


def infer_source_file(input_filename: str, source_payload: Any) -> str:
    if isinstance(source_payload, dict):
        for key in ("source_file", "pdf_file", "pdf_filename", "file_name"):
            value = source_payload.get(key)
            if isinstance(value, str) and value.strip():
                clean = value.strip()
                if clean.lower().endswith(".pdf"):
                    return clean
                return f"{Path(clean).stem}.pdf"
    return f"{Path(input_filename).stem}.pdf"


def build_user_prompt(raw_text: str) -> str:
    return (
        "Voici le texte OCR a structurer.\n"
        "Retourne uniquement un JSON valide conforme au schema demande.\n\n"
        f"TEXTE:\n{raw_text}"
    )


def parse_usage_tokens(response: Any) -> int:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
    return total_tokens if total_tokens > 0 else (prompt_tokens + output_tokens)


def debug_print(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[debug] {message}")


def is_blocking_quota_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "quota exceeded" in lowered
        and "limit: 0" in lowered
        and "free_tier" in lowered
    )


def generate_with_retries(
    model: genai.GenerativeModel,
    prompt: str,
    debug: bool,
) -> tuple[str, int]:
    last_exception: Exception | None = None
    for attempt in range(1, NETWORK_MAX_RETRIES + 1):
        try:
            started_at = time.perf_counter()
            debug_print(debug, f"Gemini attempt {attempt}/{NETWORK_MAX_RETRIES} started")
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
            )
            elapsed = time.perf_counter() - started_at
            response_text = getattr(response, "text", "") or ""
            used_tokens = parse_usage_tokens(response)
            debug_print(
                debug,
                (
                    f"Gemini attempt {attempt} success in {elapsed:.2f}s | "
                    f"response_chars={len(response_text)} | used_tokens={used_tokens}"
                ),
            )
            return response_text, used_tokens
        except Exception as exc:
            last_exception = exc
            message = str(exc)
            transient_markers = ("429", "500", "502", "503", "504", "timeout", "connection")
            lower_message = message.lower()
            debug_print(
                debug,
                f"Gemini attempt {attempt} failed: {type(exc).__name__}: {exc}",
            )

            if "429" in message:
                debug_print(debug, f"Rate limit detecte, attente {RATE_LIMIT_RETRY_SECONDS}s")
                time.sleep(RATE_LIMIT_RETRY_SECONDS)
                continue

            is_transient = any(marker in lower_message for marker in transient_markers)
            if is_transient and attempt < NETWORK_MAX_RETRIES:
                backoff = NETWORK_BACKOFF_BASE_SECONDS ** attempt
                debug_print(debug, f"Erreur transitoire, retry dans {backoff}s")
                time.sleep(backoff)
                continue
            raise RuntimeError(f"Gemini error (attempt {attempt}/{NETWORK_MAX_RETRIES}): {exc}") from exc

    if last_exception is not None:
        raise RuntimeError(
            "Echec d'appel Gemini apres retries: "
            f"{type(last_exception).__name__}: {last_exception}"
        ) from last_exception
    raise RuntimeError("Echec d'appel Gemini apres retries.")


def normalize_headers(headers_value: Any, first_row: Any) -> list[str]:
    headers: list[str] = []
    if isinstance(headers_value, list):
        headers = [str(item).strip() for item in headers_value if str(item).strip()]

    if not headers and isinstance(first_row, dict):
        headers = [str(k) for k in first_row.keys()]
    return headers


def normalize_rows(rows_value: Any, headers: list[str]) -> list[dict[str, str]]:
    rows_out: list[dict[str, str]] = []
    if not isinstance(rows_value, list):
        return rows_out

    for row in rows_value:
        if not isinstance(row, dict):
            continue
        normalized_row: dict[str, str] = {}
        for header in headers:
            value = row.get(header, "")
            normalized_row[header] = "" if value is None else str(value).strip()
        rows_out.append(normalized_row)
    return rows_out


def normalize_structured_output(raw: Any, source_file: str) -> dict[str, Any]:
    tables_out: list[dict[str, Any]] = []
    tables_in = raw.get("tables", []) if isinstance(raw, dict) else []
    if not isinstance(tables_in, list):
        tables_in = []

    for table in tables_in:
        if not isinstance(table, dict):
            continue
        category = str(table.get("category", "")).strip().upper()
        if category not in VALID_CATEGORIES:
            continue

        rows_in = table.get("rows", [])
        first_row = rows_in[0] if isinstance(rows_in, list) and rows_in else None
        headers = normalize_headers(table.get("headers"), first_row)
        if not headers:
            continue

        rows = normalize_rows(rows_in, headers)
        tables_out.append(
            {
                "category": category,
                "headers": headers,
                "rows": rows,
            }
        )

    return {
        "source_file": source_file,
        "tables": tables_out,
    }


def save_invalid_response(errors_dir: Path, input_name: str, raw_response: str) -> None:
    errors_dir.mkdir(parents=True, exist_ok=True)
    error_path = errors_dir / f"{Path(input_name).stem}_invalid_response.txt"
    error_path.write_text(raw_response or "", encoding="utf-8")


def process_one_file(
    file_path: Path,
    output_dir: Path,
    errors_dir: Path,
    model: genai.GenerativeModel,
    debug: bool,
) -> tuple[str, int]:
    source_raw = file_path.read_text(encoding="utf-8")
    source_payload: Any
    try:
        source_payload = json.loads(source_raw)
    except json.JSONDecodeError as exc:
        return f"JSON source invalide: {exc}", 0

    source_text = extract_text_from_source(source_payload)
    if not source_text.strip():
        return "Aucun texte exploitable (full_text/pages).", 0
    debug_print(debug, f"{file_path.name}: texte extrait ({len(source_text)} caracteres)")

    prompt = build_user_prompt(source_text)
    response_text, used_tokens = generate_with_retries(model, prompt, debug=debug)

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        save_invalid_response(errors_dir, file_path.name, response_text)
        return "Reponse Gemini non-JSON, sauvegardee dans errors/.", used_tokens

    output_payload = normalize_structured_output(
        parsed, infer_source_file(file_path.name, source_payload)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_path.name
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    debug_print(debug, f"{file_path.name}: fichier structure sauvegarde -> {output_path}")
    return "OK", used_tokens


def main() -> int:
    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[erreur] Variable GEMINI_API_KEY manquante.")
        return 1

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    progress_file = Path(args.progress_file)
    errors_dir = Path(args.errors_dir)
    logger = setup_logger(Path(args.log_file))

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[erreur] Dossier introuvable: {input_dir}")
        return 1

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=args.model,
        system_instruction=SYSTEM_PROMPT,
    )

    progress = load_progress(progress_file)
    maybe_reset_quota(progress)

    processed_files = set(str(name) for name in progress.get("processed_files", []))
    tokens_used_today = int(progress.get("tokens_used_today", 0) or 0)

    files = collect_input_files(input_dir)
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        print("[info] Aucun fichier JSON a traiter.")
        return 0

    print(f"[info] Fichiers detectes: {len(files)}")
    print(f"[info] Tokens utilises aujourd'hui: {tokens_used_today}/{DAILY_LIMIT}")
    debug_print(
        args.debug,
        (
            f"config model={args.model} stop_threshold={args.daily_stop_threshold} "
            f"sleep={args.sleep_between_requests}s max_files={args.max_files or 'all'}"
        ),
    )

    for file_path in files:
        if file_path.name in processed_files:
            continue

        if tokens_used_today >= int(args.daily_stop_threshold):
            print(
                f"[stop] Seuil journalier atteint: {tokens_used_today} >= "
                f"{int(args.daily_stop_threshold)}"
            )
            break

        status = "ERREUR"
        used_tokens = 0
        print(f"[processing] {file_path.name}")
        try:
            status, used_tokens = process_one_file(
                file_path=file_path,
                output_dir=output_dir,
                errors_dir=errors_dir,
                model=model,
                debug=args.debug,
            )
        except Exception as exc:
            status = f"ERREUR appel Gemini: {exc}"
            used_tokens = 0

        tokens_used_today += used_tokens
        progress["tokens_used_today"] = tokens_used_today

        if status == "OK":
            processed_files.add(file_path.name)
            progress["processed_files"] = sorted(processed_files)
            logger.info("%s | OK | tokens=%s", file_path.name, used_tokens)
            print(f"[ok] {file_path.name} | +{used_tokens} tokens")
        else:
            logger.error("%s | %s | tokens=%s", file_path.name, status, used_tokens)
            print(f"[echec] {file_path.name} | {status} | +{used_tokens} tokens")
            if is_blocking_quota_error(status):
                print(
                    "[stop] Quota Gemini bloque (free tier limit=0). "
                    "Arret automatique pour eviter des retries inutiles."
                )
                save_progress(progress_file, progress)
                break

        save_progress(progress_file, progress)
        time.sleep(max(0.0, float(args.sleep_between_requests)))

    save_progress(progress_file, progress)
    print(
        "[resume] "
        f"tokens_used_today={progress.get('tokens_used_today', 0)} | "
        f"processed_files={len(progress.get('processed_files', []))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
