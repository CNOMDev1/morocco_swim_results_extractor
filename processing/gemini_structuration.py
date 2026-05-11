#!/usr/bin/env python3
"""Structure les JSON OCR de natation avec Gemini 2.5 Flash.

Usage:
    pip install google-genai
    export GEMINI_API_KEY="AIza..."
    python structure_json.py
    python structure_json.py --input-dir ./mes_json --max-files 10 --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

# ── Nouveau SDK officiel Google (remplace google.generativeai déprécié) ────────
from google import genai
from google.genai import types

# ── Modèle & limites Gemini 2.5 Flash (tier gratuit) ──────────────────────────
DEFAULT_MODEL = "gemini-2.5-flash"

# Gemini 2.5 Flash gratuit : 10 RPM, 250 RPD, 250 000 TPM
DAILY_REQUEST_LIMIT     = 250
DAILY_REQUEST_THRESHOLD = 240      # stop avec marge de sécurité
REQUEST_SLEEP_SECONDS   = 6.5      # 60s / 10 RPM + marge

RATE_LIMIT_RETRY_SECONDS = 65
NETWORK_MAX_RETRIES      = 3
NETWORK_BACKOFF_BASE     = 2

# ── Catégories valides ─────────────────────────────────────────────────────────
VALID_CATEGORIES = {"BENJAMINS", "MINIMES", "CADETS", "JUNIORS", "SENIORS"}

# ── Prompt système ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es un extracteur de données de compétitions de natation marocaine.
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
- Ne pas inventer de nageurs : extraire UNIQUEMENT ce qui est présent dans le texte.
- Ne jamais halluciner de valeurs : si un champ est absent ou illisible, mettre "".
- category doit être exactement l'une de : BENJAMINS, MINIMES, CADETS, JUNIORS, SENIORS.
- Un fichier peut contenir plusieurs tables (une par catégorie trouvée dans le texte).
- Retourner uniquement le JSON brut, rien d'autre."""

# ── Progression par défaut ─────────────────────────────────────────────────────
DEFAULT_PROGRESS: dict[str, Any] = {
    "requests_today":  0,
    "last_reset_date": "",
    "processed_files": [],
}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Structure les JSON OCR de natation avec Gemini 2.5 Flash."
    )
    parser.add_argument(
        "--input-dir",
        default=str(project_root / "data/json_from_pdfs/pdfs_results"),
        help="Dossier contenant les fichiers JSON source.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "data/json_structures"),
        help="Dossier de sortie des JSON structurés.",
    )
    parser.add_argument(
        "--progress-file",
        default=str(script_dir / "progress.json"),
        help="Fichier de suivi de progression.",
    )
    parser.add_argument(
        "--errors-dir",
        default=str(script_dir / "errors"),
        help="Dossier pour les réponses Gemini non parsables.",
    )
    parser.add_argument(
        "--log-file",
        default=str(script_dir / "processing.log"),
        help="Fichier de log.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modèle Gemini (défaut: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--daily-threshold",
        type=int,
        default=DAILY_REQUEST_THRESHOLD,
        help=f"Requêtes/jour max avant stop (défaut: {DAILY_REQUEST_THRESHOLD}).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=REQUEST_SLEEP_SECONDS,
        help=f"Pause entre requêtes en secondes (défaut: {REQUEST_SLEEP_SECONDS}s).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Nombre max de fichiers à traiter (0 = tous).",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=[],
        metavar="FICHIER",
        help=(
            "Noms de fichiers spécifiques à traiter (séparés par des espaces). "
            "Ex: --files fichier1.json fichier2.json"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Affiche les détails de traitement.",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Logger
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Progression / quota
# ══════════════════════════════════════════════════════════════════════════════

def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return DEFAULT_PROGRESS.copy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_PROGRESS.copy()
    merged = DEFAULT_PROGRESS.copy()
    merged.update(data if isinstance(data, dict) else {})
    if not isinstance(merged.get("processed_files"), list):
        merged["processed_files"] = []
    return merged


def save_progress(path: Path, progress: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_reset_daily_quota(progress: dict[str, Any]) -> None:
    today = date.today().isoformat()
    if progress.get("last_reset_date") != today:
        progress["last_reset_date"] = today
        progress["requests_today"]  = 0
        print(f"[quota] Nouveau jour ({today}) — compteur remis à 0.")


# ══════════════════════════════════════════════════════════════════════════════
# Extraction du texte source
# ══════════════════════════════════════════════════════════════════════════════

def extract_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    # Structure principale : { "file": "...", "content": "..." }
    for key in ("content", "full_text", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # Fallback : { "pages": [{ "text": "..." }] }
    pages = payload.get("pages")
    if isinstance(pages, list):
        chunks = [
            p.get("text", "")
            for p in pages
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        ]
        joined = "\n\n".join(c.strip() for c in chunks if c.strip())
        if joined:
            return joined
    return ""


def infer_source_filename(input_name: str, payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("file", "source_file", "pdf_file", "pdf_filename", "file_name"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                clean = val.strip()
                return clean if clean.lower().endswith(".pdf") else f"{Path(clean).stem}.pdf"
    return f"{Path(input_name).stem}.pdf"


# ══════════════════════════════════════════════════════════════════════════════
# Appel Gemini avec nouveau SDK google.genai
# ══════════════════════════════════════════════════════════════════════════════

def call_gemini(
    client:     genai.Client,
    model_name: str,
    text:       str,
    debug:      bool,
) -> tuple[str, int]:
    """Envoie le texte à Gemini et retourne (réponse_brute, tokens_utilisés)."""

    prompt = (
        "Voici le texte OCR à structurer.\n"
        "Retourne uniquement un JSON valide conforme au schéma demandé.\n\n"
        f"TEXTE:\n{text}"
    )

    last_exc: Exception | None = None

    for attempt in range(1, NETWORK_MAX_RETRIES + 1):
        try:
            if debug:
                print(f"  [debug] Tentative {attempt}/{NETWORK_MAX_RETRIES}...")

            t0 = time.perf_counter()

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )

            elapsed  = time.perf_counter() - t0
            text_out = response.text or ""

            # Tokens via usage_metadata
            usage = getattr(response, "usage_metadata", None)
            if usage:
                total = int(getattr(usage, "total_token_count", 0) or 0)
                if total == 0:
                    total = int(getattr(usage, "prompt_token_count", 0) or 0) + \
                            int(getattr(usage, "candidates_token_count", 0) or 0)
            else:
                total = 0

            if debug:
                print(f"  [debug] OK en {elapsed:.2f}s | {len(text_out)} chars | {total} tokens")

            return text_out, total

        except Exception as exc:
            last_exc = exc
            msg = str(exc)

            if "429" in msg:
                print(f"  [rate-limit] Erreur 429 — attente {RATE_LIMIT_RETRY_SECONDS}s...")
                time.sleep(RATE_LIMIT_RETRY_SECONDS)
                continue

            lower = msg.lower()
            transient = any(k in lower for k in ("500", "502", "503", "504", "timeout", "connection"))
            if transient and attempt < NETWORK_MAX_RETRIES:
                wait = NETWORK_BACKOFF_BASE ** attempt
                print(f"  [retry] Erreur transitoire, retry dans {wait}s...")
                time.sleep(wait)
                continue

            raise RuntimeError(f"Gemini erreur (tentative {attempt}): {exc}") from exc

    raise RuntimeError(
        f"Gemini : échec après {NETWORK_MAX_RETRIES} tentatives. "
        f"Dernière erreur : {last_exc}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation du JSON structuré
# ══════════════════════════════════════════════════════════════════════════════

def normalize_headers(raw_headers: Any, first_row: Any) -> list[str]:
    if isinstance(raw_headers, list):
        headers = [str(h).strip() for h in raw_headers if str(h).strip()]
        if headers:
            return headers
    if isinstance(first_row, dict):
        return list(first_row.keys())
    return []


def normalize_rows(raw_rows: Any, headers: list[str]) -> list[dict[str, str]]:
    if not isinstance(raw_rows, list):
        return []
    rows_out = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        rows_out.append({
            h: ("" if row.get(h) is None else str(row.get(h, "")).strip())
            for h in headers
        })
    return rows_out


def normalize_output(raw: Any, source_file: str) -> dict[str, Any]:
    tables_in  = raw.get("tables", []) if isinstance(raw, dict) else []
    tables_out = []

    for table in (tables_in if isinstance(tables_in, list) else []):
        if not isinstance(table, dict):
            continue
        category = str(table.get("category", "")).strip().upper()
        if category not in VALID_CATEGORIES:
            continue
        rows_in   = table.get("rows", [])
        first_row = rows_in[0] if isinstance(rows_in, list) and rows_in else None
        headers   = normalize_headers(table.get("headers"), first_row)
        if not headers:
            continue
        rows = normalize_rows(rows_in, headers)
        tables_out.append({"category": category, "headers": headers, "rows": rows})

    return {"source_file": source_file, "tables": tables_out}


# ══════════════════════════════════════════════════════════════════════════════
# Traitement d'un fichier
# ══════════════════════════════════════════════════════════════════════════════

def process_file(
    file_path:  Path,
    output_dir: Path,
    errors_dir: Path,
    client:     genai.Client,
    model_name: str,
    debug:      bool,
) -> tuple[str, int]:
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"JSON source invalide : {exc}", 0

    text = extract_text(payload)
    if not text:
        return "Aucun texte exploitable (clés attendues: content, full_text, text, pages).", 0

    if debug:
        print(f"  [debug] Texte extrait : {len(text)} caractères")

    raw_response, used_tokens = call_gemini(client, model_name, text, debug=debug)

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        errors_dir.mkdir(parents=True, exist_ok=True)
        (errors_dir / f"{file_path.stem}_invalid.txt").write_text(raw_response or "", encoding="utf-8")
        return "Réponse Gemini non-JSON → sauvegardée dans errors/", used_tokens

    source_file = infer_source_filename(file_path.name, payload)
    output      = normalize_output(parsed, source_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / file_path.name).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return "OK", used_tokens


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[erreur] GEMINI_API_KEY manquante.")
        print("         export GEMINI_API_KEY='AIza...'")
        return 1

    input_dir     = Path(args.input_dir)
    output_dir    = Path(args.output_dir)
    progress_file = Path(args.progress_file)
    errors_dir    = Path(args.errors_dir)
    logger        = setup_logger(Path(args.log_file))

    if not input_dir.is_dir():
        print(f"[erreur] Dossier introuvable : {input_dir}")
        return 1

    # ── Init nouveau SDK google.genai ──────────────────────────────────────────
    client = genai.Client(api_key=api_key)

    progress = load_progress(progress_file)
    maybe_reset_daily_quota(progress)

    processed      = set(progress.get("processed_files", []))
    requests_today = int(progress.get("requests_today", 0))

    all_files = sorted(p for p in input_dir.glob("*.json") if p.is_file())

    # Sélection des fichiers à traiter
    if args.files:
        # Mode --files : fichiers explicitement nommés
        selected = []
        for name in args.files:
            p = Path(name)
            target = p if p.is_absolute() else input_dir / p.name
            if not target.exists():
                print(f"[avertissement] Fichier introuvable, ignoré : {name}")
                continue
            selected.append(target)
        pending = [f for f in selected if f.name not in processed]
    else:
        # Mode normal : tous les fichiers non traités
        pending = [f for f in all_files if f.name not in processed]
        if args.max_files > 0:
            pending = pending[: args.max_files]

    if not pending:
        print("[info] Tous les fichiers sont déjà traités.")
        return 0

    print("=" * 60)
    print(f"  Modèle         : {args.model}")
    print(f"  Fichiers total  : {len(all_files)}")
    print(f"  Déjà traités   : {len(processed)}")
    print(f"  À traiter      : {len(pending)}")
    print(f"  Requêtes aujourd'hui : {requests_today} / {DAILY_REQUEST_LIMIT}")
    print(f"  Seuil d'arrêt  : {args.daily_threshold} req/jour")
    print("=" * 60)

    ok_count  = 0
    err_count = 0

    for i, file_path in enumerate(pending, start=1):

        if requests_today >= args.daily_threshold:
            print(f"\n[stop] Quota atteint : {requests_today} requêtes.")
            print("       Relance demain (reset à 08h00 heure du Maroc).")
            break

        print(f"\n[{i}/{len(pending)}] {file_path.name}")

        status      = "ERREUR"
        used_tokens = 0
        try:
            status, used_tokens = process_file(
                file_path  = file_path,
                output_dir = output_dir,
                errors_dir = errors_dir,
                client     = client,
                model_name = args.model,
                debug      = args.debug,
            )
        except Exception as exc:
            status = f"Exception : {exc}"

        requests_today += 1
        progress["requests_today"] = requests_today

        if status == "OK":
            ok_count += 1
            processed.add(file_path.name)
            progress["processed_files"] = sorted(processed)
            logger.info("%s | OK | tokens=%d | req=%d", file_path.name, used_tokens, requests_today)
            print(f"  ✓ OK | {used_tokens} tokens | req : {requests_today}/{DAILY_REQUEST_LIMIT}")
        else:
            err_count += 1
            logger.error("%s | ERREUR | %s", file_path.name, status)
            print(f"  ✗ ERREUR : {status}")

        save_progress(progress_file, progress)

        if i < len(pending):
            time.sleep(max(0.0, args.sleep))

    print("\n" + "=" * 60)
    print(f"  ✓ Succès  : {ok_count}")
    print(f"  ✗ Erreurs : {err_count}")
    print(f"  Requêtes aujourd'hui : {requests_today} / {DAILY_REQUEST_LIMIT}")
    print(f"  Total traités : {len(processed)} / {len(all_files)}")
    print("=" * 60)

    save_progress(progress_file, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())