#!/usr/bin/env python3
"""Structure les JSON OCR de natation avec Groq.

Usage:
    pip install groq
    export GROQ_API_KEY="gsk_..."

    # Traiter TOUS les fichiers :
    python groq_structuration.py

    # Traiter des fichiers spécifiques :
    python groq_structuration.py --files "fichier1.json" "fichier2.json" --debug

    # Limiter à N fichiers :
    python groq_structuration.py --max-files 10 --debug
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

from groq import Groq, RateLimitError, APIStatusError, APIConnectionError

# ── Modèle ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "llama-3.1-8b-instant"

# ── Limites Groq tier gratuit ──────────────────────────────────────────────────
DAILY_REQUEST_LIMIT     = 14_400
DAILY_REQUEST_THRESHOLD = 14_000

# ── Timing ────────────────────────────────────────────────────────────────────
# Groq tier gratuit : 6 000 TPM (tokens par minute), fenêtre glissante de 60s.
# On attend 62s entre CHAQUE requête (chunk) pour que le compteur TPM se vide.
# C'est la seule façon fiable de ne jamais avoir de 413.
INTER_REQUEST_SLEEP  = 62.0   # secondes entre chaque appel API
RATE_LIMIT_SLEEP     = 65.0   # si on reçoit quand même un 429
NETWORK_MAX_RETRIES  = 5
NETWORK_BACKOFF_BASE = 2

# ── Chunking ──────────────────────────────────────────────────────────────────
# Budget TPM : 6000 tok/min
# - Prompt système   : ~300 tok
# - Réponse estimée  : ~2000 tok
# - Budget texte     : ~3700 tok → ~14 800 chars (1 tok ≈ 4 chars)
# On prend 12 000 chars avec marge de sécurité.
INITIAL_CHUNK_CHARS = 12_000
MIN_CHUNK_CHARS     = 3_000   # seuil minimal si réduction adaptative

# ── Catégories valides ────────────────────────────────────────────────────────
VALID_CATEGORIES = {"BENJAMINS", "MINIMES", "CADETS", "JUNIORS", "SENIORS"}

# ── Prompt système ────────────────────────────────────────────────────────────
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

DEFAULT_PROGRESS: dict[str, Any] = {
    "requests_today":  0,
    "last_reset_date": "",
    "processed_files": [],
}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    default_input = (
        "/Users/nouhailaimaneabbassi/Desktop/SwimResultsExtractor"
        "/data/json_from_pdfs/pdfs_results"
    )
    default_output = (
        "/Users/nouhailaimaneabbassi/Desktop/SwimResultsExtractor"
        "/data/json_structures"
    )
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Structure tous les JSON OCR de natation avec Groq (traitement complet)."
    )
    parser.add_argument("--input-dir",  default=default_input,
        help="Dossier source des JSON OCR.")
    parser.add_argument("--output-dir", default=default_output,
        help="Dossier de sortie des JSON structurés.")
    parser.add_argument("--progress-file",
        default=str(script_dir / "progress_groq.json"),
        help="Fichier de suivi (reprend là où on s'est arrêté).")
    parser.add_argument("--errors-dir",
        default=str(script_dir / "errors"),
        help="Dossier pour les réponses non parsables.")
    parser.add_argument("--log-file",
        default=str(script_dir / "processing_groq.log"),
        help="Fichier de log.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
        help=f"Modèle Groq (défaut: {DEFAULT_MODEL}).")
    parser.add_argument("--daily-threshold", type=int, default=DAILY_REQUEST_THRESHOLD,
        help="Limite de requêtes/jour avant arrêt automatique.")
    parser.add_argument("--inter-request-sleep", type=float, default=INTER_REQUEST_SLEEP,
        help=f"Pause entre chaque appel API en secondes (défaut: {INTER_REQUEST_SLEEP}s).")
    parser.add_argument("--max-files", type=int, default=0,
        help="Nombre max de fichiers à traiter (0 = tous).")
    parser.add_argument("--files", nargs="+", default=[], metavar="FICHIER",
        help="Fichiers spécifiques à traiter (noms seulement, pas le chemin complet).")
    parser.add_argument("--debug", action="store_true",
        help="Affiche les détails de chaque requête.")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Logger
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("groq_structuration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    h = logging.FileHandler(log_path, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
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
    for key in ("content", "full_text", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    pages = payload.get("pages")
    if isinstance(pages, list):
        parts = [
            p.get("text", "")
            for p in pages
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        ]
        joined = "\n\n".join(c.strip() for c in parts if c.strip())
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
# Chunking
# ══════════════════════════════════════════════════════════════════════════════

def split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Découpe proprement aux sauts de ligne pour ne jamais couper une ligne de nageur."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        cut = text.rfind("\n", start, end)
        if cut <= start:
            cut = end
        chunks.append(text[start:cut])
        start = cut + 1
    return [c for c in chunks if c.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# Fusion des tables multi-chunks
# ══════════════════════════════════════════════════════════════════════════════

def merge_tables(all_tables: list[list[dict]]) -> list[dict]:
    """Même catégorie dans plusieurs chunks → rows concaténées."""
    merged: dict[str, dict] = {}
    for tables in all_tables:
        for table in tables:
            cat = table.get("category", "")
            if cat not in merged:
                merged[cat] = {
                    "category": cat,
                    "headers":  table.get("headers", []),
                    "rows":     [],
                }
            merged[cat]["rows"].extend(table.get("rows", []))
    return list(merged.values())


# ══════════════════════════════════════════════════════════════════════════════
# Appel Groq — 1 chunk
# ══════════════════════════════════════════════════════════════════════════════

def call_groq_chunk(
    client:       Groq,
    model_name:   str,
    chunk:        str,
    chunk_label:  str,
    inter_sleep:  float,
    debug:        bool,
) -> tuple[str, int]:
    """
    Envoie un chunk à Groq.
    - 429 RPM  → attend retry-after puis réessaie
    - 413 TPM  → réduit le chunk de moitié, attend 62s, réessaie
    - 5xx      → backoff exponentiel
    Retourne (réponse_brute, tokens_utilisés).
    """
    current_text = chunk

    for attempt in range(1, NETWORK_MAX_RETRIES + 1):
        try:
            if debug:
                print(f"  [debug] {chunk_label} tentative {attempt} "
                      f"({len(current_text)} chars)...")

            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": (
                        f"Voici le texte OCR à structurer [{chunk_label}].\n"
                        "Retourne uniquement un JSON valide conforme au schéma demandé.\n\n"
                        f"TEXTE:\n{current_text}"
                    )},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            elapsed   = time.perf_counter() - t0
            text_out  = response.choices[0].message.content or ""
            total_tok = response.usage.total_tokens if response.usage else 0

            if debug:
                print(f"  [debug] {chunk_label} ✓ {elapsed:.1f}s | {total_tok} tokens")

            return text_out, total_tok

        except RateLimitError as exc:
            # 429 : limite RPM (requêtes/minute)
            retry_after = RATE_LIMIT_SLEEP
            resp = getattr(exc, "response", None)
            if resp is not None:
                ra = getattr(resp, "headers", {}).get("retry-after")
                if ra:
                    try:
                        retry_after = int(float(ra)) + 2
                    except ValueError:
                        pass
            print(f"  [429 RPM] {chunk_label} → attente {retry_after:.0f}s...")
            time.sleep(retry_after)
            continue

        except APIStatusError as exc:
            code = exc.status_code

            if code == 413:
                # 413 : limite TPM (tokens/minute) → réduire le chunk
                new_size = max(len(current_text) // 2, MIN_CHUNK_CHARS)
                if new_size < len(current_text) and new_size >= MIN_CHUNK_CHARS:
                    print(f"  [413 TPM] {chunk_label} : {len(current_text)} chars trop grand "
                          f"→ réduit à {new_size} chars, attente {inter_sleep:.0f}s...")
                    current_text = current_text[:new_size]
                    time.sleep(inter_sleep)
                    continue
                raise RuntimeError(
                    f"{chunk_label} : chunk à {len(current_text)} chars encore trop grand. "
                    "Essaie un autre modèle ou attends que la fenêtre TPM se réinitialise."
                ) from exc

            transient = code in (500, 502, 503, 504) or "timeout" in str(exc).lower()
            if transient and attempt < NETWORK_MAX_RETRIES:
                wait = NETWORK_BACKOFF_BASE ** attempt
                print(f"  [retry {code}] {chunk_label} → retry dans {wait}s...")
                time.sleep(wait)
                continue

            raise RuntimeError(f"Groq erreur {code} ({chunk_label}): {exc}") from exc

        except APIConnectionError as exc:
            if attempt < NETWORK_MAX_RETRIES:
                wait = NETWORK_BACKOFF_BASE ** attempt
                print(f"  [connexion] {chunk_label} → retry dans {wait}s...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Connexion impossible ({chunk_label}): {exc}") from exc

        except Exception as exc:
            raise RuntimeError(f"Erreur inattendue ({chunk_label}): {exc}") from exc

    raise RuntimeError(f"Échec après {NETWORK_MAX_RETRIES} tentatives ({chunk_label})")


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation
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
# Traitement d'un fichier (peut générer plusieurs requêtes si chunking)
# ══════════════════════════════════════════════════════════════════════════════

def process_file(
    file_path:    Path,
    output_dir:   Path,
    errors_dir:   Path,
    client:       Groq,
    model_name:   str,
    inter_sleep:  float,
    debug:        bool,
    requests_today: int,        # passé par référence via retour
    daily_threshold: int,
) -> tuple[str, int, int]:
    """
    Retourne (status, tokens_utilisés, nb_requêtes_effectuées).
    """
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"JSON source invalide : {exc}", 0, 0

    text = extract_text(payload)
    if not text:
        return "Aucun texte exploitable.", 0, 0

    chunks      = split_into_chunks(text, INITIAL_CHUNK_CHARS)
    nb_chunks   = len(chunks)
    total_chars = len(text)

    if debug:
        print(f"  [debug] {total_chars} chars → {nb_chunks} chunk(s) "
              f"de ~{INITIAL_CHUNK_CHARS} chars max")
    else:
        if nb_chunks > 1:
            print(f"  → {nb_chunks} chunks ({total_chars} chars), "
                  f"durée estimée ~{(nb_chunks - 1) * inter_sleep:.0f}s d'attente")

    all_tables:    list[list[dict]] = []
    total_tokens:  int = 0
    nb_requests:   int = 0

    for idx, chunk in enumerate(chunks, start=1):
        label = f"chunk {idx}/{nb_chunks}"

        # Vérification quota avant chaque chunk (sauf le premier qui compte
        # déjà dans la boucle principale)
        if idx > 1 and (requests_today + nb_requests) >= daily_threshold:
            print(f"  [stop quota] Quota atteint avant {label}.")
            break

        # Attente TPM entre chunks (jamais avant le 1er chunk du fichier —
        # la boucle principale gère la pause entre fichiers)
        if idx > 1:
            print(f"  [attente {inter_sleep:.0f}s] fenêtre TPM avant {label}...")
            time.sleep(inter_sleep)

        try:
            raw_response, used_tokens = call_groq_chunk(
                client, model_name, chunk, label, inter_sleep, debug
            )
            nb_requests += 1
        except RuntimeError as exc:
            errors_dir.mkdir(parents=True, exist_ok=True)
            (errors_dir / f"{file_path.stem}_chunk{idx}_error.txt").write_text(
                str(exc), encoding="utf-8"
            )
            print(f"  ✗ {label} erreur : {exc}")
            nb_requests += 1
            continue

        total_tokens += used_tokens

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            errors_dir.mkdir(parents=True, exist_ok=True)
            (errors_dir / f"{file_path.stem}_chunk{idx}_invalid.txt").write_text(
                raw_response or "", encoding="utf-8"
            )
            if debug:
                print(f"  [debug] {label} réponse non-JSON → sauvegardée dans errors/")
            continue

        tables = parsed.get("tables", []) if isinstance(parsed, dict) else []
        all_tables.append(tables)

    if not all_tables:
        return "Aucun chunk traité avec succès → voir errors/", total_tokens, nb_requests

    source_file   = infer_source_filename(file_path.name, payload)
    merged_tables = merge_tables(all_tables)
    output        = normalize_output({"tables": merged_tables}, source_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / file_path.name).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return "OK", total_tokens, nb_requests


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = parse_args()

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[erreur] GROQ_API_KEY manquante.")
        print("         export GROQ_API_KEY='gsk_...'")
        return 1

    input_dir     = Path(args.input_dir)
    output_dir    = Path(args.output_dir)
    progress_file = Path(args.progress_file)
    errors_dir    = Path(args.errors_dir)
    logger        = setup_logger(Path(args.log_file))

    if not input_dir.is_dir():
        print(f"[erreur] Dossier introuvable : {input_dir}")
        return 1

    client = Groq(api_key=api_key)

    progress = load_progress(progress_file)
    maybe_reset_daily_quota(progress)

    processed      = set(progress.get("processed_files", []))
    requests_today = int(progress.get("requests_today", 0))
    all_files      = sorted(p for p in input_dir.glob("*.json") if p.is_file())

    # ── Sélection des fichiers à traiter ──────────────────────────────────────
    if args.files:
        selected = []
        for name in args.files:
            p      = Path(name)
            target = p if p.is_absolute() else input_dir / p.name
            if not target.exists():
                print(f"[avertissement] Fichier introuvable, ignoré : {name}")
                continue
            selected.append(target)
        pending = [f for f in selected if f.name not in processed]
    else:
        # Mode principal : tous les fichiers non encore traités
        pending = [f for f in all_files if f.name not in processed]
        if args.max_files > 0:
            pending = pending[: args.max_files]

    if not pending:
        print("[info] Tous les fichiers sont déjà traités. Rien à faire.")
        return 0

    # Estimation du temps total
    avg_chunks     = 2          # estimation conservative
    total_requests = len(pending) * avg_chunks
    est_minutes    = (total_requests * args.inter_request_sleep) / 60

    print("=" * 60)
    print(f"  Modèle               : {args.model}")
    print(f"  Chunk max            : {INITIAL_CHUNK_CHARS} chars (~{INITIAL_CHUNK_CHARS//4} tokens)")
    print(f"  Pause entre appels   : {args.inter_request_sleep:.0f}s")
    print(f"  Fichiers total       : {len(all_files)}")
    print(f"  Déjà traités         : {len(processed)}")
    print(f"  À traiter            : {len(pending)}")
    print(f"  Requêtes aujourd'hui : {requests_today} / {DAILY_REQUEST_LIMIT}")
    print(f"  Durée estimée        : ~{est_minutes:.0f} min (si ~{avg_chunks} chunks/fichier)")
    print("=" * 60)

    ok_count  = 0
    err_count = 0

    for i, file_path in enumerate(pending, start=1):

        if requests_today >= args.daily_threshold:
            print(f"\n[stop] Quota journalier atteint ({requests_today} requêtes). "
                  f"Relance demain — {len(pending) - i + 1} fichier(s) restants.")
            break

        print(f"\n[{i}/{len(pending)}] {file_path.name}")

        # Pause entre fichiers (sauf avant le tout premier)
        # On attend ici pour respecter le TPM entre le dernier chunk
        # du fichier précédent et le premier chunk du fichier suivant.
        if i > 1:
            print(f"  [attente {args.inter_request_sleep:.0f}s] fenêtre TPM entre fichiers...")
            time.sleep(args.inter_request_sleep)

        status      = "ERREUR"
        used_tokens = 0
        nb_req      = 0
        try:
            status, used_tokens, nb_req = process_file(
                file_path       = file_path,
                output_dir      = output_dir,
                errors_dir      = errors_dir,
                client          = client,
                model_name      = args.model,
                inter_sleep     = args.inter_request_sleep,
                debug           = args.debug,
                requests_today  = requests_today,
                daily_threshold = args.daily_threshold,
            )
        except Exception as exc:
            status = f"Exception : {exc}"
            nb_req = 1

        requests_today             += nb_req
        progress["requests_today"]  = requests_today

        if status == "OK":
            ok_count += 1
            processed.add(file_path.name)
            progress["processed_files"] = sorted(processed)
            logger.info("%s | OK | tokens=%d | req_total=%d",
                        file_path.name, used_tokens, requests_today)
            print(f"  ✓ OK | {used_tokens} tokens | "
                  f"requêtes aujourd'hui : {requests_today}/{DAILY_REQUEST_LIMIT}")
        else:
            err_count += 1
            logger.error("%s | ERREUR | %s", file_path.name, status)
            print(f"  ✗ ERREUR : {status}")

        save_progress(progress_file, progress)

    # ── Résumé final ──────────────────────────────────────────────────────────
    remaining = len(pending) - ok_count - err_count
    print("\n" + "=" * 60)
    print(f"  ✓ Succès          : {ok_count}")
    print(f"  ✗ Erreurs         : {err_count}")
    if remaining > 0:
        print(f"  ⏸ Non traités     : {remaining} (quota atteint)")
    print(f"  Requêtes aujourd'hui : {requests_today} / {DAILY_REQUEST_LIMIT}")
    print(f"  Total traités     : {len(processed)} / {len(all_files)}")
    print("=" * 60)

    save_progress(progress_file, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())