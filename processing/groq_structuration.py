from __future__ import annotations

import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

from groq import Groq, RateLimitError, APIStatusError, APIConnectionError
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = "llama-3.1-8b-instant"

INPUT_DIR     = Path("/Users/nouhailaimaneabbassi/Desktop/SwimResultsExtractor/data/json_from_pdfs/pdfs_results")
OUTPUT_DIR    = Path("/Users/nouhailaimaneabbassi/Desktop/SwimResultsExtractor/data/json_structures/pdfs_results")
SCRIPT_DIR    = Path(__file__).resolve().parent
PROGRESS_FILE = SCRIPT_DIR / "progress_groq.json"
ERRORS_DIR    = SCRIPT_DIR / "errors"
LOG_FILE      = SCRIPT_DIR / "processing_groq.log"

DAILY_REQUEST_THRESHOLD = 14_000
INTER_REQUEST_SLEEP     = 62.0
RATE_LIMIT_SLEEP        = 65.0
NETWORK_MAX_RETRIES     = 5
NETWORK_BACKOFF_BASE    = 2

INITIAL_CHUNK_CHARS = 12_000
MIN_CHUNK_CHARS     = 3_000

DEBUG = False

VALID_CATEGORIES = {"BENJAMINS", "MINIMES", "CADETS", "JUNIORS", "SENIORS"}

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

def call_groq_chunk(
    client:      Groq,
    chunk:       str,
    chunk_label: str,
) -> tuple[str, int]:
    """
    Envoie un chunk à Groq.
    - 429 RPM  → attend retry-after puis réessaie
    - 413 TPM  → réduit le chunk de moitié, attend INTER_REQUEST_SLEEP, réessaie
    - 5xx      → backoff exponentiel
    Retourne (réponse_brute, tokens_utilisés).
    """
    current_text = chunk

    for attempt in range(1, NETWORK_MAX_RETRIES + 1):
        try:
            if DEBUG:
                print(f"  [debug] {chunk_label} tentative {attempt} "
                      f"({len(current_text)} chars)...")

            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL,
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

            if DEBUG:
                print(f"  [debug] {chunk_label} ✓ {elapsed:.1f}s | {total_tok} tokens")

            return text_out, total_tok

        except RateLimitError as exc:
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
                new_size = max(len(current_text) // 2, MIN_CHUNK_CHARS)
                if new_size < len(current_text) and new_size >= MIN_CHUNK_CHARS:
                    print(f"  [413 TPM] {chunk_label} : {len(current_text)} chars trop grand "
                          f"→ réduit à {new_size} chars, attente {INTER_REQUEST_SLEEP:.0f}s...")
                    current_text = current_text[:new_size]
                    time.sleep(INTER_REQUEST_SLEEP)
                    continue
                raise RuntimeError(
                    f"{chunk_label} : chunk à {len(current_text)} chars encore trop grand."
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


def process_file(
    file_path:      Path,
    client:         Groq,
    requests_today: int,
) -> tuple[str, int, int]:
    """Retourne (status, tokens_utilisés, nb_requêtes_effectuées)."""
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"JSON source invalide : {exc}", 0, 0

    text = extract_text(payload)
    if not text:
        return "Aucun texte exploitable.", 0, 0

    chunks    = split_into_chunks(text, INITIAL_CHUNK_CHARS)
    nb_chunks = len(chunks)

    if DEBUG:
        print(f"  [debug] {len(text)} chars → {nb_chunks} chunk(s) de ~{INITIAL_CHUNK_CHARS} chars max")
    elif nb_chunks > 1:
        print(f"  → {nb_chunks} chunks ({len(text)} chars), "
              f"durée estimée ~{(nb_chunks - 1) * INTER_REQUEST_SLEEP:.0f}s d'attente")

    all_tables:   list[list[dict]] = []
    total_tokens: int = 0
    nb_requests:  int = 0

    for idx, chunk in enumerate(chunks, start=1):
        label = f"chunk {idx}/{nb_chunks}"

        if idx > 1 and (requests_today + nb_requests) >= DAILY_REQUEST_THRESHOLD:
            print(f"  [stop quota] Quota atteint avant {label}.")
            break

        if idx > 1:
            print(f"  [attente {INTER_REQUEST_SLEEP:.0f}s] fenêtre TPM avant {label}...")
            time.sleep(INTER_REQUEST_SLEEP)

        try:
            raw_response, used_tokens = call_groq_chunk(client, chunk, label)
            nb_requests += 1
        except RuntimeError as exc:
            ERRORS_DIR.mkdir(parents=True, exist_ok=True)
            (ERRORS_DIR / f"{file_path.stem}_chunk{idx}_error.txt").write_text(
                str(exc), encoding="utf-8"
            )
            print(f"  ✗ {label} erreur : {exc}")
            nb_requests += 1
            continue

        total_tokens += used_tokens

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            ERRORS_DIR.mkdir(parents=True, exist_ok=True)
            (ERRORS_DIR / f"{file_path.stem}_chunk{idx}_invalid.txt").write_text(
                raw_response or "", encoding="utf-8"
            )
            if DEBUG:
                print(f"  [debug] {label} réponse non-JSON → sauvegardée dans errors/")
            continue

        tables = parsed.get("tables", []) if isinstance(parsed, dict) else []
        all_tables.append(tables)

    if not all_tables:
        return "Aucun chunk traité avec succès → voir errors/", total_tokens, nb_requests

    source_file   = infer_source_filename(file_path.name, payload)
    merged_tables = merge_tables(all_tables)
    output        = normalize_output({"tables": merged_tables}, source_file)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / file_path.name).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return "OK", total_tokens, nb_requests


def main() -> int:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[erreur] GROQ_API_KEY manquante.")
        print("         export GROQ_API_KEY='gsk_...'")
        return 1

    if not INPUT_DIR.is_dir():
        print(f"[erreur] Dossier introuvable : {INPUT_DIR}")
        return 1

    logger = setup_logger(LOG_FILE)
    client = Groq(api_key=api_key)

    progress = load_progress(PROGRESS_FILE)
    maybe_reset_daily_quota(progress)

    processed      = set(progress.get("processed_files", []))
    requests_today = int(progress.get("requests_today", 0))
    all_files      = sorted(p for p in INPUT_DIR.glob("*.json") if p.is_file())

    already_in_output = {p.name for p in OUTPUT_DIR.glob("*.json")} if OUTPUT_DIR.is_dir() else set()
    pending = [f for f in all_files if f.name not in processed and f.name not in already_in_output]

    skipped = len(already_in_output.intersection({f.name for f in all_files}))
    if skipped:
        print(f"[info] {skipped} fichier(s) ignorés car déjà présents dans le dossier de sortie.")

    if not pending:
        print("[info] Tous les fichiers sont déjà traités. Rien à faire.")
        return 0

    print("=" * 60)
    print(f"  Modèle               : {MODEL}")
    print(f"  Chunk max            : {INITIAL_CHUNK_CHARS} chars")
    print(f"  Pause entre appels   : {INTER_REQUEST_SLEEP:.0f}s")
    print(f"  Fichiers total       : {len(all_files)}")
    print(f"  Déjà traités         : {len(processed)}")
    print(f"  À traiter            : {len(pending)}")
    print(f"  Requêtes aujourd'hui : {requests_today} / {DAILY_REQUEST_THRESHOLD}")
    print("=" * 60)

    ok_count  = 0
    err_count = 0

    for i, file_path in enumerate(pending, start=1):

        if requests_today >= DAILY_REQUEST_THRESHOLD:
            print(f"\n[stop] Quota journalier atteint ({requests_today} requêtes). "
                  f"Relance demain — {len(pending) - i + 1} fichier(s) restants.")
            break

        print(f"\n[{i}/{len(pending)}] {file_path.name}")

        if i > 1:
            print(f"  [attente {INTER_REQUEST_SLEEP:.0f}s] fenêtre TPM entre fichiers...")
            time.sleep(INTER_REQUEST_SLEEP)

        status      = "ERREUR"
        used_tokens = 0
        nb_req      = 0
        try:
            status, used_tokens, nb_req = process_file(
                file_path      = file_path,
                client         = client,
                requests_today = requests_today,
            )
        except Exception as exc:
            status = f"Exception : {exc}"
            nb_req = 1

        requests_today            += nb_req
        progress["requests_today"] = requests_today

        if status == "OK":
            ok_count += 1
            processed.add(file_path.name)
            progress["processed_files"] = sorted(processed)
            logger.info("%s | OK | tokens=%d | req_total=%d",
                        file_path.name, used_tokens, requests_today)
            print(f"  ✓ OK | {used_tokens} tokens | "
                  f"requêtes aujourd'hui : {requests_today}/{DAILY_REQUEST_THRESHOLD}")
        else:
            err_count += 1
            logger.error("%s | ERREUR | %s", file_path.name, status)
            print(f"  ✗ ERREUR : {status}")

        save_progress(PROGRESS_FILE, progress)

    remaining = len(pending) - ok_count - err_count
    print("\n" + "=" * 60)
    print(f"  ✓ Succès             : {ok_count}")
    print(f"  ✗ Erreurs            : {err_count}")
    if remaining > 0:
        print(f"  ⏸ Non traités        : {remaining} (quota atteint)")
    print(f"  Requêtes aujourd'hui : {requests_today} / {DAILY_REQUEST_THRESHOLD}")
    print(f"  Total traités        : {len(processed)} / {len(all_files)}")
    print("=" * 60)

    save_progress(PROGRESS_FILE, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())