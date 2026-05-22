from __future__ import annotations

import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = "llama-3.1-8b-instant"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "json_from_pdfs" / "pdfs_results_actualites" / "2016"
OUTPUT_DIR = PROJECT_ROOT / "data" / "json_structures" / "results_from_actualites" / "2016"
SCRIPT_DIR = Path(__file__).resolve().parent
PROGRESS_FILE = SCRIPT_DIR / "progress_groq_actualites_2016.json"
ERRORS_DIR = SCRIPT_DIR / "errors_actualites"
LOG_FILE = SCRIPT_DIR / "processing_groq_actualites_2016.log"

DAILY_REQUEST_THRESHOLD = 14_000
INTER_REQUEST_SLEEP = 62.0
RATE_LIMIT_SLEEP = 65.0
NETWORK_MAX_RETRIES = 5
NETWORK_BACKOFF_BASE = 2

INITIAL_CHUNK_CHARS = 12_000
MIN_CHUNK_CHARS = 3_000

DEBUG = False

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

CATEGORY_ALIASES = {
    "BENJAMIN": "BENJAMINS",
    "BENJAMINS": "BENJAMINS",
    "MINIME": "MINIMES",
    "MINIMES": "MINIMES",
    "CADET": "CADETS",
    "CADETS": "CADETS",
    "JUNIOR": "JUNIORS",
    "JUNIORS": "JUNIORS",
    "SENIOR": "SENIORS",
    "SENIORS": "SENIORS",
}

VALID_CATEGORIES = set(CATEGORY_ALIASES.values())

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "Place": ("Place", "Rg", "Rang"),
    "Nom et prénom": ("Nom et prénom", "Nom et prenom", "Nom", "Nageur"),
    "Nation": ("Nation", "Nat"),
    "Naissance": ("Naissance", "Année", "Annee", "Date de naissance"),
    "Club": ("Club", "Equipe", "Équipe"),
    "Temps": ("Temps", "Tps", "Temps final"),
    "Points": ("Points", "Pts", "Point"),
    "Temps de passage": (
        "Temps de passage",
        "Passage",
        "Split",
        "Splits",
        "Obs",
        "Observation",
    ),
}

SYSTEM_PROMPT = """Tu es un extracteur de donnees de competitions de natation marocaine a partir de resultats publies dans les actualites FRMN.
Analyse le texte fourni et retourne UNIQUEMENT un objet JSON valide, sans markdown, sans backticks, sans explication.

Schema attendu :
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

Regles absolues :
- Extraire UNIQUEMENT les lignes compatibles avec ce schema.
- Ne pas inventer de nageurs ni de valeurs.
- Si un champ est absent, illisible ou non applicable, mettre "".
- category doit etre exactement l'une de : BENJAMINS, MINIMES, CADETS, JUNIORS, SENIORS.
- Ignorer les categories non autorisees (ex: POUSSINS) et les sections non compatibles avec ce schema.
- Un fichier peut contenir plusieurs tables (une par categorie trouvee dans le texte).
- Retourner uniquement le JSON brut, rien d'autre."""

DEFAULT_PROGRESS: dict[str, Any] = {
    "requests_today": 0,
    "last_reset_date": "",
    "processed_files": [],
}


class FatalGroqError(RuntimeError):
    """Erreur fatale de configuration/acces Groq: inutile de continuer le batch."""


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("groq_structuration_actualites_2016")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
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
    path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def maybe_reset_daily_quota(progress: dict[str, Any]) -> None:
    today = date.today().isoformat()
    if progress.get("last_reset_date") != today:
        progress["last_reset_date"] = today
        progress["requests_today"] = 0
        print(f"[quota] Nouveau jour ({today}) - compteur remis a 0.")


def extract_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    for key in ("content", "full_text", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    pages = payload.get("pages")
    if not isinstance(pages, list):
        return ""

    parts = [
        page.get("text", "")
        for page in pages
        if isinstance(page, dict) and isinstance(page.get("text"), str)
    ]
    joined = "\n\n".join(part.strip() for part in parts if part.strip())
    return joined.strip()


def infer_source_filename(input_name: str, payload: Any) -> str:
    if isinstance(payload, dict):
        for key in (
            "filename",
            "source_pdf",
            "source_file",
            "pdf_file",
            "pdf_filename",
            "file_name",
            "file",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                clean = Path(value.strip()).name
                return clean if clean.lower().endswith(".pdf") else f"{Path(clean).stem}.pdf"
    return f"{Path(input_name).stem}.pdf"


def split_into_chunks(text: str, max_chars: int) -> list[str]:
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

    return [chunk for chunk in chunks if chunk.strip()]


def normalize_category(raw_category: Any) -> str:
    category = str(raw_category or "").strip().upper()
    return CATEGORY_ALIASES.get(category, "")


def merge_tables(all_tables: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for tables in all_tables:
        for table in tables:
            category = normalize_category(table.get("category"))
            if category not in VALID_CATEGORIES:
                continue

            if category not in merged:
                merged[category] = {
                    "category": category,
                    "headers": DEFAULT_HEADERS[:],
                    "rows": [],
                }

            rows = table.get("rows", [])
            if isinstance(rows, list):
                merged[category]["rows"].extend(rows)

    return list(merged.values())


def call_groq_chunk(client: Groq, chunk: str, chunk_label: str) -> tuple[str, int]:
    current_text = chunk

    for attempt in range(1, NETWORK_MAX_RETRIES + 1):
        try:
            if DEBUG:
                print(
                    f"  [debug] {chunk_label} tentative {attempt} "
                    f"({len(current_text)} chars)..."
                )

            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Voici le texte OCR a structurer [{chunk_label}].\n"
                            "Retourne uniquement un JSON valide conforme au schema demande.\n\n"
                            f"TEXTE:\n{current_text}"
                        ),
                    },
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            elapsed = time.perf_counter() - t0
            text_out = response.choices[0].message.content or ""
            total_tokens = response.usage.total_tokens if response.usage else 0

            if DEBUG:
                print(f"  [debug] {chunk_label} OK {elapsed:.1f}s | {total_tokens} tokens")

            return text_out, total_tokens

        except RateLimitError as exc:
            retry_after = RATE_LIMIT_SLEEP
            response = getattr(exc, "response", None)
            if response is not None:
                headers = getattr(response, "headers", {})
                retry_header = headers.get("retry-after") if headers else None
                if retry_header:
                    try:
                        retry_after = int(float(retry_header)) + 2
                    except ValueError:
                        pass
            print(f"  [429 RPM] {chunk_label} -> attente {retry_after:.0f}s...")
            time.sleep(retry_after)
            continue

        except APIStatusError as exc:
            code = exc.status_code
            message = str(exc).lower()

            if code in (400, 401, 403):
                if (
                    "organization_restricted" in message
                    or "organization has been restricted" in message
                    or "invalid_api_key" in message
                    or "authentication" in message
                    or "unauthorized" in message
                    or "forbidden" in message
                ):
                    raise FatalGroqError(
                        "Acces Groq refuse par le compte/organisation "
                        "(organization_restricted ou erreur d'authentification). "
                        "Verifier la cle API et l'etat du compte Groq."
                    ) from exc

            if code == 413:
                new_size = max(len(current_text) // 2, MIN_CHUNK_CHARS)
                if new_size < len(current_text) and new_size >= MIN_CHUNK_CHARS:
                    print(
                        f"  [413 TPM] {chunk_label} : {len(current_text)} chars trop grand "
                        f"-> reduit a {new_size} chars, attente {INTER_REQUEST_SLEEP:.0f}s..."
                    )
                    current_text = current_text[:new_size]
                    time.sleep(INTER_REQUEST_SLEEP)
                    continue
                raise RuntimeError(
                    f"{chunk_label} : chunk a {len(current_text)} chars encore trop grand."
                ) from exc

            transient = code in (500, 502, 503, 504) or "timeout" in str(exc).lower()
            if transient and attempt < NETWORK_MAX_RETRIES:
                wait_time = NETWORK_BACKOFF_BASE ** attempt
                print(f"  [retry {code}] {chunk_label} -> retry dans {wait_time}s...")
                time.sleep(wait_time)
                continue

            raise RuntimeError(f"Groq erreur {code} ({chunk_label}) : {exc}") from exc

        except APIConnectionError as exc:
            if attempt < NETWORK_MAX_RETRIES:
                wait_time = NETWORK_BACKOFF_BASE ** attempt
                print(f"  [connexion] {chunk_label} -> retry dans {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"Connexion impossible ({chunk_label}) : {exc}") from exc

        except Exception as exc:
            raise RuntimeError(f"Erreur inattendue ({chunk_label}) : {exc}") from exc

    raise RuntimeError(f"Echec apres {NETWORK_MAX_RETRIES} tentatives ({chunk_label})")


def pick_value(row: dict[str, Any], aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        value = row.get(alias)
        if value is None:
            continue
        clean = str(value).strip()
        if clean:
            return clean
    return ""


def normalize_rows(raw_rows: Any) -> list[dict[str, str]]:
    if not isinstance(raw_rows, list):
        return []

    rows_out: list[dict[str, str]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue

        normalized_row = {
            header: pick_value(row, FIELD_ALIASES[header])
            for header in DEFAULT_HEADERS
        }
        if any(normalized_row.values()):
            rows_out.append(normalized_row)

    return rows_out


def normalize_output(raw: Any, source_file: str) -> dict[str, Any]:
    tables_in = raw.get("tables", []) if isinstance(raw, dict) else []
    tables_out = []

    for table in tables_in if isinstance(tables_in, list) else []:
        if not isinstance(table, dict):
            continue

        category = normalize_category(table.get("category"))
        if category not in VALID_CATEGORIES:
            continue

        rows = normalize_rows(table.get("rows"))
        if not rows:
            continue

        tables_out.append(
            {
                "category": category,
                "headers": DEFAULT_HEADERS[:],
                "rows": rows,
            }
        )

    return {
        "source_file": source_file,
        "tables": tables_out,
    }


def process_file(file_path: Path, client: Groq, requests_today: int) -> tuple[str, int, int]:
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"JSON source invalide : {exc}", 0, 0

    text = extract_text(payload)
    if not text:
        return "Aucun texte exploitable.", 0, 0

    chunks = split_into_chunks(text, INITIAL_CHUNK_CHARS)
    nb_chunks = len(chunks)

    if DEBUG:
        print(
            f"  [debug] {len(text)} chars -> {nb_chunks} chunk(s) "
            f"de ~{INITIAL_CHUNK_CHARS} chars max"
        )
    elif nb_chunks > 1:
        print(
            f"  -> {nb_chunks} chunks ({len(text)} chars), "
            f"duree estimee ~{(nb_chunks - 1) * INTER_REQUEST_SLEEP:.0f}s d'attente"
        )

    all_tables: list[list[dict[str, Any]]] = []
    total_tokens = 0
    nb_requests = 0

    for idx, chunk in enumerate(chunks, start=1):
        label = f"chunk {idx}/{nb_chunks}"

        if idx > 1 and (requests_today + nb_requests) >= DAILY_REQUEST_THRESHOLD:
            print(f"  [stop quota] Quota atteint avant {label}.")
            break

        if idx > 1:
            print(f"  [attente {INTER_REQUEST_SLEEP:.0f}s] fenetre TPM avant {label}...")
            time.sleep(INTER_REQUEST_SLEEP)

        try:
            raw_response, used_tokens = call_groq_chunk(client, chunk, label)
            nb_requests += 1
        except FatalGroqError:
            raise
        except RuntimeError as exc:
            ERRORS_DIR.mkdir(parents=True, exist_ok=True)
            (ERRORS_DIR / f"{file_path.stem}_chunk{idx}_error.txt").write_text(
                str(exc),
                encoding="utf-8",
            )
            print(f"  X {label} erreur : {exc}")
            nb_requests += 1
            continue

        total_tokens += used_tokens

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            ERRORS_DIR.mkdir(parents=True, exist_ok=True)
            (ERRORS_DIR / f"{file_path.stem}_chunk{idx}_invalid.txt").write_text(
                raw_response or "",
                encoding="utf-8",
            )
            if DEBUG:
                print(f"  [debug] {label} reponse non JSON -> sauvegardee dans errors/")
            continue

        tables = parsed.get("tables", []) if isinstance(parsed, dict) else []
        if isinstance(tables, list):
            all_tables.append(tables)

    if not all_tables:
        return "Aucun chunk traite avec succes -> voir errors/", total_tokens, nb_requests

    source_file = infer_source_filename(file_path.name, payload)
    merged_tables = merge_tables(all_tables)
    output = normalize_output({"tables": merged_tables}, source_file)
    if not output["tables"]:
        return "Aucune table valide extraite apres normalisation.", total_tokens, nb_requests

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / file_path.name).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
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

    processed = set(progress.get("processed_files", []))
    requests_today = int(progress.get("requests_today", 0))
    all_files = sorted(path for path in INPUT_DIR.glob("*.json") if path.is_file())
    already_in_output = (
        {path.name for path in OUTPUT_DIR.glob("*.json")}
        if OUTPUT_DIR.is_dir()
        else set()
    )
    pending = [
        file_path
        for file_path in all_files
        if file_path.name not in processed and file_path.name not in already_in_output
    ]

    skipped = len(already_in_output.intersection({path.name for path in all_files}))
    if skipped:
        print(
            f"[info] {skipped} fichier(s) ignores car deja presents "
            "dans le dossier de sortie."
        )

    if not pending:
        print("[info] Tous les fichiers sont deja traites. Rien a faire.")
        return 0

    print("=" * 60)
    print(f"  Modele               : {MODEL}")
    print(f"  Dossier source       : {INPUT_DIR}")
    print(f"  Dossier sortie       : {OUTPUT_DIR}")
    print(f"  Chunk max            : {INITIAL_CHUNK_CHARS} chars")
    print(f"  Pause entre appels   : {INTER_REQUEST_SLEEP:.0f}s")
    print(f"  Fichiers total       : {len(all_files)}")
    print(f"  Deja traites         : {len(processed)}")
    print(f"  A traiter            : {len(pending)}")
    print(f"  Requetes aujourd'hui : {requests_today} / {DAILY_REQUEST_THRESHOLD}")
    print("=" * 60)

    ok_count = 0
    err_count = 0

    for index, file_path in enumerate(pending, start=1):
        if requests_today >= DAILY_REQUEST_THRESHOLD:
            print(
                f"\n[stop] Quota journalier atteint ({requests_today} requetes). "
                f"Relance demain - {len(pending) - index + 1} fichier(s) restants."
            )
            break

        print(f"\n[{index}/{len(pending)}] {file_path.name}")

        if index > 1:
            print(
                f"  [attente {INTER_REQUEST_SLEEP:.0f}s] "
                "fenetre TPM entre fichiers..."
            )
            time.sleep(INTER_REQUEST_SLEEP)

        status = "ERREUR"
        used_tokens = 0
        nb_req = 0

        try:
            status, used_tokens, nb_req = process_file(
                file_path=file_path,
                client=client,
                requests_today=requests_today,
            )
        except FatalGroqError as exc:
            logger.error("%s | FATAL | %s", file_path.name, exc)
            save_progress(PROGRESS_FILE, progress)
            print(f"  FATAL : {exc}")
            print("[stop] Arret immediat du traitement: erreur Groq permanente.")
            return 1
        except Exception as exc:
            status = f"Exception : {exc}"
            nb_req = 1

        requests_today += nb_req
        progress["requests_today"] = requests_today

        if status == "OK":
            ok_count += 1
            processed.add(file_path.name)
            progress["processed_files"] = sorted(processed)
            logger.info(
                "%s | OK | tokens=%d | req_total=%d",
                file_path.name,
                used_tokens,
                requests_today,
            )
            print(
                f"  OK | {used_tokens} tokens | "
                f"requetes aujourd'hui : {requests_today}/{DAILY_REQUEST_THRESHOLD}"
            )
        else:
            err_count += 1
            logger.error("%s | ERREUR | %s", file_path.name, status)
            print(f"  ERREUR : {status}")

        save_progress(PROGRESS_FILE, progress)

    remaining = len(pending) - ok_count - err_count
    print("\n" + "=" * 60)
    print(f"  OK                  : {ok_count}")
    print(f"  Erreurs             : {err_count}")
    if remaining > 0:
        print(f"  Non traites         : {remaining} (quota atteint)")
    print(f"  Requetes aujourd'hui : {requests_today} / {DAILY_REQUEST_THRESHOLD}")
    print(f"  Total traites       : {len(processed)} / {len(all_files)}")
    print("=" * 60)

    save_progress(PROGRESS_FILE, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
