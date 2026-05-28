from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

load_dotenv(Path(__file__).resolve().parent / ".env")

# Clé dédiée actualités (processing/.env) ; repli sur GROQ_API_KEY si absente
GROQ_API_KEY_ENV = "GROQ_API_KEY1"

MODEL = "llama-3.1-8b-instant"

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
INPUT_DIR = BASE_DIR / "data" / "json_from_pdfs" / "pdfs_results_actualites" / "2016"
OUTPUT_DIR = BASE_DIR / "data" / "json_structures" / "results_from_actualites" / "2016"
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

# Même format que data/html_results/*.json (scraper/html_results_scraper.py)
TARGET_SCHEMA: dict[str, Any] = {
    "SwimDate": "2016-07-24",
    "SwimYear": 2016,
    "Meet": "CHAMPIONNATS DU MAROC M C J S ET OPEN - CASABLANCA",
    "location": "",
    "Country": "MAR",
    "epreuves": [
        {
            "Event": "50 FR SCM",
            "Distance": 50,
            "Stroke": "FR",
            "Course": "SCM",
            "PoolLength": 25,
            "tour": "Finale A",
            "performances": [
                {
                    "Rank": 1,
                    "club": "TSC",
                    "SwimTime": "28.14",
                    "SwimTimeSeconds": 28.14,
                    "Status": "OK",
                    "Speed": 1.7768,
                    "swimmer": {
                        "Name": "MANA Noura",
                        "Gender": "F",
                        "Year_of_birth": 1997,
                        "Age": 19,
                        "Nationality": "MAR",
                    },
                }
            ],
        }
    ],
}

STROKE_ALIASES: dict[str, str] = {
    "FR": "FR",
    "FREE": "FR",
    "CRAWL": "FR",
    "NAGE LIBRE": "FR",
    "NL": "FR",
    "DOS": "DOS",
    "BK": "DOS",
    "BACK": "DOS",
    "BR": "BR",
    "BREAST": "BR",
    "BRASSE": "BR",
    "PAP": "PAP",
    "FLY": "PAP",
    "FL": "PAP",
    "PAPILLON": "PAP",
    "4N": "4N",
    "IM": "4N",
    "4 NAGES": "4N",
    "REL": "REL",
    "RELAIS": "REL",
}

SYSTEM_PROMPT = """Tu es un extracteur de résultats de natation FRMN (actualités / PDF OCR).
Analyse le texte fourni et retourne UNIQUEMENT un objet JSON valide, sans markdown, sans backticks, sans explication.

Schéma cible (identique aux fichiers data/html_results/) :
{schema}

Le texte peut provenir d'un article ou PDF d'actualité : tableaux de résultats, records, classements.
Extraire chaque nageur avec temps et club lorsque le texte le permet.
Si le document ne contient que des totaux par club ou des points sans performances individuelles,
retourner "epreuves": [] en renseignant quand même SwimDate, SwimYear, Meet, location si possible.

Règles absolues :
- Ne pas inventer de nageurs : extraire UNIQUEMENT ce qui est dans le texte.
- Champ absent ou illisible → null (sauf "location" → chaîne vide "").
- SwimDate : date ISO YYYY-MM-DD si identifiable dans le titre ou le texte.
- SwimYear : année entière déduite de SwimDate.
- Meet : nom de la compétition / article sans date ni ville superflues.
- location : ville ou "".
- Country : "MAR" sauf indication contraire explicite.
- Bassin FRMN "Grand bassin" / "Petit bassin" → Course "SCM", PoolLength 25.
- Event : "{{distance}} {{stroke}} {{course}}" (ex. "50 DOS SCM", "100 FR SCM").
- Stroke : FR, DOS, BR, PAP, 4N, REL (NL/nage libre → FR, pap → PAP, dos → DOS).
- Relais "4 x 50 m …" : Distance = 4 × distance d'un relais, Stroke = "REL".
- DAMES → Gender "F" ; MESSIEURS → Gender "M".
- tour : catégorie d'âge (BENJAMINS, MINIMES, CADETS, JUNIORS, SENIORS, POUSSINS) ou tour (Finale A, Séries…).
- Rank : entier si "1.", "2."… ; null si "NC.".
- SwimTime / SwimTimeSeconds / Status / Speed / swimmer.Age : comme pour les résultats officiels FRMN.
- Une entrée "epreuves" par combinaison (épreuve + tour/catégorie).
- Retourner uniquement le JSON brut.""".format(
    schema=json.dumps(TARGET_SCHEMA, ensure_ascii=False, indent=2),
)

DEFAULT_PROGRESS: dict[str, Any] = {
    "requests_today": 0,
    "last_reset_date": "",
    "processed_files": [],
}


class FatalGroqError(RuntimeError):
    """Erreur fatale de configuration/acces Groq: inutile de continuer le batch."""


def get_groq_api_key() -> str:
    return (
        os.getenv(GROQ_API_KEY_ENV, "").strip()
        or os.getenv("GROQ_API_KEY", "").strip()
    )


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
        print(f"[quota] Nouveau jour ({today}) — compteur remis à 0.")


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
    return "\n\n".join(part.strip() for part in parts if part.strip())


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


def _epreuve_key(epreuve: dict) -> str:
    parts = [
        str(epreuve.get("Event", "")).strip(),
        str(epreuve.get("Distance", "")).strip(),
        str(epreuve.get("Stroke", "")).strip(),
        str(epreuve.get("Course", "")).strip(),
        str(epreuve.get("tour", "")).strip(),
    ]
    return "|".join(parts)


def merge_epreuves(all_epreuves: list[list[dict]]) -> list[dict]:
    merged: dict[str, dict] = {}
    for epreuves in all_epreuves:
        for ep in epreuves:
            if not isinstance(ep, dict):
                continue
            key = _epreuve_key(ep)
            if key not in merged:
                merged[key] = {
                    "Event": ep.get("Event"),
                    "Distance": ep.get("Distance"),
                    "Stroke": ep.get("Stroke"),
                    "Course": ep.get("Course"),
                    "PoolLength": ep.get("PoolLength"),
                    "tour": ep.get("tour"),
                    "performances": [],
                }
            perfs = ep.get("performances", [])
            if isinstance(perfs, list):
                merged[key]["performances"].extend(perfs)
    return list(merged.values())


def merge_metadata(chunks: list[dict]) -> dict[str, Any]:
    fields = ("SwimDate", "SwimYear", "Meet", "location", "Country")
    out: dict[str, Any] = {k: None for k in fields}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for key in fields:
            if out[key] is None:
                val = chunk.get(key)
                if val is not None and val != "":
                    out[key] = val
    return out


def parse_swim_time_seconds(swim_time: str | None) -> float | None:
    if not swim_time or not isinstance(swim_time, str):
        return None
    s = swim_time.strip()
    if not s or s.lower() in {"frf n.d.", "n.d.", "-"}:
        return None
    if re.search(r"dsq|disqual|abandon", s, re.IGNORECASE):
        return None
    try:
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 2:
                minutes, seconds = parts
                return int(minutes) * 60 + float(seconds.replace(",", "."))
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return (
                    int(hours) * 3600
                    + int(minutes) * 60
                    + float(seconds.replace(",", "."))
                )
        return float(s.replace(",", "."))
    except ValueError:
        return None


def compute_speed(distance: int | None, swim_time_seconds: float | None) -> float | None:
    if distance is None or swim_time_seconds is None:
        return None
    if distance <= 0 or swim_time_seconds <= 0:
        return None
    return round(distance / swim_time_seconds, 4)


def swim_year_from_date(swim_date: str | None) -> int | None:
    if not swim_date:
        return None
    try:
        return datetime.strptime(swim_date[:10], "%Y-%m-%d").year
    except ValueError:
        return None


def normalize_stroke_code(stroke: str | None) -> str | None:
    if not stroke:
        return None
    key = stroke.strip().upper()
    return STROKE_ALIASES.get(key, key or None)


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
                            f"Voici le texte OCR à structurer [{chunk_label}].\n"
                            "Retourne uniquement un JSON valide conforme au schéma demandé.\n\n"
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
            print(f"  [429 RPM] {chunk_label} → attente {retry_after:.0f}s...")
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
                        "Accès Groq refusé (organization_restricted ou authentification). "
                        "Vérifier la clé API et l'état du compte Groq."
                    ) from exc

            if code == 413:
                new_size = max(len(current_text) // 2, MIN_CHUNK_CHARS)
                if new_size < len(current_text) and new_size >= MIN_CHUNK_CHARS:
                    print(
                        f"  [413 TPM] {chunk_label} : {len(current_text)} chars trop grand "
                        f"→ réduit à {new_size} chars, attente {INTER_REQUEST_SLEEP:.0f}s..."
                    )
                    current_text = current_text[:new_size]
                    time.sleep(INTER_REQUEST_SLEEP)
                    continue
                raise RuntimeError(
                    f"{chunk_label} : chunk à {len(current_text)} chars encore trop grand."
                ) from exc

            transient = code in (500, 502, 503, 504) or "timeout" in str(exc).lower()
            if transient and attempt < NETWORK_MAX_RETRIES:
                wait_time = NETWORK_BACKOFF_BASE ** attempt
                print(f"  [retry {code}] {chunk_label} → retry dans {wait_time}s...")
                time.sleep(wait_time)
                continue

            raise RuntimeError(f"Groq erreur {code} ({chunk_label}) : {exc}") from exc

        except APIConnectionError as exc:
            if attempt < NETWORK_MAX_RETRIES:
                wait_time = NETWORK_BACKOFF_BASE ** attempt
                print(f"  [connexion] {chunk_label} → retry dans {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"Connexion impossible ({chunk_label}) : {exc}") from exc

        except Exception as exc:
            raise RuntimeError(f"Erreur inattendue ({chunk_label}) : {exc}") from exc

    raise RuntimeError(f"Échec après {NETWORK_MAX_RETRIES} tentatives ({chunk_label})")


def _null_or_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value).strip() or None


def _null_or_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _null_or_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_swimmer(raw: Any, swim_year: int | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    year_of_birth = _null_or_int(raw.get("Year_of_birth"))
    age = _null_or_int(raw.get("Age"))
    if age is None:
        age = _null_or_int(raw.get("Age_at_Performance"))
    if age is None and swim_year is not None and year_of_birth is not None:
        age = swim_year - year_of_birth
    gender = _null_or_str(raw.get("Gender"))
    if gender:
        g = gender.upper()
        gender = "F" if g in {"F", "FEMME", "DAMES", "FEMININ", "FÉMININ"} else (
            "M" if g in {"M", "HOMME", "MESSIEURS", "MASCULIN"} else gender
        )
    return {
        "Name": _null_or_str(raw.get("Name")),
        "Gender": gender,
        "Year_of_birth": year_of_birth,
        "Age": age,
        "Nationality": _null_or_str(raw.get("Nationality")),
    }


def normalize_status(swim_time: str | None, status: str | None) -> str:
    if status:
        s = status.strip().upper()
        if s in {"OK", "NC", "DSQ", "DNF", "DNS"}:
            return s
        if s in {"DQ", "DISQ"}:
            return "DSQ"
    time_l = (swim_time or "").strip().lower()
    if "dsq" in time_l or "disqual" in time_l:
        return "DSQ"
    if "abandon" in time_l:
        return "DNF"
    if "frf n.d" in time_l or time_l in {"n.d.", "-"}:
        return "DNS"
    return "OK"


def normalize_performance(
    raw: Any,
    swim_year: int | None,
    distance: int | None,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    swim_time = _null_or_str(raw.get("SwimTime"))
    swim_secs = _null_or_float(raw.get("SwimTimeSeconds"))
    if swim_secs is None:
        swim_secs = parse_swim_time_seconds(swim_time)
    rank = _null_or_int(raw.get("Rank"))
    status = normalize_status(swim_time, _null_or_str(raw.get("Status")))
    speed = _null_or_float(raw.get("Speed"))
    if speed is None:
        speed = compute_speed(distance, swim_secs)
    return {
        "Rank": rank,
        "club": _null_or_str(raw.get("club")),
        "SwimTime": swim_time,
        "SwimTimeSeconds": swim_secs,
        "Status": status,
        "Speed": speed,
        "swimmer": normalize_swimmer(raw.get("swimmer"), swim_year),
    }


def normalize_epreuve(raw: Any, swim_year: int | None) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    event = _null_or_str(raw.get("Event"))
    if not event:
        return None
    distance = _null_or_int(raw.get("Distance"))
    course = _null_or_str(raw.get("Course")) or "SCM"
    pool_length = _null_or_int(raw.get("PoolLength"))
    if pool_length is None:
        pool_length = 25 if course == "SCM" else 50
    stroke = normalize_stroke_code(_null_or_str(raw.get("Stroke")))
    perfs_in = raw.get("performances", [])
    performances = [
        normalize_performance(p, swim_year, distance)
        for p in (perfs_in if isinstance(perfs_in, list) else [])
        if isinstance(p, dict)
    ]
    return {
        "Event": event,
        "Distance": distance,
        "Stroke": stroke,
        "Course": course,
        "PoolLength": pool_length,
        "tour": _null_or_str(raw.get("tour")) or "",
        "performances": performances,
    }


def normalize_output(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    swim_date = _null_or_str(raw.get("SwimDate"))
    swim_year = _null_or_int(raw.get("SwimYear"))
    if swim_year is None:
        swim_year = swim_year_from_date(swim_date)
    country = _null_or_str(raw.get("Country")) or "MAR"
    location = raw.get("location")
    if location is None:
        location = ""
    else:
        location = str(location).strip()
    epreuves_out: list[dict[str, Any]] = []
    for ep in raw.get("epreuves", []) if isinstance(raw.get("epreuves"), list) else []:
        normalized = normalize_epreuve(ep, swim_year)
        if normalized is not None:
            epreuves_out.append(normalized)
    return {
        "SwimDate": swim_date,
        "SwimYear": swim_year,
        "Meet": _null_or_str(raw.get("Meet")) or "",
        "location": location,
        "Country": country,
        "epreuves": epreuves_out,
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
            f"  [debug] {len(text)} chars → {nb_chunks} chunk(s) "
            f"de ~{INITIAL_CHUNK_CHARS} chars max"
        )
    elif nb_chunks > 1:
        print(
            f"  → {nb_chunks} chunks ({len(text)} chars), "
            f"durée estimée ~{(nb_chunks - 1) * INTER_REQUEST_SLEEP:.0f}s d'attente"
        )

    all_parsed: list[dict] = []
    total_tokens = 0
    nb_requests = 0

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
        except FatalGroqError:
            raise
        except RuntimeError as exc:
            ERRORS_DIR.mkdir(parents=True, exist_ok=True)
            (ERRORS_DIR / f"{file_path.stem}_chunk{idx}_error.txt").write_text(
                str(exc),
                encoding="utf-8",
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
                raw_response or "",
                encoding="utf-8",
            )
            if DEBUG:
                print(f"  [debug] {label} réponse non-JSON → sauvegardée dans errors/")
            continue

        all_parsed.append(parsed if isinstance(parsed, dict) else {})

    if not all_parsed:
        return "Aucun chunk traité avec succès → voir errors/", total_tokens, nb_requests

    all_epreuves = [
        p.get("epreuves", [])
        for p in all_parsed
        if isinstance(p.get("epreuves"), list)
    ]
    metadata = merge_metadata(all_parsed)
    merged_epreuves = merge_epreuves(all_epreuves)
    output = normalize_output({**metadata, "epreuves": merged_epreuves})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / file_path.name).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return "OK", total_tokens, nb_requests


def resolve_input_file(name_or_path: str) -> Path:
    candidate = Path(name_or_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    in_dir = INPUT_DIR / name_or_path
    if in_dir.is_file():
        return in_dir.resolve()
    if not name_or_path.endswith(".json"):
        in_dir = INPUT_DIR / f"{name_or_path}.json"
        if in_dir.is_file():
            return in_dir.resolve()
    raise FileNotFoundError(
        f"Fichier introuvable : {name_or_path!r} (cherché dans {INPUT_DIR})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Structure les JSON actualités FRMN au format html_results.",
    )
    parser.add_argument(
        "--file", "-f",
        metavar="NOM_OU_CHEMIN",
        help="Traiter un seul fichier (nom dans le dossier 2016 ou chemin).",
    )
    args = parser.parse_args()

    api_key = get_groq_api_key()
    if not api_key:
        print(f"[erreur] {GROQ_API_KEY_ENV} manquante dans processing/.env")
        return 1

    if not INPUT_DIR.is_dir():
        print(f"[erreur] Dossier introuvable : {INPUT_DIR}")
        return 1

    logger = setup_logger(LOG_FILE)
    client = Groq(api_key=api_key)

    single_file = bool(args.file)
    update_progress = not single_file

    progress = load_progress(PROGRESS_FILE)
    maybe_reset_daily_quota(progress)

    processed = set(progress.get("processed_files", []))
    requests_today = int(progress.get("requests_today", 0))

    if single_file:
        try:
            file_path = resolve_input_file(args.file)
        except FileNotFoundError as exc:
            print(f"[erreur] {exc}")
            return 1
        pending = [file_path]
        print(f"[test] Fichier unique : {file_path.name}")
    else:
        all_files = sorted(path for path in INPUT_DIR.glob("*.json") if path.is_file())
        already_in_output = (
            {path.name for path in OUTPUT_DIR.glob("*.json")}
            if OUTPUT_DIR.is_dir()
            else set()
        )
        all_file_names = {path.name for path in all_files}
        treated_union = (processed | already_in_output).intersection(all_file_names)
        pending: list[Path] = []
        skipped_progress = 0
        skipped_output = 0
        for file_path in all_files:
            in_progress = file_path.name in processed
            in_output = file_path.name in already_in_output
            if in_progress or in_output:
                reasons: list[str] = []
                if in_progress:
                    skipped_progress += 1
                    reasons.append("progress")
                if in_output:
                    skipped_output += 1
                    reasons.append("already_in_output")
                print(f"[skip] {file_path.name} ({', '.join(reasons)})")
                continue
            pending.append(file_path)
        if skipped_progress or skipped_output:
            print(
                "[info] Ignorés : "
                f"{skipped_progress} via progress, "
                f"{skipped_output} déjà en sortie."
            )
        print(f"[info] Restants non traités : {len(pending)}")
        if not pending:
            print("[info] Tous les fichiers sont déjà traités.")
            print("       Astuce : --file NOM.json pour un test unitaire.")
            return 0

    print("=" * 60)
    print(f"  Modèle               : {MODEL}")
    print(f"  Source               : {INPUT_DIR}")
    print(f"  Sortie               : {OUTPUT_DIR}")
    print(f"  Fichiers restants    : {len(pending)}")
    if not single_file:
        print(f"  Déjà traités         : {len(treated_union)}")
    print(f"  Requêtes aujourd'hui : {requests_today} / {DAILY_REQUEST_THRESHOLD}")
    print("=" * 60)

    ok_count = 0
    err_count = 0

    for index, file_path in enumerate(pending, start=1):
        if requests_today >= DAILY_REQUEST_THRESHOLD:
            print(f"\n[stop] Quota journalier atteint ({requests_today} requêtes).")
            break

        print(f"\n[{index}/{len(pending)}] {file_path.name}")

        if index > 1:
            print(f"  [attente {INTER_REQUEST_SLEEP:.0f}s] entre fichiers...")
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
            if update_progress:
                save_progress(PROGRESS_FILE, progress)
            print(f"  FATAL : {exc}")
            return 1
        except Exception as exc:
            status = f"Exception : {exc}"
            nb_req = 1

        requests_today += nb_req
        progress["requests_today"] = requests_today

        if status == "OK":
            ok_count += 1
            out_path = OUTPUT_DIR / file_path.name
            print(f"  ✓ OK | {used_tokens} tokens | écrit : {out_path}")
            if update_progress:
                processed.add(file_path.name)
                progress["processed_files"] = sorted(processed)
            logger.info(
                "%s | OK | tokens=%d | req_total=%d",
                file_path.name,
                used_tokens,
                requests_today,
            )
        else:
            err_count += 1
            logger.error("%s | ERREUR | %s", file_path.name, status)
            print(f"  ✗ ERREUR : {status}")

        if update_progress:
            save_progress(PROGRESS_FILE, progress)

    print("\n" + "=" * 60)
    print(f"  ✓ Succès  : {ok_count}")
    print(f"  ✗ Erreurs : {err_count}")
    print(f"  Requêtes  : {requests_today} / {DAILY_REQUEST_THRESHOLD}")
    print("=" * 60)

    if update_progress:
        save_progress(PROGRESS_FILE, progress)
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
