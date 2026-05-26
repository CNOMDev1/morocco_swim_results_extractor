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

from groq import Groq, RateLimitError, APIStatusError, APIConnectionError
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = "llama-3.1-8b-instant"

SCRIPT_DIR    = Path(__file__).resolve().parent
BASE_DIR      = SCRIPT_DIR.parent
INPUT_DIR     = BASE_DIR / "data" / "json_from_pdfs" / "pdfs_results"
OUTPUT_DIR    = BASE_DIR / "data" / "json_structures" / "json_update"
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

AGE_CATEGORIES = (
    "SENIORS", "JUNIORS", "CADETS", "MINIMES", "BENJAMINS", "POUSSINS",
)

SYSTEM_PROMPT = """Tu es un extracteur de résultats de natation FRMN (PDF OCR).
Analyse le texte fourni et retourne UNIQUEMENT un objet JSON valide, sans markdown, sans backticks, sans explication.

Schéma cible (identique aux fichiers data/html_results/) :
{schema}

Structure typique du PDF :
- Ligne 1 : "NOM COMPÉTITION - JJ/MM/AAAA - VILLE - Grand bassin"
- Ligne épreuve : "1.  50 m DOS DAMES" ou "7.  4 x  50 m 4 NAGES MIXTE"
- Catégories d'âge : BENJAMINS, MINIMES, CADETS, JUNIORS, SENIORS, POUSSINS (→ champ "tour")
- Bloc nageur : Nom, puis lignes Rang./Club/Temps/Année naissance/Nationalité/Points

Règles absolues :
- Ne pas inventer de nageurs : extraire UNIQUEMENT ce qui est dans le texte.
- Champ absent ou illisible → null (sauf "location" → chaîne vide "").
- SwimDate : date ISO YYYY-MM-DD (souvent la date en tête de page ou la dernière date du meeting).
- SwimYear : année entière déduite de SwimDate.
- Meet : nom de la compétition sans la date, la ville ni "Grand bassin".
- location : ville (ex. FES, CASABLANCA) ou "".
- Country : "MAR" sauf indication contraire explicite.
- Bassin FRMN "Grand bassin" / "Petit bassin" → Course "SCM", PoolLength 25 (pas LCM).
- Event : "{{distance}} {{stroke}} {{course}}" (ex. "50 DOS SCM", "100 FR SCM").
- Stroke : FR (nage libre), DOS, BR (brasse), PAP (papillon), 4N (4 nages individuel), REL (relais).
- Relais "4 x 50 m …" : Distance = 4 × distance d'un relais (ex. 200), Stroke = "REL".
- DAMES → Gender "F" ; MESSIEURS → Gender "M" pour chaque nageur de l'épreuve.
- tour : catégorie d'âge ou tour de compétition (ex. "BENJAMINS", "Finale A", "SENIORS Séries").
- Rank : entier si "1.", "2."… ; null si "NC.".
- SwimTime : temps affiché ; SwimTimeSeconds : conversion en secondes (ex. 1:05.11 → 65.11).
- Status : "OK" ; "NC" si NC. ; "DSQ" si Dsq/Disqualifié ; "DNF" si abandon ; "DNS" si Frf n.d. / n.d.
- Speed : distance (m) / SwimTimeSeconds, arrondi à 4 décimales, ou null.
- swimmer.Age : SwimYear - Year_of_birth si les deux sont connus, sinon null.
- Une entrée "epreuves" par combinaison (épreuve + tour/catégorie).
- Retourner uniquement le JSON brut.""".format(
    schema=json.dumps(TARGET_SCHEMA, ensure_ascii=False, indent=2),
)

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
    """Même épreuve dans plusieurs chunks → performances concaténées."""
    merged: dict[str, dict] = {}
    for epreuves in all_epreuves:
        for ep in epreuves:
            if not isinstance(ep, dict):
                continue
            key = _epreuve_key(ep)
            if key not in merged:
                merged[key] = {
                    "Event":      ep.get("Event"),
                    "Distance":   ep.get("Distance"),
                    "Stroke":     ep.get("Stroke"),
                    "Course":     ep.get("Course"),
                    "PoolLength": ep.get("PoolLength"),
                    "tour":       ep.get("tour"),
                    "performances": [],
                }
            perfs = ep.get("performances", [])
            if isinstance(perfs, list):
                merged[key]["performances"].extend(perfs)
    return list(merged.values())


def merge_metadata(chunks: list[dict]) -> dict[str, Any]:
    """Conserve les métadonnées de meeting du premier chunk non vide."""
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
        "Name":          _null_or_str(raw.get("Name")),
        "Gender":        gender,
        "Year_of_birth": year_of_birth,
        "Age":           age,
        "Nationality":   _null_or_str(raw.get("Nationality")),
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
        "Rank":            rank,
        "club":            _null_or_str(raw.get("club")),
        "SwimTime":        swim_time,
        "SwimTimeSeconds": swim_secs,
        "Status":          status,
        "Speed":           speed,
        "swimmer":         normalize_swimmer(raw.get("swimmer"), swim_year),
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
        "Event":        event,
        "Distance":     distance,
        "Stroke":       stroke,
        "Course":       course,
        "PoolLength":   pool_length,
        "tour":         _null_or_str(raw.get("tour")) or "",
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
    epreuves_in = raw.get("epreuves", [])
    epreuves_out: list[dict[str, Any]] = []
    for ep in (epreuves_in if isinstance(epreuves_in, list) else []):
        normalized = normalize_epreuve(ep, swim_year)
        if normalized is not None:
            epreuves_out.append(normalized)
    return {
        "SwimDate":  swim_date,
        "SwimYear":  swim_year,
        "Meet":      _null_or_str(raw.get("Meet")) or "",
        "location":  location,
        "Country":   country,
        "epreuves":  epreuves_out,
    }


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

    all_parsed:   list[dict] = []
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
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return "OK", total_tokens, nb_requests


def resolve_input_file(name_or_path: str) -> Path:
    """Résout un nom de fichier ou un chemin vers un JSON d'entrée."""
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
        f"Fichier introuvable : {name_or_path!r} "
        f"(cherché dans {INPUT_DIR})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Structure les JSON PDF (Groq) au format html_results.",
    )
    parser.add_argument(
        "--file", "-f",
        metavar="NOM_OU_CHEMIN",
        help="Traiter un seul fichier (nom dans pdfs_results ou chemin absolu).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Avec --file : réécrire même si déjà en sortie / dans progress.",
    )
    args = parser.parse_args()

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

    single_file = bool(args.file)
    update_progress = not single_file

    progress = load_progress(PROGRESS_FILE)
    maybe_reset_daily_quota(progress)

    processed      = set(progress.get("processed_files", []))
    requests_today = int(progress.get("requests_today", 0))

    if single_file:
        try:
            file_path = resolve_input_file(args.file)
        except FileNotFoundError as exc:
            print(f"[erreur] {exc}")
            return 1
        pending = [file_path]
        print(f"[test] Fichier unique : {file_path.name}")
        if args.force:
            print("[test] Mode --force : sortie et progress ignorés pour ce fichier.")
    else:
        all_files = sorted(p for p in INPUT_DIR.glob("*.json") if p.is_file())
        already_in_output = (
            {p.name for p in OUTPUT_DIR.glob("*.json")} if OUTPUT_DIR.is_dir() else set()
        )
        pending = [
            f for f in all_files
            if f.name not in processed and f.name not in already_in_output
        ]
        skipped = len(already_in_output.intersection({f.name for f in all_files}))
        if skipped:
            print(f"[info] {skipped} fichier(s) ignorés car déjà présents dans le dossier de sortie.")
        if not pending:
            print("[info] Tous les fichiers sont déjà traités. Rien à faire.")
            print("       Astuce : python processing/groq_structuration.py --file NOM.json --force")
            return 0

    print("=" * 60)
    print(f"  Modèle               : {MODEL}")
    print(f"  Chunk max            : {INITIAL_CHUNK_CHARS} chars")
    print(f"  Pause entre appels   : {INTER_REQUEST_SLEEP:.0f}s")
    print(f"  À traiter            : {len(pending)}")
    if not single_file:
        print(f"  Déjà traités         : {len(processed)}")
    print(f"  Requêtes aujourd'hui : {requests_today} / {DAILY_REQUEST_THRESHOLD}")
    print(f"  Sortie               : {OUTPUT_DIR}")
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
            out_path = OUTPUT_DIR / file_path.name
            print(f"  ✓ OK | {used_tokens} tokens | écrit : {out_path}")
            if update_progress:
                processed.add(file_path.name)
                progress["processed_files"] = sorted(processed)
            logger.info("%s | OK | tokens=%d | req_total=%d",
                        file_path.name, used_tokens, requests_today)
            if update_progress:
                print(f"  requêtes aujourd'hui : {requests_today}/{DAILY_REQUEST_THRESHOLD}")
        else:
            err_count += 1
            logger.error("%s | ERREUR | %s", file_path.name, status)
            print(f"  ✗ ERREUR : {status}")

        if update_progress:
            save_progress(PROGRESS_FILE, progress)

    remaining = len(pending) - ok_count - err_count
    print("\n" + "=" * 60)
    print(f"  ✓ Succès             : {ok_count}")
    print(f"  ✗ Erreurs            : {err_count}")
    if remaining > 0:
        print(f"  ⏸ Non traités        : {remaining} (quota atteint)")
    print(f"  Requêtes aujourd'hui : {requests_today} / {DAILY_REQUEST_THRESHOLD}")
    if not single_file:
        all_count = len(sorted(p for p in INPUT_DIR.glob("*.json") if p.is_file()))
        print(f"  Total traités        : {len(processed)} / {all_count}")
    print("=" * 60)

    if update_progress:
        save_progress(PROGRESS_FILE, progress)
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())