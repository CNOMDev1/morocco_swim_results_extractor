#!/usr/bin/env python3
"""
Fichier appelant qui importe et réutilise les fonctions de groq_structuration.py

Usage:
    python run_structuration.py
    python run_structuration.py --files "fichier1.json" --debug
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Charge automatiquement le fichier .env situé dans le même dossier que ce script
load_dotenv(Path(__file__).resolve().parent / ".env")

# ── Import des fonctions et constantes du script principal ──────────────────
from groq_structuration import (
    # Client / config
    Groq,
    DEFAULT_MODEL,
    INITIAL_CHUNK_CHARS,
    DAILY_REQUEST_THRESHOLD,
    INTER_REQUEST_SLEEP,

    # Utilitaires de progression
    load_progress,
    save_progress,
    maybe_reset_daily_quota,

    # Extraction & traitement
    extract_text,
    infer_source_filename,
    split_into_chunks,
    merge_tables,
    normalize_output,

    # Appel API
    call_groq_chunk,

    # Traitement complet d'un fichier
    process_file,

    # Logger
    setup_logger,
)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration — modifie ces chemins selon ton environnement
# ══════════════════════════════════════════════════════════════════════════════

INPUT_DIR     = Path("/Users/nouhailaimaneabbassi/Desktop/SwimResultsExtractor/data/json_from_pdfs/pdfs_results")
OUTPUT_DIR    = Path("/Users/nouhailaimaneabbassi/Desktop/SwimResultsExtractor/data/json_structures")
PROGRESS_FILE = Path(__file__).resolve().parent / "progress_groq.json"
ERRORS_DIR    = Path(__file__).resolve().parent / "errors"
LOG_FILE      = Path(__file__).resolve().parent / "processing_groq.log"

# Paramètres d'exécution
MODEL           = DEFAULT_MODEL
INTER_SLEEP     = INTER_REQUEST_SLEEP   # secondes entre appels API
DAILY_THRESHOLD = DAILY_REQUEST_THRESHOLD
DEBUG           = False
MAX_FILES       = 0                     # 0 = tous les fichiers
SPECIFIC_FILES  = []                    # ex: ["fichier1.json", "fichier2.json"]


# ══════════════════════════════════════════════════════════════════════════════
# Exemples d'utilisation des fonctions importées
# ══════════════════════════════════════════════════════════════════════════════

def exemple_lire_texte(json_path: Path) -> str:
    """Extrait uniquement le texte brut d'un fichier JSON OCR."""
    import json
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    text = extract_text(payload)
    print(f"[exemple_lire_texte] {len(text)} caractères extraits de {json_path.name}")
    return text


def exemple_chunker(json_path: Path) -> list[str]:
    """Découpe le texte d'un fichier en chunks et affiche un résumé."""
    text   = exemple_lire_texte(json_path)
    chunks = split_into_chunks(text, INITIAL_CHUNK_CHARS)
    print(f"[exemple_chunker] {len(chunks)} chunk(s) de ~{INITIAL_CHUNK_CHARS} chars max")
    for i, c in enumerate(chunks, 1):
        print(f"  chunk {i} : {len(c)} chars")
    return chunks


def exemple_appel_api(json_path: Path) -> None:
    """Appelle l'API Groq sur le premier chunk d'un fichier et affiche la réponse."""
    import json

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[erreur] GROQ_API_KEY manquante.")
        return

    client = Groq(api_key=api_key)
    chunks = exemple_chunker(json_path)

    if not chunks:
        print("[avertissement] Aucun chunk à traiter.")
        return

    raw_response, tokens = call_groq_chunk(
        client      = client,
        model_name  = MODEL,
        chunk       = chunks[0],
        chunk_label = "chunk 1/1 (exemple)",
        inter_sleep = INTER_SLEEP,
        debug       = True,
    )
    print(f"[exemple_appel_api] {tokens} tokens utilisés")
    try:
        parsed = json.loads(raw_response)
        print("[exemple_appel_api] JSON reçu :", json.dumps(parsed, ensure_ascii=False, indent=2)[:500])
    except json.JSONDecodeError:
        print("[exemple_appel_api] Réponse brute :", raw_response[:300])


def traiter_fichiers_specifiques(fichiers: list[str]) -> None:
    """Traite uniquement une liste de fichiers donnés par leur nom."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[erreur] GROQ_API_KEY manquante.")
        return

    client   = Groq(api_key=api_key)
    logger   = setup_logger(LOG_FILE)
    progress = load_progress(PROGRESS_FILE)
    maybe_reset_daily_quota(progress)

    requests_today = int(progress.get("requests_today", 0))

    for name in fichiers:
        file_path = INPUT_DIR / name
        if not file_path.exists():
            print(f"[avertissement] Fichier introuvable : {name}")
            continue

        print(f"\n→ Traitement de {name}")
        status, tokens, nb_req = process_file(
            file_path        = file_path,
            output_dir       = OUTPUT_DIR,
            errors_dir       = ERRORS_DIR,
            client           = client,
            model_name       = MODEL,
            inter_sleep      = INTER_SLEEP,
            debug            = DEBUG,
            requests_today   = requests_today,
            daily_threshold  = DAILY_THRESHOLD,
        )
        requests_today += nb_req
        progress["requests_today"] = requests_today

        if status == "OK":
            progress.setdefault("processed_files", [])
            if name not in progress["processed_files"]:
                progress["processed_files"].append(name)
            logger.info("%s | OK | tokens=%d", name, tokens)
            print(f"  ✓ OK — {tokens} tokens")
        else:
            logger.error("%s | ERREUR | %s", name, status)
            print(f"  ✗ ERREUR : {status}")

        save_progress(PROGRESS_FILE, progress)


def traiter_tous_les_fichiers() -> None:
    """Traite tous les fichiers non encore traités dans INPUT_DIR."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[erreur] GROQ_API_KEY manquante.")
        return

    client   = Groq(api_key=api_key)
    logger   = setup_logger(LOG_FILE)
    progress = load_progress(PROGRESS_FILE)
    maybe_reset_daily_quota(progress)

    processed      = set(progress.get("processed_files", []))
    requests_today = int(progress.get("requests_today", 0))
    already_done   = {p.name for p in OUTPUT_DIR.glob("*.json")} if OUTPUT_DIR.is_dir() else set()

    all_files = sorted(p for p in INPUT_DIR.glob("*.json") if p.is_file())
    pending   = [f for f in all_files if f.name not in processed and f.name not in already_done]

    if MAX_FILES > 0:
        pending = pending[:MAX_FILES]

    print(f"[info] {len(pending)} fichier(s) à traiter sur {len(all_files)} total.")

    for i, file_path in enumerate(pending, 1):
        if requests_today >= DAILY_THRESHOLD:
            print(f"\n[stop] Quota journalier atteint ({requests_today} req).")
            break

        print(f"\n[{i}/{len(pending)}] {file_path.name}")

        status, tokens, nb_req = process_file(
            file_path        = file_path,
            output_dir       = OUTPUT_DIR,
            errors_dir       = ERRORS_DIR,
            client           = client,
            model_name       = MODEL,
            inter_sleep      = INTER_SLEEP,
            debug            = DEBUG,
            requests_today   = requests_today,
            daily_threshold  = DAILY_THRESHOLD,
        )
        requests_today += nb_req
        progress["requests_today"] = requests_today

        if status == "OK":
            processed.add(file_path.name)
            progress["processed_files"] = sorted(processed)
            logger.info("%s | OK | tokens=%d", file_path.name, tokens)
            print(f"  ✓ OK — {tokens} tokens")
        else:
            logger.error("%s | ERREUR | %s", file_path.name, status)
            print(f"  ✗ ERREUR : {status}")

        save_progress(PROGRESS_FILE, progress)

    print(f"\n[fin] {len(processed)} fichier(s) traités au total.")


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Choix du mode d'exécution ─────────────────────────────────────────────

    if SPECIFIC_FILES:
        # Mode 1 : traiter uniquement les fichiers listés dans SPECIFIC_FILES
        traiter_fichiers_specifiques(SPECIFIC_FILES)

    else:
        # Mode 2 : traiter tous les fichiers non encore traités
        traiter_tous_les_fichiers()

    # ── Exemples d'utilisation des fonctions bas niveau ───────────────────────
    # Décommente l'une des lignes ci-dessous pour tester une fonction précise :

    # exemple_lire_texte(INPUT_DIR / "mon_fichier.json")
    # exemple_chunker(INPUT_DIR / "mon_fichier.json")
    # exemple_appel_api(INPUT_DIR / "mon_fichier.json")