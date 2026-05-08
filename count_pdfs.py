from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:  # Compatibilite avec anciens environnements
    from PyPDF2 import PdfReader  # type: ignore[import-not-found]


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

DIRECTORIES = [
    DATA_DIR / "pdfs_results_actualites",
    DATA_DIR / "excel_results",
    DATA_DIR / "json_from_pdfs",
    DATA_DIR / "pdfs_results",
]

PDF_SOURCE_DIRECTORIES = [
    DATA_DIR / "pdfs_results",
    DATA_DIR / "pdfs_results_actualites",
]


def count_pdfs_and_pages(directory: Path) -> tuple[int, int, int]:
    if not directory.exists() or not directory.is_dir():
        return 0, 0, 0

    pdf_files = [
        file
        for file in directory.rglob("*")
        if file.is_file() and file.suffix.lower() == ".pdf"
    ]

    page_count = 0
    unreadable_count = 0
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(str(pdf_file))
            page_count += len(reader.pages)
        except Exception:
            unreadable_count += 1

    return len(pdf_files), page_count, unreadable_count


def count_files_by_suffix(directory: Path, suffixes: set[str]) -> int:
    if not directory.exists() or not directory.is_dir():
        return 0
    return sum(
        1
        for file in directory.rglob("*")
        if file.is_file() and file.suffix.lower() in suffixes
    )


def main() -> None:
    total_pdfs = 0
    total_pages = 0
    total_unreadable = 0
    source_total_pdfs = 0
    source_total_pages = 0
    source_total_unreadable = 0
    total_selected_files = 0
    selected_suffixes = {".pdf", ".xlsx", ".xls", ".json"}

    for directory in DIRECTORIES:
        selected_count = count_files_by_suffix(directory, selected_suffixes)
        total_selected_files += selected_count

        pdf_count, page_count, unreadable_count = count_pdfs_and_pages(directory)
        total_pdfs += pdf_count
        total_pages += page_count
        total_unreadable += unreadable_count

        if directory in PDF_SOURCE_DIRECTORIES:
            source_total_pdfs += pdf_count
            source_total_pages += page_count
            source_total_unreadable += unreadable_count

        print(
            f"{directory}: {selected_count} fichier(s) total "
            f"(PDF/Excel/JSON) | "
            f"{pdf_count} fichier(s) PDF | "
            f"{page_count} page(s) | "
            f"{unreadable_count} illisible(s)"
        )

    print(
        "\nTotal PDF sources (pdfs_results + pdfs_results_actualites): "
        f"{source_total_pdfs} fichier(s) PDF"
    )
    print(f"Pages PDF sources (lisibles): {source_total_pages} page(s)")
    if source_total_unreadable:
        print(f"PDF sources illisibles: {source_total_unreadable}")

    print(f"\nTotal global (PDF/Excel/JSON): {total_selected_files} fichier(s)")
    print(f"Total global PDF: {total_pdfs} fichier(s)")
    print(f"Total pages (lisibles): {total_pages} page(s)")
    if total_unreadable:
        print(f"Fichiers PDF illisibles: {total_unreadable}")


if __name__ == "__main__":
    main()
