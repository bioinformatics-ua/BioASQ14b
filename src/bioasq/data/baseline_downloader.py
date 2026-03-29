"""
Two-stage PubMed baseline downloader and parser.

Stage 1: download -- Downloads all XML.gz files to year folders
Stage 2: parse   -- Parses downloaded files to JSONL

Usage:
    python pubmed_downloader.py download 2026
    python pubmed_downloader.py download 2017 2019 2026
    python pubmed_downloader.py parse 2026 -o output.jsonl
    python pubmed_downloader.py parse 2026 --workers 8
"""

import asyncio
import gzip
import random
import time
from datetime import UTC, datetime
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from typing import Annotated
from urllib.request import urlopen

import orjson
import pubmed_parser as pp
import requests
import typer
from lxml import etree

from bioasq.common import PROJECT_DATA_BASELINES_DIR
from bioasq.common.types import Document
from bioasq.data.database import add_baseline_ids, get_if_articles_exist, insert_articles
from bioasq.data.embeddings.generate_embeddings import process_chunk_async

app = typer.Typer(
    help="Two-stage PubMed baseline downloader and parser.",
    epilog="""
Examples:
  # Stage 1: Download files
  python -m bioasq.data.baseline_downloader download 2026
  python -m bioasq.data.baseline_downloader download 2017 2019 2026 -w 8

  # Stage 2: Parse files
  python -m bioasq.data.baseline_downloader parse 2026 -o output.jsonl
  python -m bioasq.data.baseline_downloader parse 2017 2019 2026 -o combined.jsonl -w 12

  # Custom output fields
  python -m bioasq.data.baseline_downloader parse 2026 -f pmid title abstract journal
        """.strip(),
)

DEFAULT_WORKERS = max(1, cpu_count() - 1)

# ============== PARSING FUNCTIONS ==============


def parse_mesh_terms(medline: etree.Element) -> str:
    """Parse MeSH terms from a Medline element."""
    if (mesh := medline.find("MeshHeadingList")) is not None:
        mesh_terms_list = []
        for m in mesh:
            name_node = m.find("DescriptorName")
            if name_node is not None:
                ui = name_node.attrib.get("UI", "") or ""
                text = name_node.text or ""
                mesh_terms_list.append(f"{ui}:{text}")
        return "; ".join(mesh_terms_list)
    return ""


def parse_pmid(medline: etree.Element) -> str:
    """Parse PMID from a Medline element."""
    if (pmid := medline.find("PMID")) is not None:
        return pmid.text or ""

    if (article_ids := medline.find("PubmedData/ArticleIdList")) is not None and (
        pmid := article_ids.find('ArticleId[@IdType="pmid"]')
    ) is not None:
        return (pmid.text or "").strip()

    return ""


def parse_article_info(medline: etree.Element, author_list: bool = False) -> dict:
    """Parse article information from a Medline element."""
    article = medline.find("Article")
    if article is None:
        return {}

    if (title := article.find("ArticleTitle")) is not None:
        title = pp.utils.stringify_children(title).strip()

    if article.find("Abstract/AbstractText") is not None:
        abstract_texts = article.findall("Abstract/AbstractText")
        if len(abstract_texts) > 1:
            abstract_list = []
            for abstract in abstract_texts:
                section = abstract.attrib.get("Label", "")
                if section != "UNASSIGNED":
                    abstract_list.append("\n" + section)
                abstract_list.append(pp.utils.stringify_children(abstract).strip())
            abstract = "\n".join(abstract_list).strip()
        else:
            abstract = pp.utils.stringify_children(abstract_texts[0]).strip() or ""
    elif article.find("Abstract") is not None:
        abstract = pp.utils.stringify_children(article.find("Abstract")).strip() or ""
    else:
        abstract = ""

    authors_dict = pp.medline_parser.parse_author_affiliation(medline)
    if not author_list:
        affiliations = ";".join(
            [
                author.get("affiliation", "")
                for author in authors_dict
                if author.get("affiliation", "") != ""
            ]
        )
        authors = ";".join(
            [
                f"{author.get('lastname', '')}|{author.get('forename', '')}|"
                f"{author.get('initials', '')}|{author.get('identifier', '')}"
                for author in authors_dict
            ]
        )
    else:
        authors = authors_dict
        affiliations = ""

    journal = article.find("Journal")
    journal_name = " ".join(journal.xpath("Title/text()")) if journal is not None else ""

    pmid = parse_pmid(medline)
    mesh_terms = parse_mesh_terms(medline)
    keywords = pp.medline_parser.parse_keywords(medline)

    result = {
        "title": title,
        "abstract": abstract,
        "journal": journal_name,
        "authors": authors,
        "pmid": pmid,
        "mesh_terms": mesh_terms,
        "keywords": keywords,
        "delete": False,
    }

    if not author_list:
        result["affiliations"] = affiliations

    return result


# ============== DOWNLOAD STAGE ==============

type PubmedLink = tuple[int, str, str]  # year, url, filename


def get_pubmed_links(years: list[int]) -> list[PubmedLink]:
    """Fetch all XML.gz links for given years."""
    links: list[PubmedLink] = []

    for year in years:
        print(f"Fetching index page for year {year}...")

        xpath = "/html/body/pre/a" if year >= 2025 else "/html/body/ul[2]/li/a"
        snapshot = f"{year}0216212530"
        is_current_year = datetime.now(UTC).year == year
        url = (
            f"{'' if is_current_year else 'https://web.archive.org/web/' + snapshot + '/'}"
            "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline"
            if year >= 2025
            else f"https://web.archive.org/web/20241219010838/https://lhncbc.nlm.nih.gov/ii/information/MBR/Baselines/{year}.html"
        )

        while True:
            try:
                print(f"Fetching index page for year {year} from {url}...")
                output: bytes = urlopen(url).read()
                break
            except Exception as e:
                print(f"Error fetching index page for year {year} from {url}: {e}")
                time.sleep(random.uniform(1, 10))

        tree = etree.fromstring(output.decode("utf-8"), etree.HTMLParser())

        for link_elem in tree.xpath(xpath):
            if not link_elem.text.endswith("xml.gz"):
                continue
            href: str = link_elem.get("href")
            if year >= 2025:
                # href is just the filename (e.g., "pubmed25n0001.xml.gz")
                # Construct full web archive download URL with if_ flag
                prefix = "" if is_current_year else f"https://web.archive.org/web/{snapshot}if_/"
                href = f"{prefix}https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/{href}"
            else:
                # Web archive: fix the URL format for older years
                href = href.replace("/https://data.lhncbc", "if_/https://data.lhncbc")
            links.append((year, href, link_elem.text))

    return links


type DownloadStats = tuple[int, int, int]  # downloaded, skipped, failed


def download_file_worker(
    worker_id: int,
    task_queue: Queue,
    base_dir: str,
    max_retries: int = 40,
) -> DownloadStats:
    """Worker: download files from queue."""
    downloaded = 0
    skipped = 0
    failed = 0

    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:
                task_queue.put(None)
                break
            year, url, filename = task
        except Exception as e:
            print(f"[{worker_id}] Error getting task: {e}")
            break

        if url is None:
            task_queue.put(None)
            break

        year_dir = Path(base_dir) / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        local_path = year_dir / filename

        # Skip if already exists
        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"[{worker_id}] SKIP: {filename} ({size_mb:.1f} MB)")
            skipped += 1
            continue

        print(f"[{worker_id}] DOWNLOAD: {filename}")

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                local_path.write_bytes(response.content)
                size_mb = len(response.content) / (1024 * 1024)
                downloaded += 1
                print(f"[{worker_id}] ✓ SAVED: {filename} ({size_mb:.1f} MB)")
                time.sleep(0.5)
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = random.uniform(2, 10) * (2**attempt)
                    print(f"[{worker_id}]   Retry in {wait_time:.1f}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"[{worker_id}] ✗ FAILED: {filename}: {e}")
                    failed += 1

    return downloaded, skipped, failed


@app.command("download")
def download_main(
    years: Annotated[list[int], typer.Argument(help="Years to download")],
    workers: Annotated[
        int,
        typer.Option("-w", "--workers", help="Parallel download workers"),
    ] = DEFAULT_WORKERS,
    base_dir: Annotated[
        Path,
        typer.Option("-d", "--dir", help="Base directory for downloads"),
    ] = PROJECT_DATA_BASELINES_DIR / "downloaded_baselines",
) -> None:
    """Download all files for given years."""
    print(f"Downloading PubMed baselines for years: {years}")
    print(f"Base directory: {base_dir}")
    print(f"Workers: {workers}\n")

    base_dir.mkdir(parents=True, exist_ok=True)

    links = get_pubmed_links(years)
    print(f"\nTotal files to download: {len(links)}\n")

    task_queue: Queue = Queue()
    for year, url, filename in links:
        task_queue.put((year, url, filename))
    task_queue.put(None)

    # Run workers
    worker_ps = [
        Process(target=download_file_worker, args=(i, task_queue, base_dir)) for i in range(workers)
    ]

    for wp in worker_ps:
        wp.start()
    for wp in worker_ps:
        wp.join()

    print("\n" + "=" * 50)
    print("Download stage complete!")
    print("=" * 50)


# ============== PARSE STAGE ==============


def parse_local_file(filepath: str, output_fields: list[str]) -> list[dict]:
    """Parse a local XML.gz file."""
    articles = []
    try:
        with gzip.open(filepath, "rb") as xml_fd:
            for _, element in etree.iterparse(xml_fd, events=("end",)):
                if element.tag == "MedlineCitation":
                    article = parse_article_info(element)
                    filtered = {k: article[k] for k in output_fields if k in article}
                    articles.append(filtered)
                    element.clear()
    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")

    return articles


def parse_worker(
    worker_id: int,
    task_queue: Queue,
    results_queue: Queue,
    output_fields: list[str],
) -> None:
    """Worker: parse files and put articles to results."""
    files_processed = 0

    while True:
        try:
            filepath = task_queue.get(timeout=1)
        except Exception as e:
            print(f"[{worker_id}] Error getting filepath: {e}")
            break

        if filepath is None:
            task_queue.put(None)
            break

        filename = Path(filepath).name
        print(f"[{worker_id}] PARSING: {filename}")
        articles = parse_local_file(filepath, output_fields)
        files_processed += 1

        results_queue.put(
            [
                Document(
                    pmid=article["pmid"], full_text=article["title"] + " " + article["abstract"]
                )
                for article in articles
            ]
        )

        print(
            f"[{worker_id}] ✓ {filename}: {len(articles)} articles. first pmid: {articles[0]['pmid']}"
        )

        time.sleep(20)

    results_queue.put(f"WORKER_{worker_id}_DONE:{files_processed}")


def writer_process(results_queue: Queue, output_file: str, num_workers: int) -> None:
    """Write results to JSONL."""
    total_articles = 0
    workers_done = 0
    total_files = 0

    with Path(output_file).open("w", encoding="utf-8") as f_out:
        while workers_done < num_workers:
            item = results_queue.get()

            if isinstance(item, str) and item.startswith("WORKER_"):
                workers_done += 1
                files_done = int(item.split(":")[1])
                total_files += files_done
            else:
                f_out.write(orjson.dumps(item) + b"\n")
                total_articles += 1
                if total_articles % 5000 == 0:
                    print(f"  Writer: {total_articles} articles...")

    print(f"\nWriter: {total_files} files → {total_articles} articles → {output_file}")


_DEFAULT_FIELDS = ("pmid", "title", "abstract")


async def db_process(results_queue: Queue, num_workers: int, embeddings: bool, year: int) -> None:
    """Process embeddings and insert into database."""
    total_articles = 0
    workers_done = 0
    total_files = 0

    while workers_done < num_workers:
        item = results_queue.get()
        print(f"PIXA Getting item: {len(item)}, first pmid: {item[0].pmid}")

        if isinstance(item, str) and item.startswith("WORKER_"):
            workers_done += 1
            files_done = int(item.split(":")[1])
            total_files += files_done
        else:
            total_articles += 1
            if embeddings:
                await process_chunk_async(item, insert_into_db="insert")
            else:
                existing = await get_if_articles_exist([int(doc.pmid) for doc in item])
                print(f"PIXA Adding {len(item) - len(existing)} baseline ids to database")
                await insert_articles([doc for doc in item if str(doc.pmid) not in existing])
                await add_baseline_ids(year, [int(doc.pmid) for doc in item])

    print(f"\nWriter: {total_files} files → {total_articles} articles")


@app.command("parse")
def parse_main(
    year: Annotated[int, typer.Argument(help="Year to parse")],
    output_file: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Output JSONL file"),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("-w", "--workers", help="Parallel parsing workers"),
    ] = DEFAULT_WORKERS,
    base_dir: Annotated[
        Path,
        typer.Option("-d", "--dir", help="Base directory with downloads"),
    ] = PROJECT_DATA_BASELINES_DIR / "downloaded_baselines",
    output_fields: Annotated[
        list[str] | None,
        typer.Option("-f", "--fields", help="Fields to include in output"),
    ] = None,
    embeddings: Annotated[
        bool,
        typer.Option("-e", "--embeddings", help="Generate embeddings"),
    ] = False,
    chunk_size: Annotated[
        int,
        typer.Option("-c", "--chunk-size", help="Chunk size for database processing"),
    ] = 5000,
) -> None:
    """Parse downloaded files."""
    if output_fields is None:
        output_fields = list(_DEFAULT_FIELDS)

    print(f"Parsing PubMed baselines for year: {year}")
    print(f"Base directory: {base_dir}")
    print(f"Workers: {workers}\n")

    # Collect all files
    year_dir = Path(base_dir) / str(year)
    if not year_dir.exists():
        print(f"Year directory {year_dir} does not exist! Run download first.")
        return
    files_to_parse = list(year_dir.glob("*.xml.gz"))

    if not files_to_parse:
        print("No files found! Run download first.")
        return

    print(f"Files to parse: {len(files_to_parse)}\n")

    task_queue: Queue[str | None] = Queue()
    for f in files_to_parse:
        task_queue.put(str(f))
    task_queue.put(None)

    results_queue: Queue[list[dict] | str] = Queue()

    worker_ps = [
        Process(target=parse_worker, args=(i, task_queue, results_queue, output_fields))
        for i in range(workers)
    ]

    # writer_p = Process(target=writer_process, args=(results_queue, output_file, workers))

    def db_process_wrapper(results_queue: Queue, workers: int, embeddings: bool, year: int) -> None:
        asyncio.run(db_process(results_queue, workers, embeddings, year))

    db_p = Process(target=db_process_wrapper, args=(results_queue, workers, embeddings, year))

    # writer_p.start()
    db_p.start()
    for wp in worker_ps:
        wp.start()

    for wp in worker_ps:
        wp.join()
    # writer_p.join()
    db_p.join()

    print("\n" + "=" * 50)
    print("Parse stage complete!")
    print("=" * 50)


if __name__ == "__main__":
    app()
