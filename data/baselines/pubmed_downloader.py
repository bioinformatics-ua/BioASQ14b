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

import gzip
import json
import random
import time
import argparse
from pathlib import Path
from urllib.request import urlopen
from multiprocessing import Process, Queue, cpu_count

import requests
from lxml import etree
import pubmed_parser as pp


# ============== PARSING FUNCTIONS ==============


def parse_mesh_terms(medline: etree.Element) -> str:
    """Parse MeSH terms from a Medline element."""
    if (mesh := medline.find("MeshHeadingList")) is not None:
        mesh_terms_list = [
            (m.find("DescriptorName").attrib.get("UI", "") or "")  # ty:ignore[unresolved-attribute]
            + ":"
            + (m.find("DescriptorName").text or "")  # ty:ignore[unresolved-attribute]
            for m in mesh
        ]
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
                f"{author.get('lastname', '')}|{author.get('forename', '')}|{author.get('initials', '')}|{author.get('identifier', '')}"
                for author in authors_dict
            ]
        )
    else:
        authors = authors_dict
        affiliations = ""

    journal = article.find("Journal")
    journal_name = (
        " ".join(journal.xpath("Title/text()")) if journal is not None else ""
    )

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
        url = (
            f"https://web.archive.org/web/{snapshot}/https://ftp.ncbi.nlm.nih.gov/pubmed/baseline"
            if year >= 2025
            else f"https://web.archive.org/web/20241219010838/https://lhncbc.nlm.nih.gov/ii/information/MBR/Baselines/{year}.html"
        )

        output: bytes = urlopen(url).read()
        print(f"Fetching index page for year {year} from {url}")

        tree = etree.fromstring(output.decode("utf-8"), etree.HTMLParser())

        for link_elem in tree.xpath(xpath):
            if not link_elem.text.endswith("xml.gz"):
                continue
            href: str = link_elem.get("href")
            if year >= 2025:
                # href is just the filename (e.g., "pubmed25n0001.xml.gz")
                # Construct full web archive download URL with if_ flag
                href = f"https://web.archive.org/web/{snapshot}if_/https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/{href}"
            else:
                # Web archive: fix the URL format for older years
                href = href.replace("/https://data.lhncbc", "if_/https://data.lhncbc")
            links.append((year, href, link_elem.text))

    return links


type DownloadStats = tuple[int, int, int]  # downloaded, skipped, failed


def download_file_worker(
    worker_id: int,
    task_queue: Queue[PubmedLink | None],
    base_dir: str,
    max_retries: int = 10,
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


def download_main(years: list[int], workers: int, base_dir: str) -> None:
    """Download all files for given years."""
    print(f"Downloading PubMed baselines for years: {years}")
    print(f"Base directory: {base_dir}")
    print(f"Workers: {workers}\n")

    links = get_pubmed_links(years)
    print(f"\nTotal files to download: {len(links)}\n")

    task_queue: Queue[PubmedLink | None] = Queue()
    for year, url, filename in links:
        task_queue.put((year, url, filename))
    task_queue.put(None)

    # Run workers
    worker_ps = [
        Process(target=download_file_worker, args=(i, task_queue, base_dir))
        for i in range(workers)
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
    task_queue: Queue[str | None],
    results_queue: Queue[dict | str],
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

        for article in articles:
            results_queue.put(article)

        print(f"[{worker_id}] ✓ {filename}: {len(articles)} articles")

    results_queue.put(f"WORKER_{worker_id}_DONE:{files_processed}")


def writer_process(
    results_queue: Queue[dict | str], output_file: str, num_workers: int
) -> None:
    """Write results to JSONL."""
    total_articles = 0
    workers_done = 0
    total_files = 0

    with open(output_file, "w") as f_out:
        while workers_done < num_workers:
            item = results_queue.get()

            if isinstance(item, str) and item.startswith("WORKER_"):
                workers_done += 1
                files_done = int(item.split(":")[1])
                total_files += files_done
            else:
                f_out.write(json.dumps(item) + "\n")
                total_articles += 1
                if total_articles % 5000 == 0:
                    print(f"  Writer: {total_articles} articles...")

    print(f"\nWriter: {total_files} files → {total_articles} articles → {output_file}")


def parse_main(
    years: list[int],
    workers: int,
    output_file: str,
    base_dir: str,
    output_fields: list[str],
) -> None:
    """Parse downloaded files."""
    print(f"Parsing PubMed baselines for years: {years}")
    print(f"Base directory: {base_dir}")
    print(f"Output: {output_file}")
    print(f"Workers: {workers}\n")

    # Collect all files
    files_to_parse = []
    for year in years:
        year_dir = Path(base_dir) / str(year)
        if year_dir.exists():
            files_to_parse.extend(year_dir.glob("*.xml.gz"))

    if not files_to_parse:
        print("No files found! Run download first.")
        return

    print(f"Files to parse: {len(files_to_parse)}\n")

    task_queue: Queue[str | None] = Queue()
    for f in files_to_parse:
        task_queue.put(str(f))
    task_queue.put(None)

    results_queue: Queue[dict | str] = Queue()

    worker_ps = [
        Process(target=parse_worker, args=(i, task_queue, results_queue, output_fields))
        for i in range(workers)
    ]

    writer_p = Process(
        target=writer_process, args=(results_queue, output_file, workers)
    )

    writer_p.start()
    for wp in worker_ps:
        wp.start()

    for wp in worker_ps:
        wp.join()
    writer_p.join()

    print("\n" + "=" * 50)
    print("Parse stage complete!")
    print("=" * 50)


# ============== MAIN ==============


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage PubMed baseline downloader and parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1: Download files
  python pubmed_downloader.py download 2026
  python pubmed_downloader.py download 2017 2019 2026 -w 8

  # Stage 2: Parse files
  python pubmed_downloader.py parse 2026 -o output.jsonl
  python pubmed_downloader.py parse 2017 2019 2026 -o combined.jsonl -w 12

  # Custom output fields
  python pubmed_downloader.py parse 2026 -f pmid title abstract journal
        """.strip(),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Download subcommand
    download_parser = subparsers.add_parser("download", help="Download XML.gz files")
    download_parser.add_argument("years", nargs="+", type=int, help="Years to download")
    download_parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help=f"Parallel download workers (default: {max(1, cpu_count() - 1)})",
    )
    download_parser.add_argument(
        "-d",
        "--dir",
        default="downloaded_baselines",
        help="Base directory for downloads (default: downloaded_baselines)",
    )

    # Parse subcommand
    parse_parser = subparsers.add_parser(
        "parse", help="Parse downloaded files to JSONL"
    )
    parse_parser.add_argument("years", nargs="+", type=int, help="Years to parse")
    parse_parser.add_argument("-o", "--output", required=True, help="Output JSONL file")
    parse_parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help=f"Parallel parsing workers (default: {max(1, cpu_count() - 1)})",
    )
    parse_parser.add_argument(
        "-d",
        "--dir",
        default="downloaded_baselines",
        help="Base directory with downloads (default: downloaded_baselines)",
    )
    parse_parser.add_argument(
        "-f",
        "--fields",
        nargs="+",
        default=["pmid", "title", "abstract"],
        help="Fields to include in output",
    )

    args = parser.parse_args()

    if args.command == "download":
        download_main(args.years, args.workers, args.dir)
    elif args.command == "parse":
        parse_main(args.years, args.workers, args.output, args.dir, args.fields)


if __name__ == "__main__":
    main()
