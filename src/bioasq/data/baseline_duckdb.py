"""DuckDB based baseline indexer and retriever.

Replaces JSONL collections with high-performance DuckDB tables and indices.
"""

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Annotated

import duckdb
import typer

try:
    import ijson

    _HAS_IJSON = True
except ImportError:
    ijson = None  # type: ignore[assignment]
    _HAS_IJSON = False

logger = logging.getLogger(__name__)

app = typer.Typer(help="DuckDB Baseline Commands")


def init_pubmed_db(jsonl_path: Path | str, db_path: Path | str) -> duckdb.DuckDBPyConnection:
    """Creates the DuckDB database from JSONL, building the articles table and PMID index."""
    con = duckdb.connect(str(db_path))

    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name = 'articles'"
    ).fetchall()

    if not tables:
        logger.info(f"Creating database from {jsonl_path}... This will save hours in the future!")

        query = f"""
            CREATE TABLE articles AS
            SELECT pmid, title, abstract
            FROM (
                SELECT
                    CAST(pmid AS VARCHAR) as pmid,
                    CAST(title AS VARCHAR) as title,
                    CAST(abstract AS VARCHAR) as abstract,
                    ROW_NUMBER() OVER (PARTITION BY CAST(pmid AS VARCHAR) ORDER BY 1) as rn
                FROM read_json_auto('{jsonl_path}')
            ) sub
            WHERE rn = 1
        """
        con.execute(query)

        logger.info("Creating unique index on PMID...")
        con.execute("CREATE UNIQUE INDEX idx_pmid ON articles(pmid)")
        logger.info("DuckDB database created successfully!")
    else:
        logger.info("DuckDB database already exists. Connecting directly...")

    return con


def get_article(con: duckdb.DuckDBPyConnection, pmid: str) -> dict[str, str] | None:
    """Returns the title and abstract for a single PMID."""
    query = "SELECT title, abstract FROM articles WHERE pmid = ?"
    resultado = con.execute(query, [pmid]).fetchone()

    if resultado:
        return {"title": resultado[0], "abstract": resultado[1]}
    return None


def get_articles_batch(con: duckdb.DuckDBPyConnection, pmids: Sequence[str]) -> dict[str, str]:
    """Fetches hundreds of PMIDs at once and returns concatenated texts {pmid: "title abstract"}."""
    if not pmids:
        return {}

    placeholders = ",".join(["?"] * len(pmids))
    query = f"""
        SELECT pmid, concat_ws(' ', title, abstract)
        FROM articles
        WHERE pmid IN ({placeholders})
    """
    resultados = con.execute(query, list(pmids)).fetchall()
    return {str(linha[0]): str(linha[1]) for linha in resultados}


def get_total_articles(con: duckdb.DuckDBPyConnection) -> int:
    """Returns the total number of articles loaded in the database."""
    resultado = con.execute("SELECT COUNT(*) FROM articles").fetchone()
    return int(resultado[0]) if resultado else 0


def init_lookup_table(con: duckdb.DuckDBPyConnection, json_path: Path | str) -> None:
    """Creates the 'lookup' table in DuckDB from the JSON file."""
    if not _HAS_IJSON:
        raise ImportError(
            "The 'ijson' package is required for loading large lookups. "
            "Install with: pip install ijson"
        )

    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name = 'lookup'"
    ).fetchall()

    if not tables:
        logger.info(f"Creating lookup table from {json_path}... This might take a while.")

        con.execute(
            """
            CREATE TABLE lookup (
                source_pmid VARCHAR,
                target_pmid VARCHAR,
                score DOUBLE
            )
            """
        )

        def iter_lookup_rows() -> Iterator[tuple[str, str, float]]:
            with Path(json_path).open("rb") as f:
                for source_pmid, entries in ijson.kvitems(f, ""):
                    source = str(source_pmid)
                    for pair in entries:
                        yield (source, str(pair[0]), float(pair[1]))

        con.executemany("INSERT INTO lookup VALUES (?, ?, ?)", iter_lookup_rows())

        logger.info("Creating index on source_pmid...")
        con.execute("CREATE INDEX idx_lookup_source ON lookup(source_pmid)")
        logger.info("Lookup table created successfully!")
    else:
        logger.info("Lookup table already exists.")


def get_lookup_entries(
    con: duckdb.DuckDBPyConnection,
    pmid: str,
    topk: int | None = None,
    min_score: float | None = None,
) -> list[tuple[str, float]]:
    """Returns the list [(pmid, score), ...] for a given source_pmid."""
    conditions = ["source_pmid = ?"]
    params: list[object] = [pmid]
    if min_score is not None:
        conditions.append("score >= ?")
        params.append(min_score)

    query = f"""
        SELECT target_pmid, score
        FROM lookup
        WHERE {" AND ".join(conditions)}
        ORDER BY score DESC
        {"LIMIT ?" if topk is not None else ""}
    """
    if topk is not None:
        params.append(topk)

    resultados = con.execute(query, params).fetchall()
    return [(str(r[0]), float(r[1])) for r in resultados]


def get_lookup_entries_batch(
    con: duckdb.DuckDBPyConnection,
    pmids: Sequence[str],
    topk: int | None = None,
    min_score: float | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Gets lookups for multiple PMIDs at once."""
    if not pmids:
        return {}

    out: dict[str, list[tuple[str, float]]] = {str(p): [] for p in pmids}

    placeholders = ",".join(["?"] * len(pmids))
    params: list[object] = list(pmids)
    if min_score is not None:
        params.append(min_score)

    score_filter = "AND score >= ?" if min_score is not None else ""

    query = f"""
        SELECT source_pmid, target_pmid, score
        FROM lookup
        WHERE source_pmid IN ({placeholders})
        {score_filter}
        ORDER BY source_pmid, score DESC
    """
    resultados = con.execute(query, params).fetchall()

    for source, target, score in resultados:
        source_str = str(source)
        if source_str in out and (topk is None or len(out[source_str]) < topk):
            out[source_str].append((str(target), float(score)))

    return out


@app.command(name="init")
def init_cmd(
    jsonl_path: Annotated[
        Path, typer.Argument(help="The path to the JSONL file containing the articles.")
    ] = Path("./baselines/pubmed_baseline_2026.jsonl"),
    db_path: Annotated[Path, typer.Argument(help="The path to the DuckDB database file.")] = Path(
        "./db_baselines/pubmed_baseline_2026.db"
    ),
) -> None:
    """Initializes the PubMed articles DuckDB database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_pubmed_db(jsonl_path, db_path)


@app.command(name="init-lookup")
def init_lookup_cmd(
    lookup_json: Annotated[
        Path, typer.Argument(help="Path to the lookup JSON file ({pmid: [[pmid, score], ...]}).")
    ],
    db_path: Annotated[
        Path, typer.Argument(help="Path to the DuckDB database file (same as articles).")
    ] = Path("./db_baselines/pubmed_baseline_2026.db"),
) -> None:
    """Creates the 'lookup' table in DuckDB database from the JSON file."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    init_lookup_table(con, lookup_json)
    con.close()


@app.command(name="add-articles")
def add_articles_to_db_cmd(
    jsonl_path: Annotated[
        Path, typer.Argument(help="The path to the JSONL file containing the articles.")
    ] = Path("./baselines/pubmed_baseline_2026.jsonl"),
    db_path: Annotated[Path, typer.Argument(help="The path to the DuckDB database file.")] = Path(
        "./db_baselines/pubmed_baseline_2026.db"
    ),
) -> None:
    """Adds articles to the DuckDB database from the JSONL."""
    con = duckdb.connect(str(db_path))
    con.execute(
        f"""
        INSERT INTO articles
            SELECT
                CAST(pmid AS VARCHAR) as pmid,
                CAST(title AS VARCHAR) as title,
                CAST(abstract AS VARCHAR) as abstract
            FROM read_json_auto('{jsonl_path}')
        """
    )
    con.close()
