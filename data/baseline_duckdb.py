import duckdb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import typer

try:
    import ijson
    _HAS_IJSON = True
except ImportError:
    ijson = None  # type: ignore[assignment]
    _HAS_IJSON = False


def init_pubmed_db(jsonl_path: Path | str, db_path: Path | str) -> duckdb.DuckDBPyConnection:
    """
    1. Cria a base de dados DuckDB a partir do JSONL (se não existir).
    Cria a tabela 'articles' e o índice no 'pmid'.
    """
    con = duckdb.connect(str(db_path))
    
    # Verifica se a tabela já existe
    tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_name = 'articles'").fetchall()
    
    if not tables:
        print(f"A criar a base de dados a partir de {jsonl_path}...")
        print("Isto vai demorar alguns minutos na primeira vez, mas poupa horas no futuro!")
        
        # Lemos apenas pmid, title e abstract. Ignoramos mesh_terms e keywords para poupar espaço.
        # Deduplicamos por PMID (mantemos a primeira ocorrência) para permitir índice único.
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
        
        print("A criar índice único no PMID...")
        con.execute("CREATE UNIQUE INDEX idx_pmid ON articles(pmid)")
        print("Base de dados DuckDB criada com sucesso!")
    else:
        print("Base de dados DuckDB já existe. A ligar diretamente...")
        
    return con


def get_article(con: duckdb.DuckDBPyConnection, pmid: str) -> Optional[Dict[str, str]]:
    """
    2. Obtém o título e o abstract para um único PMID.
    Retorna um dicionário ou None se não existir.
    """
    query = "SELECT title, abstract FROM articles WHERE pmid = ?"
    resultado = con.execute(query, [pmid]).fetchone()
    
    if resultado:
        return {"title": resultado[0], "abstract": resultado[1]}
    return None


def get_articles_batch(con: duckdb.DuckDBPyConnection, pmids: List[str]) -> Dict[str, str]:
    """
    3. Utilidade vital para o teu PyTorch DataLoader: 
    Vai buscar dezenas/centenas de PMIDs de uma só vez e já devolve o texto concatenado.
    """
    if not pmids:
        return {}
        
    # Cria os placeholders (?, ?, ?) correspondentes ao número de PMIDs
    placeholders = ','.join(['?'] * len(pmids))
    
    # concat_ws(' ', title, abstract) junta os dois com um espaço no meio nativamente no C++
    query = f"""
        SELECT pmid, concat_ws(' ', title, abstract) 
        FROM articles 
        WHERE pmid IN ({placeholders})
    """
    
    # Retorna num formato dicionário {pmid: "title abstract"}
    resultados = con.execute(query, pmids).fetchall()
    return {linha[0]: linha[1] for linha in resultados}

def get_total_articles(con: duckdb.DuckDBPyConnection) -> int:
    """
    Utilitário extra: Devolve o total de artigos carregados na base de dados.
    """
    resultado = con.execute("SELECT COUNT(*) FROM articles").fetchone()
    return resultado[0] if resultado else 0


def init_lookup_table(
    con: duckdb.DuckDBPyConnection,
    json_path: Path | str,
) -> None:
    """
    Cria a tabela 'lookup' na base de dados DuckDB a partir do JSON.
    O lookup é um dicionário {pmid: [(pmid1, score), (pmid2, score), ...]}.

    A tabela tem colunas: source_pmid, target_pmid, score.
    Requer o pacote 'ijson' para ficheiros grandes (pip install ijson).
    """
    if not _HAS_IJSON:
        raise ImportError(
            "O pacote 'ijson' é necessário para carregar lookups grandes. "
            "Instale com: pip install ijson"
        )

    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name = 'lookup'"
    ).fetchall()

    if not tables:
        print(f"A criar tabela lookup a partir de {json_path}...")
        print("Isto pode demorar para ficheiros grandes (streaming em progresso)...")

        con.execute("""
            CREATE TABLE lookup (
                source_pmid VARCHAR,
                target_pmid VARCHAR,
                score DOUBLE
            )
        """)

        def iter_lookup_rows():
            with open(json_path, "rb") as f:
                for source_pmid, entries in ijson.kvitems(f, ""):
                    source = str(source_pmid)
                    for pair in entries:
                        target_pmid = str(pair[0])
                        score = float(pair[1])
                        yield (source, target_pmid, score)

        con.executemany(
            "INSERT INTO lookup VALUES (?, ?, ?)",
            iter_lookup_rows(),
        )

        print("A criar índice no source_pmid...")
        con.execute("CREATE INDEX idx_lookup_source ON lookup(source_pmid)")
        print("Tabela lookup criada com sucesso!")
    else:
        print("Tabela lookup já existe.")


def get_lookup_entries(
    con: duckdb.DuckDBPyConnection,
    pmid: str,
    topk: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[Tuple[str, float]]:
    """
    Obtém a lista [(pmid, score), ...] para um source_pmid.
    """
    conditions = ["source_pmid = ?"]
    params: List[object] = [pmid]
    if min_score is not None:
        conditions.append("score >= ?")
        params.append(min_score)

    query = f"""
        SELECT target_pmid, score
        FROM lookup
        WHERE {' AND '.join(conditions)}
        ORDER BY score DESC
        {'LIMIT ?' if topk is not None else ''}
    """
    if topk is not None:
        params.append(topk)

    resultados = con.execute(query, params).fetchall()
    return [(r[0], r[1]) for r in resultados]


def get_lookup_entries_batch(
    con: duckdb.DuckDBPyConnection,
    pmids: List[str],
    topk: Optional[int] = None,
    min_score: Optional[float] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Obtém lookups para múltiplos PMIDs de uma vez.
    Retorna {pmid: [(pmid1, score), (pmid2, score), ...], ...}
    """
    if not pmids:
        return {}

    out: Dict[str, List[Tuple[str, float]]] = {p: [] for p in pmids}

    placeholders = ",".join(["?"] * len(pmids))
    params: List[object] = list(pmids)
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
        if source in out:
            if topk is None or len(out[source]) < topk:
                out[source].append((target, score))

    return out


app = typer.Typer()

@app.command()
def main(
    jsonl_path: Path = typer.Argument(
        "./baselines/pubmed_baseline_2026.jsonl", help="The path to the JSONL file containing the articles."
    ),
    db_path: Path = typer.Argument(
        "./db_baselines/pubmed_baseline_2026.db", help="The path to the DuckDB database file."
    ),
):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_pubmed_db(jsonl_path, db_path)


@app.command(name="init-lookup")
def init_lookup_cmd(
    lookup_json: Path = typer.Argument(..., help="Path to the lookup JSON file ({pmid: [[pmid, score], ...]})."),
    db_path: Path = typer.Argument(
        "./db_baselines/pubmed_baseline_2026.db",
        help="Path to the DuckDB database file (same as articles).",
    ),
):
    """Cria a tabela 'lookup' na base de dados DuckDB a partir do ficheiro JSON."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    init_lookup_table(con, lookup_json)
    con.close()


if __name__ == "__main__":
    app()