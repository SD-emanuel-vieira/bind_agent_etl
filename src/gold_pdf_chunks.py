# src/gold_pdf_chunks.py
"""
Gold PDF Chunks - Chunking con manejo especial de tablas

Tipos de chunks:
- 'text': Texto narrativo, con chunking normal (fragmentación + overlap)
- 'table': Tablas completas, SIN fragmentar (cada tabla = 1 chunk)
- 'figure_enriched': Gráficos interpretados por LLM
"""
import argparse
from typing import Iterator, List, Dict
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.utils import AnalysisException


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--silver_table", required=True)
    p.add_argument("--gold_table", required=True)
    p.add_argument("--chunk_size_chars", type=int, default=2500)
    p.add_argument("--chunk_overlap_chars", type=int, default=250)
    p.add_argument("--min_chunk_chars", type=int, default=200)
    return p.parse_args()


def ensure_gold_table(gold_table: str):
    """Crea Gold si no existe."""
    exists = True
    try:
        df = spark.table(gold_table)
        existing_cols = {f.name.lower() for f in df.schema.fields}
    except AnalysisException:
        exists = False
        existing_cols = set()

    if not exists:
        spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {gold_table} (
          doc_id STRING,
          path STRING,
          modificationTime TIMESTAMP,

          file_date DATE,
          file_type STRING,

          page_id INT,
          page_num INT,
          page_label STRING,
          page_text STRING,

          -- Topic fields
          topic_heuristic STRING,
          topic_llm STRING,
          topic_content STRING,
          
          -- Segmento de negocio
          page_segment STRING,

          -- Tipo: 'text', 'table', 'figure_enriched'
          chunk_type STRING,

          page_figures_enriched_text STRING,

          page_hash STRING,

          chunk_id STRING,
          chunk_index INT,
          chunk_text STRING,
          chunk_len INT,

          gold_ingest_ts TIMESTAMP
        )
        USING DELTA
        """)
        return

    # Agregar columnas faltantes si existen
    to_add = []
    if "page_text" not in existing_cols:
        to_add.append("page_text STRING")
    if "topic_heuristic" not in existing_cols:
        to_add.append("topic_heuristic STRING")
    if "topic_llm" not in existing_cols:
        to_add.append("topic_llm STRING")
    if "topic_content" not in existing_cols:
        to_add.append("topic_content STRING")
    if "page_segment" not in existing_cols:
        to_add.append("page_segment STRING")
    if to_add:
        spark.sql(f"ALTER TABLE {gold_table} ADD COLUMNS ({', '.join(to_add)})")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Fragmenta texto con overlap. Solo para texto narrativo."""
    if not text:
        return []

    txt = "\n".join([ln.rstrip() for ln in text.splitlines()]).strip()
    if len(txt) <= chunk_size:
        return [txt]

    paras = [p.strip() for p in txt.split("\n\n") if p.strip()]
    chunks = []
    cur = ""

    def flush_cur():
        nonlocal cur
        if cur.strip():
            chunks.append(cur.strip())
        cur = ""

    for p in paras:
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= chunk_size:
            cur = cur + "\n\n" + p
        else:
            flush_cur()
            cur = p

    flush_cur()

    fixed = []
    for c in chunks:
        if len(c) <= chunk_size:
            fixed.append(c)
        else:
            i = 0
            while i < len(c):
                fixed.append(c[i:i + chunk_size])
                i += max(1, chunk_size - overlap)

    final = []
    for i, c in enumerate(fixed):
        if i == 0:
            final.append(c)
        else:
            prev = final[-1]
            tail = prev[-overlap:] if overlap > 0 and len(prev) > overlap else ""
            merged = (tail + "\n" + c).strip() if tail else c
            if len(merged) > chunk_size + overlap:
                merged = merged[-(chunk_size + overlap):]
            final.append(merged)

    return final


# Output schema
OUT_SCHEMA = T.StructType([
    T.StructField("doc_id", T.StringType(), False),
    T.StructField("path", T.StringType(), True),
    T.StructField("modificationTime", T.TimestampType(), True),

    T.StructField("file_date", T.DateType(), True),
    T.StructField("file_type", T.StringType(), True),

    T.StructField("page_id", T.IntegerType(), True),
    T.StructField("page_num", T.IntegerType(), True),
    T.StructField("page_label", T.StringType(), True),

    T.StructField("page_text", T.StringType(), True),

    T.StructField("topic_heuristic", T.StringType(), True),
    T.StructField("topic_llm", T.StringType(), True),
    T.StructField("topic_content", T.StringType(), True),
    T.StructField("page_segment", T.StringType(), True),

    T.StructField("chunk_type", T.StringType(), True),

    T.StructField("page_figures_enriched_text", T.StringType(), True),

    T.StructField("page_hash", T.StringType(), True),

    T.StructField("chunk_index", T.IntegerType(), True),
    T.StructField("chunk_text", T.StringType(), True),
    T.StructField("chunk_len", T.IntegerType(), True),
])


def make_chunks_map_in_pandas(chunk_size: int, overlap: int, min_chars: int):
    """
    Genera chunks según el tipo de contenido:
    - 'text': Chunking normal con fragmentación
    - 'table': SIN fragmentar, cada tabla es un chunk completo
    - 'figure_enriched': Chunking normal
    """
    def fn(it: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for pdf in it:
            rows: List[Dict] = []
            for r in pdf.itertuples(index=False):
                chunk_type = getattr(r, "chunk_type", None)
                text = r.content_text if isinstance(getattr(r, "content_text", None), str) else ""
                
                page_label = f"page {r.page_num}" if r.page_num is not None else "page"

                # =============================================
                # TABLAS: No fragmentar, cada tabla = 1 chunk
                # =============================================
                if chunk_type == "table":
                    # El content_text puede tener múltiples tablas separadas por ---TABLE_SEPARATOR---
                    tables = text.split("---TABLE_SEPARATOR---")
                    
                    for idx, table_content in enumerate(tables):
                        table_content = table_content.strip()
                        if not table_content or len(table_content) < 20:  # Mínimo para una tabla
                            continue
                        
                        rows.append({
                            "doc_id": r.doc_id,
                            "path": r.path,
                            "modificationTime": r.modificationTime,
                            "file_date": r.file_date,
                            "file_type": r.file_type,
                            "page_id": int(r.page_id) if r.page_id is not None else None,
                            "page_num": int(r.page_num) if r.page_num is not None else None,
                            "page_label": page_label,
                            "page_text": r.page_text
                                if isinstance(getattr(r, "page_text", None), str) else None,
                            "topic_heuristic": r.topic_heuristic
                                if isinstance(getattr(r, "topic_heuristic", None), str) else None,
                            "topic_llm": r.topic_llm
                                if isinstance(getattr(r, "topic_llm", None), str) else None,
                            "topic_content": r.topic_content
                                if isinstance(getattr(r, "topic_content", None), str) else None,
                            "page_segment": r.page_segment
                                if isinstance(getattr(r, "page_segment", None), str) else None,
                            "chunk_type": "table",
                            "page_figures_enriched_text": None,
                            "page_hash": r.page_hash,
                            "chunk_index": idx,
                            "chunk_text": table_content,  # Tabla completa, sin fragmentar
                            "chunk_len": len(table_content),
                        })
                    continue

                # =============================================
                # TEXTO y FIGURE_ENRICHED: Chunking normal
                # =============================================
                min_chars_local = min_chars if chunk_type == "text" else min(50, min_chars)

                if not text or len(text.strip()) < min_chars_local:
                    continue

                chunks = chunk_text(text, chunk_size, overlap)

                for idx, ch in enumerate(chunks):
                    ch = ch.strip()
                    if len(ch) < min_chars_local:
                        continue
                    rows.append({
                        "doc_id": r.doc_id,
                        "path": r.path,
                        "modificationTime": r.modificationTime,
                        "file_date": r.file_date,
                        "file_type": r.file_type,
                        "page_id": int(r.page_id) if r.page_id is not None else None,
                        "page_num": int(r.page_num) if r.page_num is not None else None,
                        "page_label": page_label,
                        "page_text": r.page_text
                            if isinstance(getattr(r, "page_text", None), str) else None,
                        "topic_heuristic": r.topic_heuristic
                            if isinstance(getattr(r, "topic_heuristic", None), str) else None,
                        "topic_llm": r.topic_llm
                            if isinstance(getattr(r, "topic_llm", None), str) else None,
                        "topic_content": r.topic_content
                            if isinstance(getattr(r, "topic_content", None), str) else None,
                        "page_segment": r.page_segment
                            if isinstance(getattr(r, "page_segment", None), str) else None,
                        "chunk_type": chunk_type,
                        "page_figures_enriched_text": r.page_figures_enriched_text
                            if isinstance(getattr(r, "page_figures_enriched_text", None), str) else None,
                        "page_hash": r.page_hash,
                        "chunk_index": idx,
                        "chunk_text": ch,
                        "chunk_len": len(ch),
                    })
            yield pd.DataFrame(rows, columns=[f.name for f in OUT_SCHEMA.fields])
    return fn


def main():
    args = parse_args()

    print(f"[gold_pdf_chunks] silver_table        = {args.silver_table}")
    print(f"[gold_pdf_chunks] gold_table          = {args.gold_table}")
    print(f"[gold_pdf_chunks] chunk_size_chars    = {args.chunk_size_chars}")
    print(f"[gold_pdf_chunks] chunk_overlap_chars = {args.chunk_overlap_chars}")
    print(f"[gold_pdf_chunks] min_chunk_chars     = {args.min_chunk_chars}")

    ensure_gold_table(args.gold_table)

    # ------------------------------------------------------------
    # Construimos Gold con 3 tipos de chunks:
    # - chunk_type='text'            -> page_text (narrativo)
    # - chunk_type='table'           -> page_table_text (tablas completas)
    # - chunk_type='figure_enriched' -> page_figures_enriched_text
    # ------------------------------------------------------------
    silver_df = spark.table(args.silver_table)
    available_cols = set(silver_df.columns)
    
    # Construir select dinámico para manejar columnas opcionales
    select_cols = [
        "doc_id", "path", "modificationTime",
        F.col("file_date").cast("date").alias("file_date"),
        F.col("file_type").cast("string").alias("file_type"),
        F.col("page_id").cast("int").alias("page_id"),
        F.col("page_num").cast("int").alias("page_num"),
    ]

    # Columnas opcionales
    if "page_text" in available_cols:
        select_cols.append(F.col("page_text").cast("string").alias("page_text"))
    else:
        select_cols.append(F.lit(None).cast("string").alias("page_text"))

    if "page_table_text" in available_cols:
        select_cols.append(F.col("page_table_text").cast("string").alias("page_table_text"))
    else:
        select_cols.append(F.lit(None).cast("string").alias("page_table_text"))
    
    if "page_figures_enriched_text" in available_cols:
        select_cols.append(F.col("page_figures_enriched_text").cast("string").alias("page_figures_enriched_text"))
    else:
        select_cols.append(F.lit(None).cast("string").alias("page_figures_enriched_text"))

    if "topic_heuristic" in available_cols:
        select_cols.append(F.col("topic_heuristic").cast("string").alias("topic_heuristic"))
    else:
        select_cols.append(F.lit(None).cast("string").alias("topic_heuristic"))
    
    if "topic_llm" in available_cols:
        select_cols.append(F.col("topic_llm").cast("string").alias("topic_llm"))
    else:
        select_cols.append(F.lit(None).cast("string").alias("topic_llm"))
    
    if "topic_content" in available_cols:
        select_cols.append(F.col("topic_content").cast("string").alias("topic_content"))
    else:
        select_cols.append(F.lit(None).cast("string").alias("topic_content"))
    
    if "page_segment" in available_cols:
        select_cols.append(F.col("page_segment").cast("string").alias("page_segment"))
    else:
        select_cols.append(F.lit(None).cast("string").alias("page_segment"))
    
    silver_base = silver_df.select(*select_cols)

    # Texto narrativo (sin tablas)
    pages_text = (
        silver_base
        .where("page_text IS NOT NULL AND length(trim(page_text)) > 0")
        .withColumn("chunk_type", F.lit("text"))
        .withColumn("content_text", F.col("page_text"))
        .drop("page_table_text")
    )

    # Tablas (sin fragmentar)
    pages_tables = (
        silver_base
        .where("page_table_text IS NOT NULL AND length(trim(page_table_text)) > 0")
        .withColumn("chunk_type", F.lit("table"))
        .withColumn("content_text", F.col("page_table_text"))
        .drop("page_table_text")
    )

    # Figuras enriquecidas
    pages_figures = (
        silver_base
        .where("page_figures_enriched_text IS NOT NULL AND length(trim(page_figures_enriched_text)) > 0")
        .withColumn("chunk_type", F.lit("figure_enriched"))
        .withColumn("content_text", F.col("page_figures_enriched_text"))
        .drop("page_table_text")
    )

    silver = (
        pages_text
        .unionByName(pages_tables, allowMissingColumns=True)
        .unionByName(pages_figures, allowMissingColumns=True)
        .withColumn(
            "page_hash",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.lit("v7"),  # Incrementar versión por agregar page_segment
                    F.col("chunk_type"),
                    F.coalesce(F.col("content_text"), F.lit("")),
                    F.coalesce(F.col("page_text"), F.lit("")),
                ),
                256,
            ),
        )
    )

    # Procesar solo páginas nuevas o cuyo page_hash cambió
    gold_keys = (
        spark.table(args.gold_table)
        .select("doc_id", "modificationTime", "page_id", "chunk_type", "page_hash")
        .dropDuplicates(["doc_id", "modificationTime", "page_id", "chunk_type", "page_hash"])
    )

    pages_to_process = (
        silver.join(
            gold_keys,
            on=["doc_id", "modificationTime", "page_id", "chunk_type", "page_hash"],
            how="left_anti"
        )
    )

    if pages_to_process.limit(1).count() == 0:
        print("[gold_pdf_chunks] No hay páginas nuevas/cambiadas para procesar.")
        return

    chunker = make_chunks_map_in_pandas(
        chunk_size=args.chunk_size_chars,
        overlap=args.chunk_overlap_chars,
        min_chars=args.min_chunk_chars,
    )

    chunks = (
        pages_to_process
        .select(
            "doc_id", "path", "modificationTime",
            "file_date", "file_type",
            "page_id", "page_num", "chunk_type", "content_text",
            "page_text",
            "page_figures_enriched_text", "page_hash",
            "topic_heuristic", "topic_llm", "topic_content", "page_segment",
        )
        .mapInPandas(chunker, schema=OUT_SCHEMA)
        .withColumn(
            "chunk_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.col("doc_id"),
                    F.coalesce(F.col("modificationTime").cast("string"), F.lit("")),
                    F.coalesce(F.col("page_id").cast("string"), F.lit("")),
                    F.coalesce(F.col("chunk_type"), F.lit("")),
                    F.coalesce(F.col("chunk_index").cast("string"), F.lit("")),
                    F.coalesce(F.col("page_hash"), F.lit(""))
                ),
                256
            )
        )
        .withColumn("gold_ingest_ts", F.current_timestamp())
    )

    chunks.createOrReplaceTempView("gold_chunks_updates")

    # 1) UPSERT
    spark.sql(f"""
    MERGE INTO {args.gold_table} AS t
    USING gold_chunks_updates AS s
    ON t.chunk_id = s.chunk_id
    WHEN MATCHED THEN UPDATE SET
      t.doc_id = s.doc_id,
      t.path = s.path,
      t.modificationTime = s.modificationTime,
      t.file_date = s.file_date,
      t.file_type = s.file_type,
      t.page_id = s.page_id,
      t.page_num = s.page_num,
      t.page_label = s.page_label,
      t.page_text = s.page_text,
      t.topic_heuristic = s.topic_heuristic,
      t.topic_llm = s.topic_llm,
      t.topic_content = s.topic_content,
      t.page_segment = s.page_segment,
      t.chunk_type = s.chunk_type,
      t.page_figures_enriched_text = s.page_figures_enriched_text,
      t.page_hash = s.page_hash,
      t.chunk_index = s.chunk_index,
      t.chunk_text = s.chunk_text,
      t.chunk_len = s.chunk_len,
      t.gold_ingest_ts = s.gold_ingest_ts
    WHEN NOT MATCHED THEN INSERT (
      doc_id, path, modificationTime,
      file_date, file_type,
      page_id, page_num, page_label, page_text,
      topic_heuristic, topic_llm, topic_content, page_segment,
      chunk_type,
      page_figures_enriched_text,
      page_hash,
      chunk_id, chunk_index, chunk_text, chunk_len,
      gold_ingest_ts
    ) VALUES (
      s.doc_id, s.path, s.modificationTime,
      s.file_date, s.file_type,
      s.page_id, s.page_num, s.page_label, s.page_text,
      s.topic_heuristic, s.topic_llm, s.topic_content, s.page_segment,
      s.chunk_type,
      s.page_figures_enriched_text,
      s.page_hash,
      s.chunk_id, s.chunk_index, s.chunk_text, s.chunk_len,
      s.gold_ingest_ts
    )
    """)

    # 2) Borrar chunks viejos
    spark.sql(f"""
    DELETE FROM {args.gold_table} AS t
    WHERE EXISTS (
      SELECT 1
      FROM (
        SELECT DISTINCT doc_id, modificationTime, page_id, chunk_type
        FROM gold_chunks_updates
      ) k
      WHERE t.doc_id = k.doc_id
        AND t.modificationTime = k.modificationTime
        AND t.page_id = k.page_id
        AND t.chunk_type = k.chunk_type
    )
    AND NOT EXISTS (
      SELECT 1
      FROM gold_chunks_updates s
      WHERE s.chunk_id = t.chunk_id
    )
    """)

    # 3) Limpieza de legado
    spark.sql(f"""
    DELETE FROM {args.gold_table} AS t
    WHERE t.chunk_type IS NULL
      AND EXISTS (
        SELECT 1
        FROM (
          SELECT DISTINCT doc_id, modificationTime, page_id
          FROM gold_chunks_updates
        ) k
        WHERE t.doc_id = k.doc_id
          AND t.modificationTime = k.modificationTime
          AND t.page_id = k.page_id
      )
    """)

    print("[gold_pdf_chunks] Upsert + cleanup completo.")


if __name__ == "__main__":
    main()
