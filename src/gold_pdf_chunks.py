# src/gold_pdf_chunks.py
import argparse
from typing import Iterator, List, Dict
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.utils import AnalysisException


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--silver_table", required=True)      # bind_agent.docs.pdf_ocr_silver
    p.add_argument("--gold_table", required=True)        # bind_agent.docs.pdf_chunks_gold
    p.add_argument("--chunk_size_chars", type=int, default=2500)
    p.add_argument("--chunk_overlap_chars", type=int, default=250)
    p.add_argument("--min_chunk_chars", type=int, default=200)
    p.add_argument("--topic_mode", default="heuristic", choices=["heuristic", "none"])
    return p.parse_args()


def ensure_gold_table(gold_table: str):
    """
    Crea Gold si no existe. Si existe, asegura columnas nuevas (ALTER TABLE).
    """
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

          -- heredados desde Silver (filename)
          file_date DATE,
          file_type STRING,

          page_id INT,
          page_num INT,
          page_label STRING,
          topic STRING,

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

    # Si ya existe, agregar columnas faltantes
    to_add = []
    if "file_date" not in existing_cols:
        to_add.append("file_date DATE")
    if "file_type" not in existing_cols:
        to_add.append("file_type STRING")
    if to_add:
        spark.sql(f"ALTER TABLE {gold_table} ADD COLUMNS ({', '.join(to_add)})")


def _first_meaningful_line(text: str) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        s = line.strip().lstrip("•-–—").strip()
        if len(s) >= 3:
            return s
    return ""


def build_topic_and_label(text: str, page_num: int, mode: str):
    page_label = f"page {page_num}" if page_num is not None else "page"
    if mode == "none":
        return "", page_label
    top = _first_meaningful_line(text)
    if len(top) > 80:
        top = top[:80].rstrip()
    return top, page_label


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
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


# Output schema para mapInPandas
OUT_SCHEMA = T.StructType([
    T.StructField("doc_id", T.StringType(), False),
    T.StructField("path", T.StringType(), True),
    T.StructField("modificationTime", T.TimestampType(), True),

    T.StructField("file_date", T.DateType(), True),
    T.StructField("file_type", T.StringType(), True),

    T.StructField("page_id", T.IntegerType(), True),
    T.StructField("page_num", T.IntegerType(), True),
    T.StructField("page_label", T.StringType(), True),
    T.StructField("topic", T.StringType(), True),

    T.StructField("page_hash", T.StringType(), True),

    T.StructField("chunk_index", T.IntegerType(), True),
    T.StructField("chunk_text", T.StringType(), True),
    T.StructField("chunk_len", T.IntegerType(), True),
])


def make_chunks_map_in_pandas(chunk_size: int, overlap: int, min_chars: int, topic_mode: str):
    def fn(it: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for pdf in it:
            rows: List[Dict] = []
            for r in pdf.itertuples(index=False):
                text = r.page_text if isinstance(r.page_text, str) else ""
                if not text or len(text.strip()) < min_chars:
                    continue

                topic, page_label = build_topic_and_label(text, r.page_num, topic_mode)
                chunks = chunk_text(text, chunk_size, overlap)

                for idx, ch in enumerate(chunks):
                    ch = ch.strip()
                    if len(ch) < min_chars:
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
                        "topic": topic,

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
    print(f"[gold_pdf_chunks] topic_mode          = {args.topic_mode}")

    ensure_gold_table(args.gold_table)

    silver = (
        spark.table(args.silver_table)
        .select(
            "doc_id", "path", "modificationTime",
            F.col("file_date").cast("date").alias("file_date"),
            F.col("file_type").cast("string").alias("file_type"),
            F.col("page_id").cast("int").alias("page_id"),
            F.col("page_num").cast("int").alias("page_num"),
            F.col("page_text").cast("string").alias("page_text"),
        )
        .withColumn("page_hash", F.sha2(F.coalesce(F.col("page_text"), F.lit("")), 256))
        .where("page_text IS NOT NULL AND length(trim(page_text)) > 0")
    )

    # Procesar sólo páginas nuevas o cuyo page_hash cambió
    gold_keys = (
        spark.table(args.gold_table)
        .select("doc_id", "modificationTime", "page_id", "page_hash")
        .dropDuplicates(["doc_id", "modificationTime", "page_id", "page_hash"])
    )

    pages_to_process = (
        silver.join(
            gold_keys,
            on=["doc_id", "modificationTime", "page_id", "page_hash"],
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
        topic_mode=args.topic_mode
    )

    chunks = (
        pages_to_process
        .select(
            "doc_id", "path", "modificationTime",
            "file_date", "file_type",
            "page_id", "page_num", "page_text", "page_hash"
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
                    F.coalesce(F.col("chunk_index").cast("string"), F.lit("")),
                    F.coalesce(F.col("page_hash"), F.lit(""))
                ),
                256
            )
        )
        .withColumn("gold_ingest_ts", F.current_timestamp())
    )

    chunks.createOrReplaceTempView("gold_chunks_updates")

    # 1) UPSERT por chunk_id (idempotente)
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
      t.topic = s.topic,
      t.page_hash = s.page_hash,
      t.chunk_index = s.chunk_index,
      t.chunk_text = s.chunk_text,
      t.chunk_len = s.chunk_len,
      t.gold_ingest_ts = s.gold_ingest_ts
    WHEN NOT MATCHED THEN INSERT (
      doc_id, path, modificationTime,
      file_date, file_type,
      page_id, page_num, page_label, topic,
      page_hash,
      chunk_id, chunk_index, chunk_text, chunk_len,
      gold_ingest_ts
    ) VALUES (
      s.doc_id, s.path, s.modificationTime,
      s.file_date, s.file_type,
      s.page_id, s.page_num, s.page_label, s.topic,
      s.page_hash,
      s.chunk_id, s.chunk_index, s.chunk_text, s.chunk_len,
      s.gold_ingest_ts
    )
    """)

    # 2) Borrar chunks viejos de páginas re-procesadas (sin multi-col IN)
    spark.sql(f"""
    DELETE FROM {args.gold_table} AS t
    WHERE EXISTS (
      SELECT 1
      FROM (
        SELECT DISTINCT doc_id, modificationTime, page_id
        FROM gold_chunks_updates
      ) k
      WHERE t.doc_id = k.doc_id
        AND t.modificationTime = k.modificationTime
        AND t.page_id = k.page_id
    )
    AND NOT EXISTS (
      SELECT 1
      FROM gold_chunks_updates s
      WHERE s.chunk_id = t.chunk_id
    )
    """)

    print("[gold_pdf_chunks] Upsert + cleanup completo.")


if __name__ == "__main__":
    main()
