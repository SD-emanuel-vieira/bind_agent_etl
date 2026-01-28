# src/embeddings_pdf_chunks.py
# ------------------------------------------------------------
# Genera / actualiza la tabla de embeddings de manera incremental
# a partir de la tabla Gold de chunks.
# ------------------------------------------------------------

import argparse
import time
from typing import Any, List, Optional

import mlflow.deployments
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incremental embeddings from Gold chunks")
    p.add_argument("--gold_table", required=True)
    p.add_argument("--embeddings_table", required=True)
    p.add_argument("--embedding_endpoint", required=True)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--max_retries", type=int, default=6)
    p.add_argument("--sleep_base_seconds", type=float, default=1.0)
    p.add_argument("--mode", choices=["driver", "distributed"], default="driver")
    p.add_argument("--max_rows_driver", type=int, default=20000)
    return p.parse_args()


def table_exists(full_name: str) -> bool:
    try:
        spark.table(full_name).limit(1).collect()
        return True
    except AnalysisException:
        return False


def ensure_embeddings_table(emb_table: str) -> None:
    if not table_exists(emb_table):
        spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {emb_table} (
          chunk_id          STRING,
          chunk_type        STRING,
          doc_id            STRING,
          path              STRING,
          modificationTime  TIMESTAMP,
          file_date         DATE,
          file_type         STRING,
          page_id           INT,
          page_num          INT,
          
          -- Topic fields (desde Silver/Gold)
          topic_heuristic   STRING,
          topic_llm         STRING,
          topic_content     STRING,
          
          -- Segmento de negocio
          page_segment      STRING,
          
          -- Metadata estructurada (JSON)
          metadata_enrich   STRING,
          
          chunk_text        STRING,
          chunk_char_len    INT,
          embed_text        STRING,
          chunk_hash        STRING,
          embedding         ARRAY<FLOAT>,
          embedding_model   STRING,
          embedding_dim     INT,
          embed_ts          TIMESTAMP
        )
        USING DELTA
        """)
        return


def ensure_cdf_enabled(emb_table: str) -> None:
    spark.sql(f"""
      ALTER TABLE {emb_table}
      SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    existing_cols = set([c.name for c in spark.table(emb_table).schema.fields])
    desired = [
        ("chunk_type", "STRING"),
        ("file_date", "DATE"),
        ("file_type", "STRING"),
        ("chunk_hash", "STRING"),
        ("embedding_model", "STRING"),
        ("embedding_dim", "INT"),
        ("embed_ts", "TIMESTAMP"),
        ("topic_heuristic", "STRING"),
        ("topic_llm", "STRING"),
        ("topic_content", "STRING"),
        ("page_segment", "STRING"),
        ("metadata_enrich", "STRING"),
    ]
    for col_name, col_type in desired:
        if col_name not in existing_cols:
            spark.sql(f"ALTER TABLE {emb_table} ADD COLUMNS ({col_name} {col_type})")


def normalize_embed_text(df: DataFrame) -> DataFrame:
    cleaned = F.regexp_replace(
        F.col("chunk_text"),
        r"(?s)^\[SOURCE:[^\]]*\]\s*\n\[TOPIC:[^\]]*\]\s*\n\s*",
        "",
    )
    return (
        df.withColumn("embed_text", F.trim(cleaned))
          .withColumn("chunk_char_len", F.length(F.col("embed_text")).cast("int"))
          .withColumn("chunk_hash", F.sha2(F.coalesce(F.col("embed_text"), F.lit("")), 256))
          .where(F.length(F.col("embed_text")) > 0)
    )


def _parse_embeddings_response(resp: Any, n_expected: int) -> List[List[float]]:
    if isinstance(resp, list):
        if len(resp) == n_expected and (len(resp) == 0 or isinstance(resp[0], list)):
            return resp

    if isinstance(resp, dict):
        if "data" in resp and isinstance(resp["data"], list):
            vecs = [d.get("embedding") for d in resp["data"]]
            vecs = [v for v in vecs if v is not None]
            if len(vecs) == n_expected:
                return vecs

        if "predictions" in resp:
            preds = resp["predictions"]
            if isinstance(preds, list) and len(preds) == n_expected:
                if len(preds) == 0:
                    return []
                if isinstance(preds[0], dict) and "embedding" in preds[0]:
                    return [p["embedding"] for p in preds]
                if isinstance(preds[0], list):
                    return preds

        if "embeddings" in resp and isinstance(resp["embeddings"], list):
            vecs = resp["embeddings"]
            if len(vecs) == n_expected:
                return vecs

        if "result" in resp and isinstance(resp["result"], list):
            vecs = resp["result"]
            if len(vecs) == n_expected:
                return vecs

    raise ValueError(f"No pude interpretar la respuesta del endpoint (esperaba {n_expected} embeddings).")


def embed_texts(client, endpoint: str, texts: List[str], max_retries: int, sleep_base_seconds: float) -> List[List[float]]:
    attempt = 0
    last_err: Optional[Exception] = None

    while attempt <= max_retries:
        try:
            resp = client.predict(endpoint=endpoint, inputs={"input": texts})
            return _parse_embeddings_response(resp, len(texts))
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                break
            sleep_s = sleep_base_seconds * (2 ** attempt)
            time.sleep(min(sleep_s, 30))
            attempt += 1

    raise RuntimeError(f"Falló llamada al endpoint después de {max_retries} reintentos. Último error: {last_err}")


def embed_driver(to_process: DataFrame, endpoint: str, batch_size: int, max_retries: int, sleep_base_seconds: float) -> DataFrame:
    pdf = to_process.toPandas()
    texts = pdf["embed_text"].astype(str).tolist()

    client = mlflow.deployments.get_deploy_client("databricks")

    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vecs = embed_texts(client, endpoint, batch, max_retries=max_retries, sleep_base_seconds=sleep_base_seconds)
        all_vecs.extend(vecs)

    pdf["embedding"] = all_vecs
    pdf["embedding_model"] = endpoint
    pdf["embedding_dim"] = pdf["embedding"].apply(lambda v: int(len(v)) if v is not None else None)

    updates = spark.createDataFrame(pdf)
    updates = updates.withColumn("embedding", F.expr("transform(embedding, x -> cast(x as float))"))
    updates = updates.withColumn("embed_ts", F.current_timestamp())
    return updates


def embed_distributed(to_process: DataFrame, endpoint: str, batch_size: int, max_retries: int, sleep_base_seconds: float) -> DataFrame:
    def iterator_embed(pdf_iter):
        client = mlflow.deployments.get_deploy_client("databricks")
        for pdf in pdf_iter:
            texts = pdf["embed_text"].astype(str).tolist()

            vecs_all: List[List[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                vecs = embed_texts(client, endpoint, batch, max_retries=max_retries, sleep_base_seconds=sleep_base_seconds)
                vecs_all.extend(vecs)

            pdf["embedding"] = vecs_all
            pdf["embedding_model"] = endpoint
            pdf["embedding_dim"] = [len(v) if v is not None else None for v in vecs_all]
            yield pdf

    base = to_process
    updates = base.mapInPandas(iterator_embed, schema=(
        base.schema
            .add("embedding", "array<float>")
            .add("embedding_model", "string")
            .add("embedding_dim", "int")
    ))
    updates = updates.withColumn("embed_ts", F.current_timestamp())
    return updates


def upsert_embeddings(emb_table: str, updates: DataFrame) -> None:
    updates.createOrReplaceTempView("emb_updates")
    spark.sql(f"""
    MERGE INTO {emb_table} t
    USING emb_updates s
    ON t.chunk_id = s.chunk_id
    WHEN MATCHED THEN UPDATE SET
      t.chunk_type       = s.chunk_type,
      t.doc_id           = s.doc_id,
      t.path             = s.path,
      t.modificationTime = s.modificationTime,
      t.file_date        = s.file_date,
      t.file_type        = s.file_type,
      t.page_id          = s.page_id,
      t.page_num         = s.page_num,
      t.topic_heuristic  = s.topic_heuristic,
      t.topic_llm        = s.topic_llm,
      t.topic_content    = s.topic_content,
      t.page_segment     = s.page_segment,
      t.metadata_enrich  = s.metadata_enrich,
      t.chunk_text       = s.chunk_text,
      t.chunk_char_len   = s.chunk_char_len,
      t.embed_text       = s.embed_text,
      t.chunk_hash       = s.chunk_hash,
      t.embedding        = s.embedding,
      t.embedding_model  = s.embedding_model,
      t.embedding_dim    = s.embedding_dim,
      t.embed_ts         = s.embed_ts
    WHEN NOT MATCHED THEN INSERT (
      chunk_id, chunk_type, doc_id, path, modificationTime,
      file_date, file_type,
      page_id, page_num, 
      topic_heuristic, topic_llm, topic_content, page_segment, metadata_enrich,
      chunk_text, chunk_char_len, embed_text, chunk_hash,
      embedding, embedding_model, embedding_dim, embed_ts
    )
    VALUES (
      s.chunk_id, s.chunk_type, s.doc_id, s.path, s.modificationTime,
      s.file_date, s.file_type,
      s.page_id, s.page_num,
      s.topic_heuristic, s.topic_llm, s.topic_content, s.page_segment, s.metadata_enrich,
      s.chunk_text, s.chunk_char_len, s.embed_text, s.chunk_hash,
      s.embedding, s.embedding_model, s.embedding_dim, s.embed_ts
    )
    """)


def main() -> None:
    args = parse_args()

    print("[embeddings] gold_table        =", args.gold_table)
    print("[embeddings] embeddings_table  =", args.embeddings_table)
    print("[embeddings] embedding_endpoint=", args.embedding_endpoint)
    print("[embeddings] mode              =", args.mode)

    ensure_embeddings_table(args.embeddings_table)
    ensure_cdf_enabled(args.embeddings_table)

    gold = spark.table(args.gold_table)

    # Soportar versiones antiguas de Gold (sin chunk_type)
    if "chunk_type" not in gold.columns:
        gold = gold.withColumn("chunk_type", F.lit("text"))
    else:
        gold = gold.withColumn("chunk_type", F.coalesce(F.col("chunk_type"), F.lit("text")))

    # Campos opcionales con defaults
    if "file_date" not in gold.columns:
        gold = gold.withColumn("file_date", F.lit(None).cast("date"))
    if "file_type" not in gold.columns:
        gold = gold.withColumn("file_type", F.lit(None).cast("string"))
    if "topic_heuristic" not in gold.columns:
        gold = gold.withColumn("topic_heuristic", F.lit(None).cast("string"))
    if "topic_llm" not in gold.columns:
        gold = gold.withColumn("topic_llm", F.lit(None).cast("string"))
    if "topic_content" not in gold.columns:
        gold = gold.withColumn("topic_content", F.lit(None).cast("string"))
    if "page_segment" not in gold.columns:
        gold = gold.withColumn("page_segment", F.lit(None).cast("string"))
    if "metadata_enrich" not in gold.columns:
        gold = gold.withColumn("metadata_enrich", F.lit(None).cast("string"))

    base = gold.select(
        "chunk_id", "chunk_type", "doc_id", "path", "modificationTime",
        "file_date", "file_type",
        "page_id", "page_num", 
        "topic_heuristic", "topic_llm", "topic_content", "page_segment", "metadata_enrich",
        "chunk_text"
    )

    base = normalize_embed_text(base)

    existing = spark.table(args.embeddings_table).select("chunk_id", "chunk_hash", "embedding_model").dropDuplicates(["chunk_id"])
    to_process = (
        base.alias("b")
        .join(existing.alias("e"), on="chunk_id", how="left")
        .where(
            (F.col("e.chunk_id").isNull()) |
            (F.col("e.chunk_hash") != F.col("b.chunk_hash")) |
            (F.col("e.embedding_model") != F.lit(args.embedding_endpoint))
        )
        .select("b.*")
    )

    n = to_process.count()
    print("[embeddings] chunks a procesar =", n)

    if n == 0:
        print("[embeddings] Nada para hacer.")
        return

    mode = args.mode
    if mode == "driver" and n > args.max_rows_driver:
        print(f"[embeddings] n={n} supera max_rows_driver={args.max_rows_driver}. Fuerzo mode=distributed.")
        mode = "distributed"

    if mode == "driver":
        updates = embed_driver(to_process, args.embedding_endpoint, args.batch_size, args.max_retries, args.sleep_base_seconds)
    else:
        updates = embed_distributed(to_process, args.embedding_endpoint, args.batch_size, args.max_retries, args.sleep_base_seconds)

    upsert_embeddings(args.embeddings_table, updates)
    print("[embeddings] Upsert completo.")


if __name__ == "__main__":
    main()
