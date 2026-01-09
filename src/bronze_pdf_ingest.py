# src/bronze_pdf_ingest.py
import argparse
from pyspark.sql import functions as F

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)        # e.g. dbfs:/Volumes/<cat>/<schema>/pdf/
    p.add_argument("--checkpoint_path", required=True)   # e.g. dbfs:/Volumes/<cat>/<schema>/docs_state/.../checkpoints/
    p.add_argument("--schema_location", required=True)   # e.g. dbfs:/Volumes/<cat>/<schema>/docs_state/.../schema/
    p.add_argument("--bronze_table", required=True)      # e.g. catalog.schema.pdf_bronze
    return p.parse_args()

def ensure_bronze_table(bronze_table: str):
    """
    En enfoque productivo, UC schema/volumes se crean vía bundle (deploy).
    Acá solo aseguramos la tabla Bronze (idempotente).
    """
    parts = bronze_table.split(".")
    if len(parts) != 3:
        raise ValueError(f"bronze_table debe ser catalog.schema.table. Recibido: {bronze_table}")

    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {bronze_table} (
      path             STRING,
      modificationTime TIMESTAMP,
      length           BIGINT,
      content          BINARY,
      ingest_ts        TIMESTAMP,
      doc_id           STRING
    )
    USING DELTA
    """)

def main():
    args = parse_args()

    # Logs útiles (para debug de prod)
    print(f"[bronze_pdf_ingest] input_path      = {args.input_path}")
    print(f"[bronze_pdf_ingest] schema_location = {args.schema_location}")
    print(f"[bronze_pdf_ingest] checkpoint_path = {args.checkpoint_path}")
    print(f"[bronze_pdf_ingest] bronze_table    = {args.bronze_table}")

    ensure_bronze_table(args.bronze_table)

    df = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("cloudFiles.schemaLocation", args.schema_location)
        .option("cloudFiles.useManagedFileEvents", "false")
        .option("pathGlobFilter", "*.pdf")
        .load(args.input_path)
        .withColumn("ingest_ts", F.current_timestamp())
        .withColumn("doc_id", F.sha2(F.col("path"), 256))
        .select(
            "path",
            F.col("modificationTime").cast("timestamp").alias("modificationTime"),
            F.col("length").cast("bigint").alias("length"),
            "content",
            "ingest_ts",
            "doc_id",
        )
    )

    (
        df.writeStream
        .option("checkpointLocation", args.checkpoint_path)
        .trigger(availableNow=True)   # incremental: procesa lo nuevo y termina
        .toTable(args.bronze_table)
    )

if __name__ == "__main__":
    main()
