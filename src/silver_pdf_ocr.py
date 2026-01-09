# src/silver_pdf_ocr.py
import argparse
from pyspark.sql import functions as F
from pyspark.sql.functions import expr
from pyspark.sql.utils import AnalysisException


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bronze_table", required=True)      # bind_agent.docs.pdf_bronze
    p.add_argument("--silver_table", required=True)      # bind_agent.docs.pdf_ocr_silver
    p.add_argument("--image_output_path", required=False, default="")  # /Volumes/... (opcional)
    return p.parse_args()


def ensure_silver_table(silver_table: str):
    """
    Crea la tabla Silver (1 fila por página) si no existe.
    Si existe, asegura columnas nuevas (ALTER TABLE ADD COLUMNS).
    """
    exists = True
    try:
        df = spark.table(silver_table)
        existing_cols = {f.name.lower() for f in df.schema.fields}
    except AnalysisException:
        exists = False
        existing_cols = set()

    if not exists:
        spark.sql(f"""
        CREATE TABLE {silver_table} (
          doc_id STRING,
          path STRING,
          modificationTime TIMESTAMP,
          bronze_ingest_ts TIMESTAMP,

          -- NUEVO (doc-level, derivado del filename)
          file_date DATE,
          file_type STRING,

          page_id INT,           -- 0-based
          page_num INT,          -- 1-based
          page_image_uri STRING, -- si imageOutputPath está habilitado

          page_text STRING,
          page_elements ARRAY<VARIANT>,

          error_status VARIANT,     -- doc-level
          parsed_metadata VARIANT,  -- doc-level
          ocr_ingest_ts TIMESTAMP
        )
        USING DELTA
        """)
        return

    # Si ya existe, asegurar columnas nuevas
    to_add = []
    if "file_date" not in existing_cols:
        to_add.append("file_date DATE")
    if "file_type" not in existing_cols:
        to_add.append("file_type STRING")

    if to_add:
        spark.sql(f"ALTER TABLE {silver_table} ADD COLUMNS ({', '.join(to_add)})")


def build_parse_expr(image_output_path: str) -> str:
    kvs = [
        "'version','2.0'",
        "'descriptionElementTypes',''"  # deshabilita descripciones de figuras
    ]
    if image_output_path and image_output_path.strip():
        kvs.append(f"'imageOutputPath','{image_output_path}'")

    return f"ai_parse_document(content, map({','.join(kvs)}))"


def add_file_metadata(df):
    """
    Agrega:
      - file_type: primer token alfabético del filename (p.ej. 'CdG', 'Directorio')
      - file_date: fecha extraída del filename en DATE (YYYY-MM-DD)
        Soporta:
          * YYYY_MM_DD  (CdG_Intro_2025_11_03 ...)
          * YYYY-MM-DD
          * DD-MM-YYYY  (Directorio ... 18-11-2025 ...)
    """
    # basename del path
    filename = F.regexp_extract(F.col("path"), r"([^/]+)$", 1)

    # file_type: primer bloque de letras (incluye acentos/ñ)
    file_type = F.regexp_extract(filename, r"^([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)", 1)
    file_type = F.when(F.length(file_type) > 0, file_type).otherwise(F.lit(None).cast("string"))

    # Formato 1: YYYY[_-]MM[_-]DD
    y1 = F.regexp_extract(filename, r"(\d{4})[_-](\d{2})[_-](\d{2})", 1)
    m1 = F.regexp_extract(filename, r"(\d{4})[_-](\d{2})[_-](\d{2})", 2)
    d1 = F.regexp_extract(filename, r"(\d{4})[_-](\d{2})[_-](\d{2})", 3)

    # Formato 2: DD-MM-YYYY
    d2 = F.regexp_extract(filename, r"(\d{2})-(\d{2})-(\d{4})", 1)
    m2 = F.regexp_extract(filename, r"(\d{2})-(\d{2})-(\d{4})", 2)
    y2 = F.regexp_extract(filename, r"(\d{2})-(\d{2})-(\d{4})", 3)

    date_str = (
        F.when(y1 != "", F.concat_ws("-", y1, m1, d1))
         .when(y2 != "", F.concat_ws("-", y2, m2, d2))
         .otherwise(F.lit(None).cast("string"))
    )

    file_date = F.to_date(date_str, "yyyy-MM-dd")

    return (
        df.withColumn("file_type", file_type)
          .withColumn("file_date", file_date)
    )


def main():
    args = parse_args()

    print(f"[silver_pdf_ocr] bronze_table      = {args.bronze_table}")
    print(f"[silver_pdf_ocr] silver_table      = {args.silver_table}")
    print(f"[silver_pdf_ocr] image_output_path = {args.image_output_path}")

    ensure_silver_table(args.silver_table)

    # -----------------------------
    # 1) Leer Bronze
    # -----------------------------
    bronze_df = (
        spark.table(args.bronze_table)
        .select("doc_id", "path", "modificationTime", "ingest_ts", "content")
    )

    # -----------------------------
    # 2) Filtrar docs nuevos (doc_id + modificationTime)
    # -----------------------------
    already = (
        spark.table(args.silver_table)
        .select("doc_id", "modificationTime")
        .distinct()
    )

    to_process = bronze_df.join(already, on=["doc_id", "modificationTime"], how="left_anti")

    if to_process.limit(1).count() == 0:
        print("[silver_pdf_ocr] No hay documentos nuevos para procesar.")
        return

    # -----------------------------
    # 3) OCR parse con ai_parse_document()
    # -----------------------------
    parse_expr = build_parse_expr(args.image_output_path)

    parsed_docs = (
        to_process
        .withColumn("parsed", expr(parse_expr))
        .withColumn("error_status", expr("parsed:error_status"))
        .withColumn("parsed_metadata", expr("parsed:metadata"))
        .withColumn("bronze_ingest_ts", F.col("ingest_ts"))
        .withColumn("ocr_ingest_ts", F.current_timestamp())
        .select(
            "doc_id", "path", "modificationTime", "bronze_ingest_ts",
            "parsed", "error_status", "parsed_metadata", "ocr_ingest_ts"
        )
    )

    # Solo docs sin error top-level (sin agrupar por VARIANT)
    parsed_docs_ok = parsed_docs.where("try_cast(error_status AS STRING) IS NULL")

    # Agregar file_date / file_type (doc-level)
    parsed_docs_ok = add_file_metadata(parsed_docs_ok)

    # -----------------------------
    # 4) Expandir a 1 fila por página (sin groupBy de VARIANT)
    # -----------------------------
    docs_base = (
        parsed_docs_ok
        .select(
            "doc_id", "path", "modificationTime", "bronze_ingest_ts",
            "file_date", "file_type",
            "error_status", "parsed_metadata", "ocr_ingest_ts",
            expr("try_cast(parsed:document:elements AS ARRAY<VARIANT>) AS elements"),
            expr("try_cast(parsed:document:pages AS ARRAY<VARIANT>) AS pages")
        )
    )

    # pages: page_id + image_uri (si imageOutputPath habilitado)
    pages = (
        docs_base
        .select("doc_id", "modificationTime", expr("explode_outer(pages) AS p"))
        .select(
            "doc_id", "modificationTime",
            expr("try_cast(p:id AS INT) AS page_id"),
            expr("try_cast(p:image_uri AS STRING) AS page_image_uri")
        )
    )

    # elements: explode y asignar page_id desde bbox[0].page_id
    elements = (
        docs_base
        .select("doc_id", "path", "modificationTime", "bronze_ingest_ts", "elements")
        .withColumn("e", expr("explode_outer(elements)"))
        .select(
            "doc_id", "path", "modificationTime", "bronze_ingest_ts",
            expr("try_cast(e:id AS INT) AS element_id"),
            expr("e AS element"),
            expr("try_cast(e:bbox[0]:page_id AS INT) AS page_id"),
            expr("try_cast(e:content AS STRING) AS content")
        )
        .where("page_id IS NOT NULL AND content IS NOT NULL AND length(trim(content)) > 0")
    )

    # Aggregate por página (NO incluir VARIANT en groupBy)
    per_page = (
        elements
        .groupBy("doc_id", "path", "modificationTime", "bronze_ingest_ts", "page_id")
        .agg(
            expr("""
              transform(
                array_sort(
                  collect_list(named_struct('id', element_id, 'content', content))
                ),
                x -> x.content
              ) AS contents_sorted
            """),
            expr("collect_list(element) AS page_elements")
        )
        .withColumn("page_text", expr("concat_ws('\n\n', contents_sorted)"))
        .withColumn("page_num", expr("page_id + 1"))
        .drop("contents_sorted")
    )

    # Re-adjuntar campos doc-level (VARIANT) + file_date/file_type sin agrupar
    doc_meta = (
        docs_base
        .select(
            "doc_id", "modificationTime",
            "file_date", "file_type",
            "error_status", "parsed_metadata", "ocr_ingest_ts"
        )
        .dropDuplicates(["doc_id", "modificationTime"])
    )

    updates = (
        per_page
        .join(pages, on=["doc_id", "modificationTime", "page_id"], how="left")
        .join(doc_meta, on=["doc_id", "modificationTime"], how="left")
        .select(
            "doc_id", "path", "modificationTime", "bronze_ingest_ts",
            "file_date", "file_type",
            "page_id", "page_num", "page_image_uri",
            "page_text", "page_elements",
            "error_status", "parsed_metadata",
            "ocr_ingest_ts"
        )
    )

    # -----------------------------
    # 5) MERGE idempotente a Silver
    # -----------------------------
    updates.createOrReplaceTempView("pdf_pages_ocr_updates")

    spark.sql(f"""
    MERGE INTO {args.silver_table} AS t
    USING pdf_pages_ocr_updates AS s
    ON  t.doc_id = s.doc_id
    AND t.modificationTime = s.modificationTime
    AND t.page_id = s.page_id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """)

    print("[silver_pdf_ocr] MERGE completo.")


if __name__ == "__main__":
    main()
