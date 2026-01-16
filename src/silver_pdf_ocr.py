# src/silver_pdf_ocr.py
import argparse
from pyspark.sql import functions as F
from pyspark.sql.functions import expr
from pyspark.sql.utils import AnalysisException
import pyspark.sql.functions as F


def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("1","true","t","yes","y","on"): return True
    if v in ("0","false","f","no","n","off"): return False
    raise ValueError(f"Invalid bool: {v}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bronze_table", required=True)      # bind_agent.docs.pdf_bronze
    p.add_argument("--silver_table", required=True)      # bind_agent.docs.pdf_ocr_silver
    p.add_argument("--image_output_path", required=False, default="")  # /Volumes/... (opcional)

    # Enrichment (opcional): describir gráficos con un modelo multimodal (ai_query)
    p.add_argument("--do_enrich_graphs", action="store_true", help="Si se setea, enriquece páginas con gráficos usando ai_query sobre la imagen de la página.")
    # parser.add_argument("--do_enrich_graphs", type=str2bool, default=False)
    p.add_argument("--enrich_model", required=False, default="databricks-gemma-3-12b", help="Nombre del endpoint/modelo para ai_query (multimodal).")

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

          page_figure_count INT,
          page_figures ARRAY<STRUCT<y:INT, x:INT, element_id:INT, bbox:VARIANT, description:STRING, content:STRING>>,
          page_figures_text STRING,

          -- Enrichment (solo páginas seleccionadas)
          page_figures_enriched_text STRING,
          page_figures_enriched_error STRING,
          page_figures_enriched_ts TIMESTAMP,
          page_figures_enriched_model STRING,

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

    if "page_figure_count" not in existing_cols:
        to_add.append("page_figure_count INT")
    if "page_figures" not in existing_cols:
        to_add.append("page_figures ARRAY<STRUCT<y:INT, x:INT, element_id:INT, bbox:VARIANT, description:STRING, content:STRING>>")
    if "page_figures_text" not in existing_cols:
        to_add.append("page_figures_text STRING")

    if "page_figures_enriched_text" not in existing_cols:
        to_add.append("page_figures_enriched_text STRING")
    if "page_figures_enriched_error" not in existing_cols:
        to_add.append("page_figures_enriched_error STRING")
    if "page_figures_enriched_ts" not in existing_cols:
        to_add.append("page_figures_enriched_ts TIMESTAMP")
    if "page_figures_enriched_model" not in existing_cols:
        to_add.append("page_figures_enriched_model STRING")

    if to_add:
        spark.sql(f"ALTER TABLE {silver_table} ADD COLUMNS ({', '.join(to_add)})")


def build_parse_expr(image_output_path: str) -> str:
    kvs = [
        "'version','2.0'",
        "'descriptionElementTypes','figure'"  # habilita descripciones AI para figuras
    ]
    if image_output_path and image_output_path.strip():
        kvs.append(f"'imageOutputPath','{image_output_path}'")

    return f"ai_parse_document(content, map({','.join(kvs)}))"



def enrich_graph_pages(pages_df, image_output_path: str, enrich_model: str):
    """
    Silver enrichment (1 fila por página):
      - Solo aplica a páginas con gráficos, usando estos filtros:
          * page_figure_count > 1
          * lower(page_figures_text) LIKE '%graph%'
      - Usa ai_query multimodal sobre la imagen de la página (page_image_uri) para generar una
        interpretación más detallada en texto.

    Devuelve el mismo DF con columnas nuevas:
      - page_figures_enriched_text (STRING)
      - page_figures_enriched_error (STRING)
      - page_figures_enriched_ts (TIMESTAMP)
      - page_figures_enriched_model (STRING)
    """
    # orig_cols = pages_df.columns
    base = (
        pages_df
        .withColumn("page_figures_enriched_text", F.lit(None).cast("string"))
        .withColumn("page_figures_enriched_error", F.lit(None).cast("string"))
        .withColumn("page_figures_enriched_ts", F.lit(None).cast("timestamp"))
        .withColumn("page_figures_enriched_model", F.lit(None).cast("string"))
    )

    if not image_output_path or not image_output_path.strip():
        # No hay imágenes disponibles para enriquecer
        return base

    candidates = (
        base
        .filter(F.col("page_figure_count") > 1)
        # .filter(F.lower(F.coalesce(F.col("page_figures_text"), F.lit(""))).like("%graph%"))
        .filter(F.col("page_image_uri").isNotNull())
        .select("doc_id", "modificationTime", "page_id", "page_image_uri", "page_figures_text")
    )
    
    txt = F.concat(
        F.lit(" "),
        F.lower(F.coalesce(F.col("page_figures_text"), F.lit(""))),
        F.lit(" ")
    )

    is_graph_like = (
        txt.like("% graph %")  |
        txt.like("% graphs %") |
        txt.like("% bar %")    |
        txt.like("% bars %")
    )

    candidates = (
        base
        .filter(F.col("page_figure_count") > 1)
        .filter(F.col("page_image_uri").isNotNull())
        .filter(is_graph_like)
        .select("doc_id", "modificationTime", "page_id", "page_image_uri", "page_figures_text")
    )

    # Evitar leer imágenes / invocar modelo si no hay candidatos
    if candidates.limit(1).count() == 0:
        return base

    # Cargar imágenes renderizadas desde el Volume y unir por path
    imgs = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load(image_output_path)
        .select(
            # normalizar posibles prefijos file: / dbfs:
            F.regexp_replace(F.regexp_replace(F.col("path"), r"^file:", ""), r"^dbfs:", "").alias("page_image_uri"),
            F.col("content").alias("image_content"),
        )
        .filter(F.lower(F.col("page_image_uri")).rlike(r"\.(jpeg|jpg|png)$"))
    )

    cand_with_img = (
        candidates.join(imgs, on="page_image_uri", how="left")
        .filter(F.col("image_content").isNotNull())
    )

    # Prompt: forzar salida en español, enumerada, sin inventar números
    enrich_prompt = F.concat(
        F.lit(
            "Analiza la imagen adjunta (una página de un PDF) y enfócate SOLO en los gráficos (charts/plots).\n"
            "Tu tarea:\n"
            "1) Identificar cada gráfico presente en la página y enumerarlos como 'Gráfico 1', 'Gráfico 2', etc.\n"
            "2) Para cada gráfico, describe: tipo (barras/líneas/etc), qué representa el eje X y el eje Y (y unidades si se ven),\n"
            "   series/leyenda si existen, y una lectura en lenguaje natural (tendencias, máximos/mínimos, cambios relevantes).\n"
            "3) Si hay valores numéricos legibles, inclúyelos. Si NO son legibles, indica explícitamente 'no legible' y no inventes.\n\n"
            "Contexto (descripción preliminar):\n"
        ),
        F.coalesce(F.col("page_figures_text"), F.lit("")),
        F.lit("\n\nDevuelve SOLO el texto final, en español, con separadores claros entre gráficos.")
    )

    enriched = (
        cand_with_img
        .withColumn("enrich_prompt", enrich_prompt)
        .withColumn(
            "enrich_out",
            expr(
                "ai_query("
                f"'{enrich_model}', "
                "enrich_prompt, "
                "files => image_content, "
                "failOnError => false"
                ")"
            )
        )
        .select(
            "doc_id", "modificationTime", "page_id",
            F.col("enrich_out.result").cast("string").alias("page_figures_enriched_text_new"),
            F.col("enrich_out.errorMessage").cast("string").alias("page_figures_enriched_error_new"),
            F.current_timestamp().alias("page_figures_enriched_ts_new"),
            F.lit(enrich_model).alias("page_figures_enriched_model_new"),
        )
    )

    keys = ["doc_id", "modificationTime", "page_id"]
    out = (
        base
        .join(enriched, on=keys, how="left")
        .withColumn(
            "page_figures_enriched_text",
            F.coalesce(F.col("page_figures_enriched_text_new"), F.col("page_figures_enriched_text"))
        )
        .withColumn(
            "page_figures_enriched_error",
            F.coalesce(F.col("page_figures_enriched_error_new"), F.col("page_figures_enriched_error"))
        )
        .withColumn(
            "page_figures_enriched_ts",
            F.coalesce(F.col("page_figures_enriched_ts_new"), F.col("page_figures_enriched_ts"))
        )
        .withColumn(
            "page_figures_enriched_model",
            F.coalesce(F.col("page_figures_enriched_model_new"), F.col("page_figures_enriched_model"))
        )
        .drop(
            "page_figures_enriched_text_new",
            "page_figures_enriched_error_new",
            "page_figures_enriched_ts_new",
            "page_figures_enriched_model_new",
        )
    )
    
    return out


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
    # Importante:
    # - Incluimos `description` para poder contar e interpretar figuras (gráficos/diagramas).
    # - NO filtramos solo por `content`, porque las figuras pueden venir con `content` vacío pero `description` poblada.
    elements = (
        docs_base
        .select("doc_id", "path", "modificationTime", "bronze_ingest_ts", "elements")
        .withColumn("e", expr("explode_outer(elements)"))
        .select(
            "doc_id", "path", "modificationTime", "bronze_ingest_ts",
            expr("try_cast(e:id AS INT) AS element_id"),
            expr("try_cast(e:type AS STRING) AS element_type"),
            expr("e AS element"),
            expr("e:bbox AS bbox"),
            expr("try_cast(e:bbox[0]:page_id AS INT) AS page_id"),
            expr("try_cast(e:content AS STRING) AS content"),
            expr("try_cast(e:description AS STRING) AS description"),
            expr("try_cast(e:bbox[0]:coord[0] AS INT) AS x"),
            expr("try_cast(e:bbox[0]:coord[1] AS INT) AS y")
        )
        .where("""
            page_id IS NOT NULL AND (
              (content IS NOT NULL AND length(trim(content)) > 0)
              OR
              (description IS NOT NULL AND length(trim(description)) > 0)
            )
        """)
    )

    # Aggregate por página (NO incluir VARIANT en groupBy) (NO incluir VARIANT en groupBy)
    per_page = (
        elements
        .groupBy("doc_id", "path", "modificationTime", "bronze_ingest_ts", "page_id")
        .agg(
            # Texto "normal" (solo content no vacío), ordenado por element_id
            expr("""
              transform(
                array_sort(
                  filter(
                    collect_list(
                      CASE WHEN content IS NOT NULL AND length(trim(content)) > 0
                           THEN named_struct('id', element_id, 'content', content)
                      END
                    ),
                    x -> x IS NOT NULL
                  )
                ),
                x -> x.content
              ) AS contents_sorted
            """),
            # Elementos de la página (incluye figuras con description)
            expr("collect_list(element) AS page_elements"),
            # Figuras/gráficos detectados (para conteo + texto interpretado), ordenados por (y, x)
            expr("""
              filter(
                collect_list(
                  CASE WHEN element_type = 'figure' THEN named_struct(
                    'y', y,
                    'x', x,
                    'element_id', element_id,
                    'bbox', bbox,
                    'description', description,
                    'content', content
                  ) END
                ),
                x -> x IS NOT NULL
              ) AS page_figures_raw
            """)
        )
        .withColumn("page_figures", expr("""
            array_sort(
                page_figures_raw,
                (l, r) -> CASE
                WHEN l.y < r.y THEN -1
                WHEN l.y > r.y THEN 1
                WHEN l.x < r.x THEN -1
                WHEN l.x > r.x THEN 1
                WHEN l.element_id < r.element_id THEN -1
                WHEN l.element_id > r.element_id THEN 1
                ELSE 0
                END
            )
            """))
        .withColumn("page_figure_count", expr("size(page_figures)"))
        .withColumn(
            "page_figures_text",
            expr("concat_ws('\n\n', transform(page_figures, f -> coalesce(f.description, f.content)))")
        )
        .drop("page_figures_raw")
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
            "page_figure_count", "page_figures", "page_figures_text",
            "page_text", "page_elements",
            "error_status", "parsed_metadata",
            "ocr_ingest_ts"
        )
    )


    # -----------------------------
    # 4b) Silver enrichment (opcional): solo páginas con gráficos
    # -----------------------------
    if args.do_enrich_graphs:
        updates = enrich_graph_pages(updates, args.image_output_path, args.enrich_model)
    else:
        updates = (
            updates
            .withColumn("page_figures_enriched_text", F.lit(None).cast("string"))
            .withColumn("page_figures_enriched_error", F.lit(None).cast("string"))
            .withColumn("page_figures_enriched_ts", F.lit(None).cast("timestamp"))
            .withColumn("page_figures_enriched_model", F.lit(None).cast("string"))
        )

    # -----------------------------
    # 5) MERGE idempotente a Silver
    # -----------------------------
    # # Debugg
    from collections import Counter
    dups = [c for c, n in Counter(updates.columns).items() if n > 1]
    
    print("DUP COLS:", dups)

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
