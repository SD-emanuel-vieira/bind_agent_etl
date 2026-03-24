# src/silver_pdf_ocr.py
"""
Silver PDF OCR - Paso 1: Parsing y extracción de texto
Este script procesa los PDFs y guarda los resultados en Silver.
El enrichment (topic_llm, graphs) se hace en un script separado.

IMPORTANTE: Separa el contenido de tablas (page_table_text) del texto narrativo (page_text)
para permitir chunking diferenciado en Gold.
"""
import argparse
from pyspark.sql import functions as F
from pyspark.sql.functions import expr
from pyspark.sql.utils import AnalysisException


def str2bool(v):
    if isinstance(v, bool): return v
    if v is None: return False
    v = str(v).lower().strip()
    if v in ("1","true","t","yes","y","on"): return True
    if v in ("0","false","f","no","n","off",""): return False
    raise ValueError(f"Invalid bool: {v}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bronze_table", required=True)
    p.add_argument("--silver_table", required=True)
    p.add_argument("--image_output_path", required=False, default="")
    p.add_argument("--verify_tables", required=False, type=str2bool, default=True,
                    help="Verificar tablas con Gemini via ai_query (default: True)")
    p.add_argument("--gemini_endpoint", required=False, default="databricks-gemini-2.5-pro",
                    help="Endpoint de Gemini para verificación de tablas")
    return p.parse_args()


def ensure_silver_table(silver_table: str):
    """Crea la tabla Silver si no existe, o agrega columnas nuevas."""
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
          file_date DATE,
          file_type STRING,
          page_id INT,
          page_num INT,
          page_image_uri STRING,
          page_figure_count INT,
          page_figures ARRAY<STRUCT<y:INT, x:INT, element_id:INT, bbox:VARIANT, description:STRING, content:STRING>>,
          page_figures_text STRING,
          page_figures_enriched_text STRING,
          page_figures_enriched_error STRING,
          page_figures_enriched_ts TIMESTAMP,
          page_figures_enriched_model STRING,
          
          -- Texto narrativo (sin tablas)
          page_text STRING,
          
          -- Tablas completas (separadas para chunking sin fragmentar)
          page_table_count INT,
          page_table_text STRING,
          
          page_elements ARRAY<VARIANT>,
          
          -- topic_heuristic: primera línea del texto
          topic_heuristic STRING,
          
          -- page_segment: segmento de negocio detectado (Empresas, Corporate, Institucional, Minorista, BaaS, General)
          page_segment STRING,
          
          -- topic_llm: generado por LLM analizando la IMAGEN
          topic_llm STRING,
          topic_llm_error STRING,
          topic_llm_ts TIMESTAMP,
          topic_llm_model STRING,
          
          -- topic_content: generado por LLM analizando el TEXTO (page_text + page_table_text)
          topic_content STRING,
          topic_content_error STRING,
          topic_content_ts TIMESTAMP,
          topic_content_model STRING,
          
          -- metadata_enrich: metadatos estructurados extraídos por LLM (JSON)
          metadata_enrich STRING,
          metadata_enrich_error STRING,
          metadata_enrich_ts TIMESTAMP,
          metadata_enrich_model STRING,
          
          -- context: contexto semántico asociado al topic detectado
          context_template STRING,
          context_text STRING,
          
          error_status VARIANT,
          parsed_metadata VARIANT,
          ocr_ingest_ts TIMESTAMP
        )
        USING DELTA
        """)
        return

    # Si ya existe, asegurar columnas nuevas
    to_add = []
    for col_name, col_type in [
        ("file_date", "DATE"), ("file_type", "STRING"),
        ("page_figure_count", "INT"),
        ("page_figures", "ARRAY<STRUCT<y:INT, x:INT, element_id:INT, bbox:VARIANT, description:STRING, content:STRING>>"),
        ("page_figures_text", "STRING"),
        ("page_figures_enriched_text", "STRING"), ("page_figures_enriched_error", "STRING"),
        ("page_figures_enriched_ts", "TIMESTAMP"), ("page_figures_enriched_model", "STRING"),
        ("topic_heuristic", "STRING"), 
        ("page_segment", "STRING"),  # Segmento de negocio
        ("topic_llm", "STRING"),
        ("topic_llm_error", "STRING"), ("topic_llm_ts", "TIMESTAMP"), ("topic_llm_model", "STRING"),
        # topic_content: generado por LLM analizando el texto (no imagen)
        ("topic_content", "STRING"), ("topic_content_error", "STRING"),
        ("topic_content_ts", "TIMESTAMP"), ("topic_content_model", "STRING"),
        # metadata_enrich: metadatos estructurados extraídos por LLM (JSON)
        ("metadata_enrich", "STRING"), ("metadata_enrich_error", "STRING"),
        ("metadata_enrich_ts", "TIMESTAMP"), ("metadata_enrich_model", "STRING"),
        # context: contexto semántico asociado al topic detectado
        ("context_template", "STRING"), ("context_text", "STRING"),
        # Nuevos campos para tablas
        ("page_table_count", "INT"),
        ("page_table_text", "STRING"),
    ]:
        if col_name.lower() not in existing_cols:
            to_add.append(f"{col_name} {col_type}")

    if to_add:
        spark.sql(f"ALTER TABLE {silver_table} ADD COLUMNS ({', '.join(to_add)})")


def build_parse_expr(image_output_path: str) -> str:
    kvs = ["'version','2.0'", "'descriptionElementTypes','figure'"]
    if image_output_path and image_output_path.strip():
        kvs.append(f"'imageOutputPath','{image_output_path}'")
    return f"ai_parse_document(content, map({','.join(kvs)}))"


# ---------------------------------------------------------------------------
# Verificación de tablas con Gemini (double-pass)
# ---------------------------------------------------------------------------
VERIFY_PROMPT = (
    "You are a numerical accuracy auditor. "
    "Compare the extracted table text below against what you see in the image. "
    "Check EVERY number digit-by-digit. Look for: transposed digits (e.g. 3278 vs 3728), "
    "missing digits, extra digits, wrong decimal separators. "
    "If you find ANY discrepancy, return the CORRECTED full table text. "
    "If everything is correct, return the table text UNCHANGED. "
    "Return ONLY the table text, nothing else. No explanation, no markdown fences. "
    "Extracted table text: "
)


def build_verify_expr(gemini_endpoint: str) -> str:
    """
    Expresión SQL que envía la imagen de la página + el texto de tabla extraído
    a Gemini para verificación numérica.
    Requiere columnas: page_image_bytes (BINARY), page_table_text (STRING)
    """
    prompt_escaped = VERIFY_PROMPT.replace("'", "\\'")
    return f"""
        ai_query(
            '{gemini_endpoint}',
            CONCAT('{prompt_escaped}', page_table_text),
            files => page_image_bytes
        )
    """


def add_file_metadata(df):
    filename = F.regexp_extract(F.col("path"), r"([^/]+)$", 1)
    file_type = F.regexp_extract(filename, r"^([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)", 1)
    file_type = F.when(F.length(file_type) > 0, file_type).otherwise(F.lit(None).cast("string"))

    y1 = F.regexp_extract(filename, r"(\d{4})[_-](\d{2})[_-](\d{2})", 1)
    m1 = F.regexp_extract(filename, r"(\d{4})[_-](\d{2})[_-](\d{2})", 2)
    d1 = F.regexp_extract(filename, r"(\d{4})[_-](\d{2})[_-](\d{2})", 3)
    d2 = F.regexp_extract(filename, r"(\d{2})-(\d{2})-(\d{4})", 1)
    m2 = F.regexp_extract(filename, r"(\d{2})-(\d{2})-(\d{4})", 2)
    y2 = F.regexp_extract(filename, r"(\d{2})-(\d{2})-(\d{4})", 3)

    date_str = (
        F.when(y1 != "", F.concat_ws("-", y1, m1, d1))
         .when(y2 != "", F.concat_ws("-", y2, m2, d2))
         .otherwise(F.lit(None).cast("string"))
    )
    file_date = F.to_date(date_str, "yyyy-MM-dd")

    return df.withColumn("file_type", file_type).withColumn("file_date", file_date)


def add_topic_heuristic(df):
    first_line = F.when(
        F.col("page_text").isNotNull() & (F.length(F.trim(F.col("page_text"))) > 0),
        F.trim(F.element_at(F.split(F.col("page_text"), r"\n"), 1))
    ).otherwise(F.lit(None).cast("string"))
    
    topic_heuristic = F.when(
        F.length(first_line) > 200,
        F.concat(F.substring(first_line, 1, 197), F.lit("..."))
    ).otherwise(first_line)
    
    return df.withColumn("topic_heuristic", topic_heuristic)


def add_page_segment(df):
    """
    Detecta el segmento de negocio basándose en page_text, page_table_text y page_figures_text.
    Segmentos: Empresas, Corporate, Institucional, Minorista, BaaS
    Si encuentra más de un segmento, asigna 'General'.
    Si no encuentra ninguno, asigna NULL.
    """
    # Combinar todos los campos de texto en uno solo para buscar (case insensitive)
    combined = F.lower(F.concat_ws(
        " ",
        F.coalesce(F.col("page_text"), F.lit("")),
        F.coalesce(F.col("page_table_text"), F.lit("")),
        F.coalesce(F.col("page_figures_text"), F.lit(""))
    ))
    
    # Detectar cada segmento
    has_empresas = combined.contains("empresas")
    has_corporate = combined.contains("corporate")
    has_institucional = combined.contains("institucional")
    has_minorista = combined.contains("minorista")
    has_baas = combined.contains("baas")
    
    # Contar cuántos segmentos se encontraron
    segment_count = (
        has_empresas.cast("int") +
        has_corporate.cast("int") +
        has_institucional.cast("int") +
        has_minorista.cast("int") +
        has_baas.cast("int")
    )
    
    # Asignar valor según la lógica:
    # - Más de 1 segmento → "General"
    # - Exactamente 1 segmento → el nombre del segmento
    # - Ningún segmento → NULL
    page_segment = (
        F.when(segment_count > 1, F.lit("BIND"))
        .when(has_empresas, F.lit("Empresas"))
        .when(has_corporate, F.lit("Corporate"))
        .when(has_institucional, F.lit("Institucional"))
        .when(has_minorista, F.lit("Minorista"))
        .when(has_baas, F.lit("BaaS"))
        .otherwise(F.lit(None).cast("string"))
    )
    
    return df.withColumn("page_segment", page_segment)


def main():
    args = parse_args()

    print(f"[silver_pdf_ocr] bronze_table      = {args.bronze_table}")
    print(f"[silver_pdf_ocr] silver_table      = {args.silver_table}")
    print(f"[silver_pdf_ocr] image_output_path = {args.image_output_path}")
    print(f"[silver_pdf_ocr] verify_tables     = {args.verify_tables}")
    print(f"[silver_pdf_ocr] gemini_endpoint   = {args.gemini_endpoint}")

    ensure_silver_table(args.silver_table)

    # 1) Leer Bronze
    bronze_df = spark.table(args.bronze_table).select(
        "doc_id", "path", "modificationTime", "ingest_ts", "content"
    )

    # 2) Filtrar docs nuevos
    already = spark.table(args.silver_table).select("doc_id", "modificationTime").distinct()
    to_process = bronze_df.join(already, on=["doc_id", "modificationTime"], how="left_anti")

    if to_process.limit(1).count() == 0:
        print("[silver_pdf_ocr] No hay documentos nuevos para procesar.")
        return

    # 3) OCR parse
    parse_expr = build_parse_expr(args.image_output_path)
    parsed_docs = (
        to_process
        .withColumn("parsed", expr(parse_expr))
        .withColumn("error_status", expr("parsed:error_status"))
        .withColumn("parsed_metadata", expr("parsed:metadata"))
        .withColumn("bronze_ingest_ts", F.col("ingest_ts"))
        .withColumn("ocr_ingest_ts", F.current_timestamp())
        .select("doc_id", "path", "modificationTime", "bronze_ingest_ts",
                "parsed", "error_status", "parsed_metadata", "ocr_ingest_ts")
    )

    parsed_docs_ok = parsed_docs.where("try_cast(error_status AS STRING) IS NULL")
    parsed_docs_ok = add_file_metadata(parsed_docs_ok)

    # 4) Expandir a 1 fila por página
    docs_base = parsed_docs_ok.select(
        "doc_id", "path", "modificationTime", "bronze_ingest_ts",
        "file_date", "file_type", "error_status", "parsed_metadata", "ocr_ingest_ts",
        expr("try_cast(parsed:document:elements AS ARRAY<VARIANT>) AS elements"),
        expr("try_cast(parsed:document:pages AS ARRAY<VARIANT>) AS pages")
    )

    pages = (
        docs_base.select("doc_id", "modificationTime", expr("explode_outer(pages) AS p"))
        .select("doc_id", "modificationTime",
                expr("try_cast(p:id AS INT) AS page_id"),
                expr("try_cast(p:image_uri AS STRING) AS page_image_uri"))
    )

    elements = (
        docs_base.select("doc_id", "path", "modificationTime", "bronze_ingest_ts", "elements")
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
              OR (description IS NOT NULL AND length(trim(description)) > 0)
            )
        """)
    )

    # Agregación por página - SEPARANDO TABLAS DEL TEXTO NARRATIVO
    per_page = (
        elements
        .groupBy("doc_id", "path", "modificationTime", "bronze_ingest_ts", "page_id")
        .agg(
            # Texto narrativo (NO tablas), ordenado por element_id
            expr("""
              transform(
                array_sort(filter(collect_list(
                  CASE WHEN element_type != 'table' 
                       AND content IS NOT NULL 
                       AND length(trim(content)) > 0
                       THEN named_struct('id', element_id, 'content', content) 
                  END
                ), x -> x IS NOT NULL)),
                x -> x.content
              ) AS text_contents_sorted
            """),
            
            # Tablas completas, ordenadas por posición (y, x)
            expr("""
              array_sort(
                filter(collect_list(
                  CASE WHEN element_type = 'table' 
                       AND content IS NOT NULL 
                       AND length(trim(content)) > 0
                       THEN named_struct('y', y, 'x', x, 'element_id', element_id, 'content', content) 
                  END
                ), x -> x IS NOT NULL),
                (l, r) -> CASE
                  WHEN l.y < r.y THEN -1 WHEN l.y > r.y THEN 1
                  WHEN l.x < r.x THEN -1 WHEN l.x > r.x THEN 1
                  ELSE 0 
                END
              ) AS tables_sorted
            """),
            
            # Elementos de la página (todos)
            expr("collect_list(element) AS page_elements"),
            
            # Figuras/gráficos detectados
            expr("""
              filter(collect_list(
                CASE WHEN element_type = 'figure' THEN named_struct(
                  'y', y, 'x', x, 'element_id', element_id, 'bbox', bbox,
                  'description', description, 'content', content
                ) END
              ), x -> x IS NOT NULL) AS page_figures_raw
            """)
        )
        # Procesar figuras
        .withColumn("page_figures", expr("""
            array_sort(page_figures_raw, (l, r) -> CASE
                WHEN l.y < r.y THEN -1 WHEN l.y > r.y THEN 1
                WHEN l.x < r.x THEN -1 WHEN l.x > r.x THEN 1
                WHEN l.element_id < r.element_id THEN -1
                WHEN l.element_id > r.element_id THEN 1 ELSE 0 END)
        """))
        .withColumn("page_figure_count", expr("size(page_figures)"))
        .withColumn("page_figures_text", 
            expr("concat_ws('\n\n', transform(page_figures, f -> coalesce(f.description, f.content)))"))
        .drop("page_figures_raw")
        
        # Texto narrativo (sin tablas)
        .withColumn("page_text", expr("concat_ws('\n\n', text_contents_sorted)"))
        .drop("text_contents_sorted")
        
        # Tablas - cada tabla separada por doble salto de línea con marcador
        .withColumn("page_table_count", expr("size(tables_sorted)"))
        .withColumn("page_table_text", 
            expr("concat_ws('\n\n---TABLE_SEPARATOR---\n\n', transform(tables_sorted, t -> t.content))"))
        .drop("tables_sorted")
        
        .withColumn("page_num", expr("page_id + 1"))
    )

    doc_meta = (
        docs_base.select("doc_id", "modificationTime", "file_date", "file_type",
                        "error_status", "parsed_metadata", "ocr_ingest_ts")
        .dropDuplicates(["doc_id", "modificationTime"])
    )

    updates = (
        per_page
        .join(pages, on=["doc_id", "modificationTime", "page_id"], how="left")
        .join(doc_meta, on=["doc_id", "modificationTime"], how="left")
        .select(
            "doc_id", "path", "modificationTime", "bronze_ingest_ts",
            "file_date", "file_type", "page_id", "page_num", "page_image_uri",
            "page_figure_count", "page_figures", "page_figures_text",
            "page_text",
            "page_table_count", "page_table_text",
            "page_elements", "error_status", "parsed_metadata", "ocr_ingest_ts"
        )
    )

    # Agregar topic_heuristic (basado en page_text, no tablas)
    updates = add_topic_heuristic(updates)
    
    # Agregar page_segment (segmento de negocio detectado)
    updates = add_page_segment(updates)

    # ------------------------------------------------------------------
    # 4b) Verificación de tablas con Gemini (double-pass)
    #     Solo para páginas que tienen tablas Y una imagen disponible.
    #     Lee la imagen desde el Volume, la envía con el texto de tabla
    #     extraído a Gemini, y reemplaza page_table_text si Gemini
    #     devuelve una corrección válida.
    # ------------------------------------------------------------------
    if args.verify_tables and args.image_output_path:
        from pyspark.sql.functions import udf, col, lit
        from pyspark.sql.types import BinaryType

        # UDF: lee la imagen desde el path en el Volume
        @udf(BinaryType())
        def read_image_bytes(image_uri):
            if not image_uri:
                return None
            try:
                with open(image_uri, "rb") as f:
                    return f.read()
            except Exception:
                return None

        # Separar páginas con tablas vs sin tablas
        with_tables = updates.where("page_table_count > 0 AND page_image_uri IS NOT NULL")
        without_tables = updates.where("page_table_count = 0 OR page_image_uri IS NULL")

        table_page_count = with_tables.count()
        print(f"[silver_pdf_ocr] Páginas con tablas a verificar: {table_page_count}")

        if table_page_count > 0:
            verify_expr = build_verify_expr(args.gemini_endpoint)

            # Flujo lineal: leer imagen → verificar → fallback a original si falla
            verified = (
                with_tables
                .withColumn("page_image_bytes", read_image_bytes(col("page_image_uri")))
                # Solo llamar a Gemini si pudimos leer la imagen
                .withColumn("verified_table_text",
                    F.when(
                        F.col("page_image_bytes").isNotNull(),
                        expr(verify_expr)
                    ).otherwise(F.lit(None).cast("string"))
                )
                # Usar versión verificada si es válida, sino mantener original
                .withColumn("page_table_text",
                    F.when(
                        F.col("verified_table_text").isNotNull()
                        & (F.length(F.trim(F.col("verified_table_text"))) > 0)
                        & (~F.col("verified_table_text").contains("error")),
                        F.col("verified_table_text")
                    ).otherwise(F.col("page_table_text"))
                )
                .drop("verified_table_text", "page_image_bytes")
            )

            updates = verified.unionByName(without_tables)
            print(f"[silver_pdf_ocr] Verificación de tablas completa.")
        else:
            print("[silver_pdf_ocr] No hay páginas con tablas para verificar.")
    else:
        if not args.verify_tables:
            print("[silver_pdf_ocr] Verificación de tablas desactivada (--verify_tables false).")
        elif not args.image_output_path:
            print("[silver_pdf_ocr] Verificación de tablas requiere --image_output_path para leer imágenes.")

    # Agregar columnas de enrichment como NULL
    updates = (
        updates
        .withColumn("page_figures_enriched_text", F.lit(None).cast("string"))
        .withColumn("page_figures_enriched_error", F.lit(None).cast("string"))
        .withColumn("page_figures_enriched_ts", F.lit(None).cast("timestamp"))
        .withColumn("page_figures_enriched_model", F.lit(None).cast("string"))
        .withColumn("topic_llm", F.lit(None).cast("string"))
        .withColumn("topic_llm_error", F.lit(None).cast("string"))
        .withColumn("topic_llm_ts", F.lit(None).cast("timestamp"))
        .withColumn("topic_llm_model", F.lit(None).cast("string"))
        .withColumn("topic_content", F.lit(None).cast("string"))
        .withColumn("topic_content_error", F.lit(None).cast("string"))
        .withColumn("topic_content_ts", F.lit(None).cast("timestamp"))
        .withColumn("topic_content_model", F.lit(None).cast("string"))
        .withColumn("metadata_enrich", F.lit(None).cast("string"))
        .withColumn("metadata_enrich_error", F.lit(None).cast("string"))
        .withColumn("metadata_enrich_ts", F.lit(None).cast("timestamp"))
        .withColumn("metadata_enrich_model", F.lit(None).cast("string"))
        .withColumn("context_template", F.lit(None).cast("string"))
        .withColumn("context_text", F.lit(None).cast("string"))
    )

    # 5) MERGE
    from collections import Counter
    dups = [c for c, n in Counter(updates.columns).items() if n > 1]
    print("DUP COLS:", dups)

    updates.createOrReplaceTempView("pdf_pages_ocr_updates")

    spark.sql(f"""
    MERGE INTO {args.silver_table} AS t
    USING pdf_pages_ocr_updates AS s
    ON t.doc_id = s.doc_id AND t.modificationTime = s.modificationTime AND t.page_id = s.page_id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """)

    print("[silver_pdf_ocr] MERGE completo. Ejecutar silver_pdf_enrich.py para enrichment.")


if __name__ == "__main__":
    main()