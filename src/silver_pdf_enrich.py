# src/silver_pdf_enrich.py
"""
Silver PDF Enrich - Paso 2: Enrichment con LLM
Este script lee páginas de Silver que necesitan enrichment y las actualiza.
Debe ejecutarse DESPUÉS de silver_pdf_ocr.py para que las imágenes estén disponibles.

Enrichments disponibles:
- topic_llm: Analiza la IMAGEN de la página (visión)
- topic_content: Analiza el TEXTO de la página (page_text + page_table_text)
- graphs: Describe gráficos detectados
- metadata_enrich: Extrae metadatos estructurados (keywords, período, categoría, métricas, entidades)
"""
import argparse
from pyspark.sql import functions as F
from pyspark.sql.functions import expr


def str2bool(v):
    if isinstance(v, bool): return v
    if v is None: return False
    v = str(v).lower().strip()
    if v in ("1","true","t","yes","y","on"): return True
    if v in ("0","false","f","no","n","off",""): return False
    raise ValueError(f"Invalid bool: {v}")


def normalize_path_col(col):
    """Normaliza paths removiendo prefijos file:, dbfs:, file:/dbfs:"""
    normalized = F.regexp_replace(col, r"^file:/dbfs:", "")
    normalized = F.regexp_replace(normalized, r"^file:", "")
    normalized = F.regexp_replace(normalized, r"^dbfs:", "")
    return normalized


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--silver_table", required=True)
    p.add_argument("--image_output_path", required=False, default="")
    
    # Enrichment de gráficos (requiere imágenes)
    p.add_argument("--do_enrich_graphs", type=str2bool, default=False)
    p.add_argument("--enrich_model", default="databricks-gemma-3-12b")
    
    # Enrichment de topic por IMAGEN (requiere imágenes)
    p.add_argument("--do_enrich_topic", type=str2bool, default=False)
    p.add_argument("--topic_model", default="databricks-gemma-3-12b")
    p.add_argument("--topic_min_chars", type=int, default=50)
    
    # Enrichment de topic por TEXTO (NO requiere imágenes)
    p.add_argument("--do_enrich_topic_content", type=str2bool, default=False)
    p.add_argument("--topic_content_model", default="databricks-gemma-3-12b")
    p.add_argument("--topic_content_min_chars", type=int, default=50)
    
    # Enrichment de METADATA (NO requiere imágenes)
    p.add_argument("--do_enrich_metadata", type=str2bool, default=False)
    p.add_argument("--metadata_model", default="databricks-gemma-3-12b")
    p.add_argument("--metadata_min_chars", type=int, default=100)
    
    return p.parse_args()


def load_images_df(image_output_path: str):
    """Carga las imágenes desde el Volume y normaliza los paths."""
    return (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load(image_output_path)
        .select(
            normalize_path_col(F.col("path")).alias("image_path_normalized"),
            F.col("content").alias("image_content"),
        )
        .filter(F.lower(F.col("image_path_normalized")).rlike(r"\.(jpeg|jpg|png)$"))
    )


def enrich_topic_llm(silver_table: str, image_output_path: str, topic_model: str, min_chars: int):
    """
    Genera topic_llm analizando la IMAGEN de la página.
    Requiere que las imágenes estén disponibles.
    """
    print(f"[enrich_topic_llm] Buscando páginas con >= {min_chars} chars sin topic_llm...")
    
    # Leer páginas candidatas desde Silver (que aún no tienen topic_llm)
    # Solo páginas CON texto suficiente (>= min_chars)
    candidates = (
        spark.table(silver_table)
        .filter(F.col("topic_llm").isNull())
        .filter(F.col("page_image_uri").isNotNull())
        .filter(
            F.col("page_text").isNotNull() & 
            (F.length(F.col("page_text")) >= min_chars)
        )
        .select("doc_id", "modificationTime", "page_id", "page_image_uri")
        .withColumn("image_path_normalized", normalize_path_col(F.col("page_image_uri")))
    )
    
    cand_count = candidates.count()
    if cand_count == 0:
        print("[enrich_topic_llm] No hay páginas candidatas para enrichment.")
        return
    
    print(f"[enrich_topic_llm] Encontradas {cand_count} páginas para procesar.")
    
    # Cargar imágenes
    imgs = load_images_df(image_output_path)
    
    # Debug: mostrar ejemplos de paths
    print("[enrich_topic_llm] Ejemplo de paths en candidatos:")
    candidates.select("image_path_normalized").show(3, truncate=False)
    print("[enrich_topic_llm] Ejemplo de paths en imágenes:")
    imgs.select("image_path_normalized").show(3, truncate=False)
    
    # Join con imágenes
    cand_with_img = (
        candidates
        .join(imgs, on="image_path_normalized", how="inner")
    )
    
    matched_count = cand_with_img.count()
    if matched_count == 0:
        print("[enrich_topic_llm] ADVERTENCIA: No se encontraron imágenes que coincidan.")
        return
    
    print(f"[enrich_topic_llm] {matched_count} páginas con imagen encontrada. Invocando LLM...")
    
    # Prompt
    topic_prompt = F.lit(
        "Analiza la imagen adjunta (una página de un documento PDF).\n"
        "Tu tarea es identificar el TÓPICO o TEMA principal de esta página.\n\n"
        "REGLAS ESTRICTAS:\n"
        "1. Responde ÚNICAMENTE con el tópico, SIN explicaciones adicionales.\n"
        "2. Máximo 8 palabras.\n"
        "3. Sé específico pero conciso (ej: 'Gráfico de ventas Q3 2024', 'Organigrama departamento TI').\n"
        "4. Si la página está en blanco, responde: 'Página sin contenido'.\n"
        "5. Responde en español.\n\n"
        "TÓPICO:"
    )
    
    enriched = (
        cand_with_img
        .withColumn("topic_prompt", topic_prompt)
        .withColumn(
            "topic_out",
            expr(f"""
                ai_query(
                    '{topic_model}',
                    topic_prompt,
                    files => image_content,
                    failOnError => false
                )
            """)
        )
        .withColumn(
            "topic_llm_raw",
            F.trim(F.col("topic_out.result")).cast("string")
        )
        .withColumn(
            "topic_llm",
            F.when(
                F.col("topic_llm_raw").isNotNull(),
                F.concat_ws(" ", F.slice(F.split(F.col("topic_llm_raw"), r"\s+"), 1, 8))
            ).otherwise(F.lit(None).cast("string"))
        )
        .withColumn("topic_llm_error", F.col("topic_out.errorMessage").cast("string"))
        .withColumn("topic_llm_ts", F.current_timestamp())
        .withColumn("topic_llm_model", F.lit(topic_model))
        .select(
            "doc_id", "modificationTime", "page_id",
            "topic_llm", "topic_llm_error", "topic_llm_ts", "topic_llm_model"
        )
    )
    
    # Crear temp view y hacer UPDATE
    enriched.createOrReplaceTempView("topic_enriched_updates")
    
    spark.sql(f"""
    MERGE INTO {silver_table} AS t
    USING topic_enriched_updates AS s
    ON t.doc_id = s.doc_id 
       AND t.modificationTime = s.modificationTime 
       AND t.page_id = s.page_id
    WHEN MATCHED THEN UPDATE SET
        t.topic_llm = s.topic_llm,
        t.topic_llm_error = s.topic_llm_error,
        t.topic_llm_ts = s.topic_llm_ts,
        t.topic_llm_model = s.topic_llm_model
    """)
    
    print(f"[enrich_topic_llm] Actualizado topic_llm para {matched_count} páginas.")


def enrich_topic_content(silver_table: str, topic_content_model: str, min_chars: int):
    """
    Genera topic_content analizando el TEXTO de la página (page_text + page_table_text).
    NO requiere imágenes, solo contenido textual.
    """
    print(f"[enrich_topic_content] Buscando páginas con contenido textual >= {min_chars} chars sin topic_content...")
    
    # Leer páginas candidatas
    # Concatenar page_text y page_table_text para evaluar contenido total
    silver_df = spark.table(silver_table)
    
    # Verificar si la columna topic_content existe, si no, todas las filas son candidatas
    if "topic_content" not in silver_df.columns:
        print("[enrich_topic_content] Columna topic_content no existe, se procesarán todas las páginas con contenido.")
        has_topic_content = F.lit(False)
    else:
        has_topic_content = F.col("topic_content").isNotNull()
    
    # Verificar si page_table_text existe
    if "page_table_text" not in silver_df.columns:
        combined_content = F.coalesce(F.col("page_text"), F.lit(""))
    else:
        combined_content = F.concat_ws(
            "\n\n---\n\n",
            F.coalesce(F.col("page_text"), F.lit("")),
            F.coalesce(F.col("page_table_text"), F.lit(""))
        )
    
    content_length = F.length(F.trim(combined_content))
    
    candidates = (
        silver_df
        .withColumn("combined_content", combined_content)
        .withColumn("content_length", content_length)
        .filter(~has_topic_content)  # Sin topic_content
        .filter(F.col("content_length") >= min_chars)  # Con contenido suficiente
        .select("doc_id", "modificationTime", "page_id", "combined_content")
    )
    
    cand_count = candidates.count()
    if cand_count == 0:
        print("[enrich_topic_content] No hay páginas candidatas para enrichment.")
        return
    
    print(f"[enrich_topic_content] Encontradas {cand_count} páginas para procesar. Invocando LLM...")
    
    # Prompt para análisis de texto
    # Truncamos el contenido a ~4000 chars para no exceder límites del modelo
    topic_prompt = F.concat(
        F.lit(
            "Analiza el siguiente contenido de una página de documento PDF.\n"
            "Tu tarea es identificar el TÓPICO o TEMA principal de esta página.\n\n"
            "REGLAS ESTRICTAS:\n"
            "1. Responde ÚNICAMENTE con el tópico, SIN explicaciones adicionales.\n"
            "2. Máximo 8 palabras.\n"
            "3. Sé específico pero conciso (ej: 'Resultados financieros Q3 2024', 'Estructura organizacional', 'Tabla de indicadores operativos').\n"
            "4. Si hay tablas, menciona el tipo de datos que contienen.\n"
            "5. Responde en español.\n\n"
            "CONTENIDO DE LA PÁGINA:\n"
        ),
        F.substring(F.col("combined_content"), 1, 4000),  # Truncar a 4000 chars
        F.lit("\n\nTÓPICO:")
    )
    
    enriched = (
        candidates
        .withColumn("topic_prompt", topic_prompt)
        .withColumn(
            "topic_out",
            expr(f"""
                ai_query(
                    '{topic_content_model}',
                    topic_prompt,
                    failOnError => false
                )
            """)
        )
        .withColumn(
            "topic_content_raw",
            F.trim(F.col("topic_out.result")).cast("string")
        )
        .withColumn(
            "topic_content",
            F.when(
                F.col("topic_content_raw").isNotNull(),
                F.concat_ws(" ", F.slice(F.split(F.col("topic_content_raw"), r"\s+"), 1, 8))
            ).otherwise(F.lit(None).cast("string"))
        )
        .withColumn("topic_content_error", F.col("topic_out.errorMessage").cast("string"))
        .withColumn("topic_content_ts", F.current_timestamp())
        .withColumn("topic_content_model", F.lit(topic_content_model))
        .select(
            "doc_id", "modificationTime", "page_id",
            "topic_content", "topic_content_error", "topic_content_ts", "topic_content_model"
        )
    )
    
    # Crear temp view y hacer MERGE
    enriched.createOrReplaceTempView("topic_content_updates")
    
    spark.sql(f"""
    MERGE INTO {silver_table} AS t
    USING topic_content_updates AS s
    ON t.doc_id = s.doc_id 
       AND t.modificationTime = s.modificationTime 
       AND t.page_id = s.page_id
    WHEN MATCHED THEN UPDATE SET
        t.topic_content = s.topic_content,
        t.topic_content_error = s.topic_content_error,
        t.topic_content_ts = s.topic_content_ts,
        t.topic_content_model = s.topic_content_model
    """)
    
    print(f"[enrich_topic_content] Actualizado topic_content para {cand_count} páginas.")


def enrich_metadata(silver_table: str, metadata_model: str, min_chars: int):
    """
    Extrae metadatos estructurados del contenido (page_text + page_table_text) usando LLM.
    Genera un JSON con: keywords, data_period, content_category, table_metrics, entities.
    NO requiere imágenes.
    """
    print(f"[enrich_metadata] Buscando páginas con contenido >= {min_chars} chars sin metadata_enrich...")
    
    silver_df = spark.table(silver_table)
    
    # Verificar si la columna metadata_enrich existe
    if "metadata_enrich" not in silver_df.columns:
        print("[enrich_metadata] Columna metadata_enrich no existe, se procesarán todas las páginas con contenido.")
        has_metadata = F.lit(False)
    else:
        has_metadata = F.col("metadata_enrich").isNotNull()
    
    # Verificar si page_table_text existe
    if "page_table_text" not in silver_df.columns:
        combined_content = F.coalesce(F.col("page_text"), F.lit(""))
    else:
        combined_content = F.concat_ws(
            "\n\n---TABLA---\n\n",
            F.coalesce(F.col("page_text"), F.lit("")),
            F.coalesce(F.col("page_table_text"), F.lit(""))
        )
    
    content_length = F.length(F.trim(combined_content))
    
    candidates = (
        silver_df
        .withColumn("combined_content", combined_content)
        .withColumn("content_length", content_length)
        .filter(~has_metadata)  # Sin metadata_enrich
        .filter(F.col("content_length") >= min_chars)  # Con contenido suficiente
        .select("doc_id", "modificationTime", "page_id", "combined_content")
    )
    
    cand_count = candidates.count()
    if cand_count == 0:
        print("[enrich_metadata] No hay páginas candidatas para enrichment.")
        return
    
    print(f"[enrich_metadata] Encontradas {cand_count} páginas para procesar. Invocando LLM...")
    
    # Prompt para extraer metadatos estructurados
    metadata_prompt = F.concat(
        F.lit(
            "Analiza el siguiente contenido de una página de documento y extrae metadatos estructurados.\n\n"
            "RESPONDE ÚNICAMENTE con un objeto JSON válido (sin explicaciones ni texto adicional) con estos campos:\n"
            "{\n"
            '  "keywords": ["palabra1", "palabra2", ...],\n'
            '  "data_period": "Q3 2024" o "Oct 2024" o "2024" o null,\n'
            '  "content_category": "financiero" o "operativo" o "estrategico" o "legal" o "otro",\n'
            '  "table_metrics": ["metrica1", "metrica2", ...] o [],\n'
            '  "entities": ["entidad1", "entidad2", ...] o []\n'
            "}\n\n"
            "INSTRUCCIONES:\n"
            "- keywords: 5-10 términos clave relevantes del contenido (sustantivos, métricas, conceptos importantes)\n"
            "- data_period: período temporal de los datos si se menciona (trimestre, mes, año). null si no hay.\n"
            "- content_category: categoría principal del contenido\n"
            "- table_metrics: si hay tablas, lista las métricas/indicadores mencionados (ej: 'Ingresos', 'EBITDA', 'Margen bruto')\n"
            "- entities: empresas, organizaciones, países, monedas mencionadas\n\n"
            "CONTENIDO:\n"
        ),
        F.substring(F.col("combined_content"), 1, 3500),  # Truncar para dejar espacio al prompt
        F.lit("\n\nJSON:")
    )
    
    enriched = (
        candidates
        .withColumn("metadata_prompt", metadata_prompt)
        .withColumn(
            "metadata_out",
            expr(f"""
                ai_query(
                    '{metadata_model}',
                    metadata_prompt,
                    failOnError => false
                )
            """)
        )
        .withColumn(
            "metadata_raw",
            F.trim(F.col("metadata_out.result")).cast("string")
        )
        # Limpiar el JSON: remover posibles backticks de markdown
        .withColumn(
            "metadata_enrich",
            F.regexp_replace(
                F.regexp_replace(F.col("metadata_raw"), r"^```json\s*", ""),
                r"\s*```$", ""
            )
        )
        .withColumn("metadata_enrich_error", F.col("metadata_out.errorMessage").cast("string"))
        .withColumn("metadata_enrich_ts", F.current_timestamp())
        .withColumn("metadata_enrich_model", F.lit(metadata_model))
        .select(
            "doc_id", "modificationTime", "page_id",
            "metadata_enrich", "metadata_enrich_error", "metadata_enrich_ts", "metadata_enrich_model"
        )
    )
    
    # Crear temp view y hacer MERGE
    enriched.createOrReplaceTempView("metadata_enrich_updates")
    
    spark.sql(f"""
    MERGE INTO {silver_table} AS t
    USING metadata_enrich_updates AS s
    ON t.doc_id = s.doc_id 
       AND t.modificationTime = s.modificationTime 
       AND t.page_id = s.page_id
    WHEN MATCHED THEN UPDATE SET
        t.metadata_enrich = s.metadata_enrich,
        t.metadata_enrich_error = s.metadata_enrich_error,
        t.metadata_enrich_ts = s.metadata_enrich_ts,
        t.metadata_enrich_model = s.metadata_enrich_model
    """)
    
    print(f"[enrich_metadata] Actualizado metadata_enrich para {cand_count} páginas.")


def enrich_graph_pages(silver_table: str, image_output_path: str, enrich_model: str):
    """
    Actualiza páginas con gráficos agregando descripción enriquecida.
    """
    print("[enrich_graph_pages] Buscando páginas con gráficos sin enrichment...")
    
    # Filtro de gráficos
    txt = F.concat(
        F.lit(" "),
        F.lower(F.coalesce(F.col("page_figures_text"), F.lit(""))),
        F.lit(" ")
    )
    is_graph_like = (
        txt.like("% graph %") | txt.like("% graphs %") |
        txt.like("% bar %") | txt.like("% bars %")
    )
    
    # Leer páginas candidatas
    candidates = (
        spark.table(silver_table)
        .filter(F.col("page_figures_enriched_text").isNull())
        .filter(F.col("page_figure_count") > 1)
        .filter(F.col("page_image_uri").isNotNull())
        .filter(is_graph_like)
        .select("doc_id", "modificationTime", "page_id", "page_image_uri", "page_figures_text")
        .withColumn("image_path_normalized", normalize_path_col(F.col("page_image_uri")))
    )
    
    cand_count = candidates.count()
    if cand_count == 0:
        print("[enrich_graph_pages] No hay páginas candidatas para enrichment.")
        return
    
    print(f"[enrich_graph_pages] Encontradas {cand_count} páginas con gráficos.")
    
    # Cargar imágenes
    imgs = load_images_df(image_output_path)
    
    # Join con imágenes
    cand_with_img = candidates.join(imgs, on="image_path_normalized", how="inner")
    
    matched_count = cand_with_img.count()
    if matched_count == 0:
        print("[enrich_graph_pages] ADVERTENCIA: No se encontraron imágenes que coincidan.")
        return
    
    print(f"[enrich_graph_pages] {matched_count} páginas con imagen encontrada. Invocando LLM...")
    
    # Prompt
    enrich_prompt = F.concat(
        F.lit(
            "Analiza la imagen adjunta (una página de un PDF) y enfócate SOLO en los gráficos (charts/plots).\n"
            "Tu tarea:\n"
            "1) Identificar cada gráfico presente y enumerarlos como 'Gráfico 1', 'Gráfico 2', etc.\n"
            "2) Para cada gráfico, describe: tipo (barras/líneas/etc), qué representa el eje X y el eje Y,\n"
            "   series/leyenda si existen, y una lectura en lenguaje natural.\n"
            "3) Si hay valores numéricos legibles, inclúyelos. Si NO son legibles, indica 'no legible'.\n\n"
            "Contexto (descripción preliminar):\n"
        ),
        F.coalesce(F.col("page_figures_text"), F.lit("")),
        F.lit("\n\nDevuelve SOLO el texto final, en español.")
    )
    
    enriched = (
        cand_with_img
        .withColumn("enrich_prompt", enrich_prompt)
        .withColumn(
            "enrich_out",
            expr(f"""
                ai_query(
                    '{enrich_model}',
                    enrich_prompt,
                    files => image_content,
                    failOnError => false
                )
            """)
        )
        .select(
            "doc_id", "modificationTime", "page_id",
            F.col("enrich_out.result").cast("string").alias("page_figures_enriched_text"),
            F.col("enrich_out.errorMessage").cast("string").alias("page_figures_enriched_error"),
            F.current_timestamp().alias("page_figures_enriched_ts"),
            F.lit(enrich_model).alias("page_figures_enriched_model"),
        )
    )
    
    # Crear temp view y hacer UPDATE
    enriched.createOrReplaceTempView("graph_enriched_updates")
    
    spark.sql(f"""
    MERGE INTO {silver_table} AS t
    USING graph_enriched_updates AS s
    ON t.doc_id = s.doc_id 
       AND t.modificationTime = s.modificationTime 
       AND t.page_id = s.page_id
    WHEN MATCHED THEN UPDATE SET
        t.page_figures_enriched_text = s.page_figures_enriched_text,
        t.page_figures_enriched_error = s.page_figures_enriched_error,
        t.page_figures_enriched_ts = s.page_figures_enriched_ts,
        t.page_figures_enriched_model = s.page_figures_enriched_model
    """)
    
    print(f"[enrich_graph_pages] Actualizado page_figures_enriched para {matched_count} páginas.")


def main():
    args = parse_args()
    
    print(f"[silver_pdf_enrich] silver_table           = {args.silver_table}")
    print(f"[silver_pdf_enrich] image_output_path      = {args.image_output_path}")
    print(f"[silver_pdf_enrich] do_enrich_topic        = {args.do_enrich_topic}")
    print(f"[silver_pdf_enrich] topic_model            = {args.topic_model}")
    print(f"[silver_pdf_enrich] topic_min_chars        = {args.topic_min_chars}")
    print(f"[silver_pdf_enrich] do_enrich_topic_content= {args.do_enrich_topic_content}")
    print(f"[silver_pdf_enrich] topic_content_model    = {args.topic_content_model}")
    print(f"[silver_pdf_enrich] topic_content_min_chars= {args.topic_content_min_chars}")
    print(f"[silver_pdf_enrich] do_enrich_metadata     = {args.do_enrich_metadata}")
    print(f"[silver_pdf_enrich] metadata_model         = {args.metadata_model}")
    print(f"[silver_pdf_enrich] metadata_min_chars     = {args.metadata_min_chars}")
    print(f"[silver_pdf_enrich] do_enrich_graphs       = {args.do_enrich_graphs}")
    print(f"[silver_pdf_enrich] enrich_model           = {args.enrich_model}")
    
    # Enrichments que requieren imágenes
    needs_images = args.do_enrich_topic or args.do_enrich_graphs
    
    if needs_images and (not args.image_output_path or not args.image_output_path.strip()):
        print("[silver_pdf_enrich] ERROR: image_output_path es requerido para topic_llm y graphs enrichment.")
        if not args.do_enrich_topic_content and not args.do_enrich_metadata:
            return
        print("[silver_pdf_enrich] Continuando solo con enrichments que no requieren imágenes...")
    
    # Topic por IMAGEN (visión)
    if args.do_enrich_topic and args.image_output_path:
        enrich_topic_llm(
            args.silver_table,
            args.image_output_path,
            args.topic_model,
            args.topic_min_chars
        )
    
    # Topic por TEXTO (no requiere imágenes)
    if args.do_enrich_topic_content:
        enrich_topic_content(
            args.silver_table,
            args.topic_content_model,
            args.topic_content_min_chars
        )
    
    # Metadata estructurada (no requiere imágenes)
    if args.do_enrich_metadata:
        enrich_metadata(
            args.silver_table,
            args.metadata_model,
            args.metadata_min_chars
        )
    
    # Gráficos (visión)
    if args.do_enrich_graphs and args.image_output_path:
        enrich_graph_pages(
            args.silver_table,
            args.image_output_path,
            args.enrich_model
        )
    
    print("[silver_pdf_enrich] Enrichment completo.")


if __name__ == "__main__":
    main()
