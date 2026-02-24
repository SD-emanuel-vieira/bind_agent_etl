# src/bronze_excel_ingest.py
"""
Bronze Excel Ingest - Ingesta de archivos Excel (.xlsx)

Lee archivos Excel desde un Volume, extrae la hoja "BASE" y guarda
cada fila como un registro en la tabla Bronze.

Solo procesa archivos cuyo nombre contiene una fecha YYYYMMDD antes de .xlsx.
Patrón similar a bronze_pdf_ingest.py pero adaptado para datos tabulares.
"""
import argparse
from datetime import datetime
from io import BytesIO

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, DateType, TimestampType, LongType
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)        # e.g. dbfs:/Volumes/<cat>/<schema>/excel/
    p.add_argument("--checkpoint_path", required=True)   # e.g. dbfs:/Volumes/<cat>/<schema>/docs_state/.../checkpoints/
    p.add_argument("--schema_location", required=True)   # e.g. dbfs:/Volumes/<cat>/<schema>/docs_state/.../schema/
    p.add_argument("--bronze_table", required=True)      # e.g. catalog.schema.excel_bronze
    p.add_argument("--sheet_name", default="BASE")       # Nombre de la hoja a leer
    return p.parse_args()


# Schema de la tabla Bronze para Excel
BRONZE_SCHEMA = StructType([
    # Metadata del archivo
    StructField("doc_id", StringType(), False),
    StructField("path", StringType(), True),
    StructField("modificationTime", TimestampType(), True),
    StructField("file_size", LongType(), True),
    StructField("sheet_name", StringType(), True),
    StructField("row_index", IntegerType(), True),
    StructField("ingest_ts", TimestampType(), True),
    
    # Columnas de la hoja BASE
    StructField("Ano", IntegerType(), True),  # Año → Ano (evitar ñ en nombres de columna)
    StructField("Mes", IntegerType(), True),
    StructField("Fecha", DateType(), True),
    StructField("Cuenta_BT", StringType(), True),
    StructField("CUIT", StringType(), True),
    StructField("Cliente", StringType(), True),
    StructField("Producto", StringType(), True),
    StructField("Sub_producto", StringType(), True),
    StructField("Moneda", StringType(), True),
    StructField("FLAG_Remunerada", StringType(), True),
    StructField("Volumen_Promedio", DoubleType(), True),
    StructField("Tasa_Activa", DoubleType(), True),
    StructField("TT", DoubleType(), True),
    StructField("Interes_Cobrado", DoubleType(), True),
    StructField("Interes_Pagado", DoubleType(), True),
    StructField("Resultado_Neto_IIBB", DoubleType(), True),
    StructField("IIBB_SEDESA", DoubleType(), True),
    StructField("Resultado_Bruto", DoubleType(), True),
    StructField("Oficial", StringType(), True),
    StructField("Banca", StringType(), True),
])

# Mapeo de nombres de columnas del Excel a nombres normalizados
COLUMN_MAPPING = {
    "Año": "Ano",
    "Mes": "Mes",
    "Fecha": "Fecha",
    "Cuenta_BT": "Cuenta_BT",
    "CUIT": "CUIT",
    "Cliente": "Cliente",
    "Producto": "Producto",
    "Sub_producto": "Sub_producto",
    "Moneda": "Moneda",
    "FLAG_Remunerada": "FLAG_Remunerada",
    "Volumen Promedio": "Volumen_Promedio",
    "Tasa Activa": "Tasa_Activa",
    "TT": "TT",
    "Interes_Cobrado": "Interes_Cobrado",
    "Interes_Pagado": "Interes_Pagado",
    "Resultado_Neto_IIBB": "Resultado_Neto_IIBB",
    "IIBB / SEDESA": "IIBB_SEDESA",
    "Resultado_Bruto": "Resultado_Bruto",
    "Oficial": "Oficial",
    "Banca": "Banca",
}


def ensure_bronze_table(bronze_table: str):
    """
    Crea la tabla Bronze para Excel si no existe.
    """
    parts = bronze_table.split(".")
    if len(parts) != 3:
        raise ValueError(f"bronze_table debe ser catalog.schema.table. Recibido: {bronze_table}")

    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {bronze_table} (
      -- Metadata del archivo
      doc_id             STRING NOT NULL,
      path               STRING,
      modificationTime   TIMESTAMP,
      file_size          BIGINT,
      sheet_name         STRING,
      row_index          INT,
      ingest_ts          TIMESTAMP,
      
      -- Columnas de la hoja BASE
      Ano                INT,
      Mes                INT,
      Fecha              DATE,
      Cuenta_BT          STRING,
      CUIT               STRING,
      Cliente            STRING,
      Producto           STRING,
      Sub_producto       STRING,
      Moneda             STRING,
      FLAG_Remunerada    STRING,
      Volumen_Promedio   DOUBLE,
      Tasa_Activa        DOUBLE,
      TT                 DOUBLE,
      Interes_Cobrado    DOUBLE,
      Interes_Pagado     DOUBLE,
      Resultado_Neto_IIBB DOUBLE,
      IIBB_SEDESA        DOUBLE,
      Resultado_Bruto    DOUBLE,
      Oficial            STRING,
      Banca              STRING
    )
    USING DELTA
    """)


def parse_excel_file(content: bytes, sheet_name: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(
            BytesIO(content),
            sheet_name=sheet_name,
            dtype=str,
            header=1,              # <-- header en la segunda fila (fila 2 en Excel)
            engine="openpyxl"
        )

        # Normalizar nombres de columnas (por si hay espacios, saltos de línea, etc.)
        df.columns = (
            df.columns.astype(str)
            .str.replace("\n", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # Renombrar columnas según el mapeo
        df = df.rename(columns=COLUMN_MAPPING)

        # (Opcional) eliminar filas completamente vacías
        df = df.dropna(how="all")

        # (Recomendado) validar que quedaron las columnas esperadas
        expected = set(COLUMN_MAPPING.values())
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(
                f"Columnas esperadas faltantes: {sorted(missing)}. Columnas leídas: {list(df.columns)}"
            )

        return df

    except Exception as e:
        print(f"[bronze_excel_ingest] Error parseando Excel: {e}")
        return pd.DataFrame()



def process_excel_batch(batch_df, batch_id, bronze_table: str, sheet_name: str):
    """
    Procesa cada batch de archivos Excel detectados por Auto Loader.
    Lee la hoja especificada y escribe las filas a la tabla Bronze.
    """
    if batch_df.isEmpty():
        print(f"[bronze_excel_ingest] Batch {batch_id}: vacío, saltando.")
        return
    
    rows = batch_df.collect()
    print(f"[bronze_excel_ingest] Batch {batch_id}: procesando {len(rows)} archivo(s)")
    
    all_records = []
    ingest_ts = datetime.now()
    
    for row in rows:
        content = bytes(row.content)
        path = row.path
        mod_time = row.modificationTime
        file_size = row.length
        doc_id = row.doc_id
        
        print(f"[bronze_excel_ingest] Procesando: {path}")
        
        # Parsear Excel
        df_excel = parse_excel_file(content, sheet_name)
        
        if df_excel.empty:
            print(f"[bronze_excel_ingest] Archivo vacío o error en: {path}")
            continue
        
        # Convertir cada fila a un registro
        for idx, excel_row in df_excel.iterrows():
            record = {
                "doc_id": doc_id,
                "path": path,
                "modificationTime": mod_time,
                "file_size": file_size,
                "sheet_name": sheet_name,
                "row_index": int(idx),
                "ingest_ts": ingest_ts,
            }
            
            # Agregar columnas de datos
            for col in COLUMN_MAPPING.values():
                value = excel_row.get(col)
                # Convertir NaN a None
                if pd.isna(value):
                    record[col] = None
                else:
                    record[col] = value
            
            all_records.append(record)
    
    if not all_records:
        print(f"[bronze_excel_ingest] Batch {batch_id}: sin registros para escribir.")
        return
    
    # Crear DataFrame de Spark y escribir
    result_pdf = pd.DataFrame(all_records)
    
    # Convertir tipos de datos
    result_pdf["Ano"] = pd.to_numeric(result_pdf["Ano"], errors="coerce").astype("Int64")
    result_pdf["Mes"] = pd.to_numeric(result_pdf["Mes"], errors="coerce").astype("Int64")
    result_pdf["Fecha"] = pd.to_datetime(result_pdf["Fecha"], errors="coerce")
    
    numeric_cols = [
        "Volumen_Promedio", "Tasa_Activa", "TT", "Interes_Cobrado", 
        "Interes_Pagado", "Resultado_Neto_IIBB", "IIBB_SEDESA", "Resultado_Bruto"
    ]
    for col in numeric_cols:
        result_pdf[col] = pd.to_numeric(result_pdf[col], errors="coerce")
    
    # Crear DataFrame de Spark
    spark_df = spark.createDataFrame(result_pdf, schema=BRONZE_SCHEMA)
    
    # Escribir a la tabla Bronze
    spark_df.write.mode("append").saveAsTable(bronze_table)
    
    print(f"[bronze_excel_ingest] Batch {batch_id}: {len(all_records)} filas escritas a {bronze_table}")


def main():
    args = parse_args()

    print(f"[bronze_excel_ingest] input_path      = {args.input_path}")
    print(f"[bronze_excel_ingest] schema_location = {args.schema_location}")
    print(f"[bronze_excel_ingest] checkpoint_path = {args.checkpoint_path}")
    print(f"[bronze_excel_ingest] bronze_table    = {args.bronze_table}")
    print(f"[bronze_excel_ingest] sheet_name      = {args.sheet_name}")

    ensure_bronze_table(args.bronze_table)

    # Auto Loader para detectar archivos Excel nuevos
    # Solo procesa archivos cuyo nombre contiene fecha YYYYMMDD
    df = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("cloudFiles.schemaLocation", args.schema_location)
        .option("cloudFiles.useManagedFileEvents", "false")
        .option("pathGlobFilter", "*.xlsx")
        .load(args.input_path)
        .withColumn("doc_id", F.sha2(F.col("path"), 256))
        # --- FILTRO: solo archivos con fecha YYYYMMDD en el nombre ---
        .withColumn("_filename", F.element_at(F.split(F.col("path"), "/"), -1))
        .filter(
            F.regexp_extract(F.col("_filename"), r"(\d{8})\.xlsx$", 1) != ""
        )
        .drop("_filename")
    )

    # Procesar cada batch de archivos
    (
        df.writeStream
        .option("checkpointLocation", args.checkpoint_path)
        .trigger(availableNow=True)
        .foreachBatch(
            lambda batch_df, batch_id: process_excel_batch(
                batch_df, batch_id, args.bronze_table, args.sheet_name
            )
        )
        .start()
        .awaitTermination()
    )

    print("[bronze_excel_ingest] Ingesta completada.")


if __name__ == "__main__":
    main()