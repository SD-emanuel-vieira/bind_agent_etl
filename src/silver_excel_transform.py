# src/silver_excel_transform.py
import argparse
from pyspark.sql import functions as F
from pyspark.sql.window import Window

DOUBLE_COLS = [
    "Volumen_Promedio",
    "Tasa_Activa",
    "TT",
    "Interes_Cobrado",
    "Interes_Pagado",
    "Resultado_Neto_IIBB",
    "IIBB_SEDESA",
    "Resultado_Bruto",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bronze_table", required=True)
    p.add_argument("--silver_table", required=True)
    return p.parse_args()

def main():
    args = parse_args()

    bronze_df = spark.table(args.bronze_table)

    # 1) Extraer filename y determinar source_type con CONTAINS (no match exacto)
    df = (
        bronze_df
        .withColumn("filename", F.element_at(F.split(F.col("path"), "/"), -1))
        .withColumn(
            "source_type",
            F.when(F.col("filename").like("Matriz_EMP_%"), F.lit("Empresas"))
             .when(F.col("filename").like("Matriz_CORPO_INST_%"), F.lit("Corporate & Institucional"))
             .otherwise(F.lit(None))
        )
        .filter(F.col("source_type").isNotNull())
    )

    # 2) Extraer la fecha YYYYMMDD del nombre del archivo y derivar el año
    #    Solo conservar archivos que tengan fecha válida en el nombre
    df = (
        df
        .withColumn(
            "file_date_str",
            F.regexp_extract(F.col("filename"), r"(\d{8})\.xlsx$", 1)
        )
        .filter(F.col("file_date_str") != "")  # descartar archivos sin fecha
        .withColumn(
            "file_date",
            F.to_date(F.col("file_date_str"), "yyyyMMdd")
        )
        .filter(F.col("file_date").isNotNull())  # descartar fechas inválidas
        .withColumn("file_year", F.year(F.col("file_date")))
    )

    # 3) Por cada (source_type, file_year), quedarse con el archivo más reciente
    #    Usamos file_date (del nombre) como criterio de recencia, no modificationTime
    w = Window.partitionBy("source_type", "file_year")
    df = (
        df
        .withColumn("latest_file_date", F.max("file_date").over(w))
        .filter(F.col("file_date") == F.col("latest_file_date"))
        .drop("latest_file_date")
    )

    # 4) Redondear DOUBLE -> entero (BIGINT)
    for c in DOUBLE_COLS:
        df = df.withColumn(c, F.round(F.col(c), 0).cast("bigint"))

    # 5) Selección final con nombres en minúsculas + normalización de strings
    def normalize_str(col_name):
        """trim + lower + colapsar espacios múltiples a uno solo"""
        return F.regexp_replace(F.lower(F.trim(F.col(col_name))), r'\s+', ' ')

    silver_df = df.select(
        F.col("doc_id").alias("doc_id"),
        F.col("path").alias("path"),
        F.col("modificationTime").alias("modification_time"),
        F.col("file_size").alias("file_size"),
        F.col("sheet_name").alias("sheet_name"),
        F.col("row_index").alias("row_index"),
        F.col("ingest_ts").alias("ingest_ts"),

        normalize_str("source_type").alias("source_type"),

        # Metadata extraída del filename
        F.col("file_date").alias("file_date"),
        F.col("file_year").alias("file_year"),

        F.col("Ano").cast("int").alias("year"),
        F.col("Mes").cast("int").alias("month"),
        F.col("Fecha").cast("date").alias("date"),

        normalize_str("Cuenta_BT").alias("cuenta_bt"),
        normalize_str("CUIT").alias("cuit"),
        normalize_str("Cliente").alias("cliente"),
        normalize_str("Producto").alias("producto"),
        normalize_str("Sub_producto").alias("sub_producto"),
        normalize_str("Moneda").alias("moneda"),
        normalize_str("FLAG_Remunerada").alias("flag_remunerada"),

        F.col("Volumen_Promedio").alias("volumen_promedio"),
        F.col("Tasa_Activa").alias("tasa_activa"),
        F.col("TT").alias("tt"),
        F.col("Interes_Cobrado").alias("interes_cobrado"),
        F.col("Interes_Pagado").alias("interes_pagado"),
        F.col("Resultado_Neto_IIBB").alias("resultado_neto_iibb"),
        F.col("IIBB_SEDESA").alias("iibb_sedesa"),
        F.col("Resultado_Bruto").alias("resultado_bruto"),

        normalize_str("Oficial").alias("oficial"),
        normalize_str("Banca").alias("banca"),
    )

    # Dedupe por source_type + sheet_name + row_index + file_date
    silver_df = silver_df.dropDuplicates(["source_type", "sheet_name", "row_index", "file_date"])

    # 6) Overwrite completo de la tabla Silver
    (
        silver_df.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(args.silver_table)
    )

    print(f"[silver_excel_transform] OK -> wrote {args.silver_table}")

if __name__ == "__main__":
    main()