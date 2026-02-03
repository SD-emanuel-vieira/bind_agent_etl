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

    # 1) Identificar archivo y source_type
    # path puede venir como /Volumes/... o dbfs:/Volumes/... -> split por "/"
    df = (
        bronze_df
        .withColumn("filename", F.element_at(F.split(F.col("path"), "/"), -1))
        .withColumn(
            "source_type",
            F.when(F.col("filename") == F.lit("Matriz_EMP.xlsx"), F.lit("Empresas"))
             .when(F.col("filename") == F.lit("Matriz_CORPO_INST.xlsx"), F.lit("Corporate & Institucional"))
             .otherwise(F.lit(None))
        )
        .filter(F.col("source_type").isNotNull())
    )

    # 2) Quedarse con el último archivo ingresado por cada caso (por ingest_ts)
    w = Window.partitionBy("source_type")
    df = (
        df.withColumn("latest_mod_time", F.max("modificationTime").over(w))
            .filter(F.col("modificationTime") == F.col("latest_mod_time"))
            .drop("latest_mod_time")
    )

    # 3) Redondear DOUBLE -> entero (BIGINT)
    for c in DOUBLE_COLS:
        df = df.withColumn(c, F.round(F.col(c), 0).cast("bigint"))

    # 4) Selección final con nombres en minúsculas + renombres Ano/Mes/Fecha
    #    Aplicamos normalize_str() a todos los campos string:
    #    trim + lower + regexp_replace para espacios múltiples
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

    # Dedupe útil por si reingestaste con otro checkpoint:
    # row_index se reinicia por archivo, por eso incluyo source_type
    silver_df = silver_df.dropDuplicates(["source_type", "sheet_name", "row_index"])

    # 5) Overwrite completo de la tabla Silver
    (
        silver_df.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(args.silver_table)
    )

    print(f"[silver_excel_transform] OK -> wrote {args.silver_table}")

if __name__ == "__main__":
    main()
