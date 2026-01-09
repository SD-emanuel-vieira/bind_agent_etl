# src/vector_search_index.py
import argparse
import time
from typing import Any, Optional, Dict

from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # Vector Search
    p.add_argument("--vs_endpoint", required=True, help="Nombre del Vector Search endpoint (ej: bind_agent_vs)")
    p.add_argument(
        "--vs_index_full_name",
        required=True,
        help="Nombre UC completo del índice (catalog.schema.index). Ej: bind_agent.docs.pdf_chunks_vs_index",
    )

    # Delta source
    p.add_argument(
        "--source_table",
        required=True,
        help="Tabla UC fuente (Delta) con embeddings. Ej: bind_agent.docs.pdf_chunks_embeddings",
    )
    p.add_argument("--primary_key", required=True, help="Columna PK única en source_table (ej: chunk_id)")
    p.add_argument("--embedding_column", required=True, help="Columna del vector embedding (ARRAY<FLOAT>), ej: embedding")

    # Delta Sync settings
    p.add_argument(
        "--pipeline_type",
        default="TRIGGERED",
        choices=["TRIGGERED", "CONTINUOUS"],
        help="TRIGGERED (sync manual) o CONTINUOUS (auto-sync).",
    )

    # Optional: dimension (si omitís, se infiere)
    p.add_argument("--embedding_dimension", type=int, default=None)

    # Robustness / waits
    p.add_argument("--poll_seconds", type=int, default=15)
    p.add_argument("--endpoint_timeout_seconds", type=int, default=1200)  # 20 min
    p.add_argument("--index_timeout_seconds", type=int, default=1800)     # 30 min

    # CDF (recomendado para delta sync)
    p.add_argument("--ensure_cdf", action="store_true", help="Habilita CDF en source_table (idempotente).")

    # Para TRIGGERED: ejecutar sync() luego de crear/verificar index
    p.add_argument("--trigger_sync", action="store_true", help="Si pipeline_type=TRIGGERED, dispara sync()")

    return p.parse_args()


# -------------------------
# Helpers
# -------------------------
def ensure_cdf_enabled(table_name: str) -> None:
    spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"[vector_search] CDF asegurado en: {table_name}")


def _normalize_state(x: Any) -> str:
    """
    Normaliza estados que pueden venir como:
      - "ONLINE"
      - {"STATE": "ONLINE"}
      - {"state": "ONLINE"}
      - {"detailed_state": "ONLINE"}
      - {"status": {"detailed_state": "ONLINE"}}
    """
    if x is None:
        return "UNKNOWN"

    if isinstance(x, str):
        return x.strip().upper()

    if isinstance(x, dict):
        # claves comunes
        for k in ("STATE", "state", "detailed_state", "status", "endpoint_status"):
            if k in x:
                return _normalize_state(x.get(k))
        # si no encontramos nada, string del dict
        return str(x).upper()

    return str(x).upper()


def _get_endpoint_record(vsc: VectorSearchClient, endpoint_name: str) -> Optional[Dict[str, Any]]:
    eps = vsc.list_endpoints().get("endpoints", []) or []
    return next((e for e in eps if e.get("name") == endpoint_name), None)


def get_endpoint_state(vsc: VectorSearchClient, endpoint_name: str) -> str:
    ep = _get_endpoint_record(vsc, endpoint_name)
    if not ep:
        return "NOT_FOUND"
    # según versión, el estado puede venir en diferentes campos
    candidate = ep.get("state") or ep.get("status") or ep.get("endpoint_status") or ep
    return _normalize_state(candidate)


def ensure_vs_endpoint(vsc: VectorSearchClient, endpoint_name: str) -> None:
    state = get_endpoint_state(vsc, endpoint_name)
    if state != "NOT_FOUND":
        print(f"[vector_search] Endpoint ya existe: {endpoint_name} (state={state})")
        return

    print(f"[vector_search] Creando endpoint: {endpoint_name}")
    vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
    print("[vector_search] create_endpoint enviado (provisioning puede tardar varios minutos).")


def wait_endpoint_ready(vsc: VectorSearchClient, endpoint_name: str, poll_seconds: int, timeout_seconds: int) -> None:
    print(f"[vector_search] Esperando endpoint ONLINE/READY (timeout={timeout_seconds}s)...")
    t0 = time.time()
    while True:
        state = get_endpoint_state(vsc, endpoint_name)
        print(f"[vector_search] endpoint_state={state}")

        if state in ("ONLINE", "READY"):
            print("[vector_search] Endpoint ONLINE/READY.")
            return

        if time.time() - t0 > timeout_seconds:
            raise TimeoutError(f"Timeout esperando endpoint {endpoint_name} ONLINE/READY. Último state={state}")

        time.sleep(poll_seconds)


def index_exists(vsc: VectorSearchClient, endpoint_name: str, index_full_name: str) -> bool:
    try:
        vsc.get_index(endpoint_name=endpoint_name, index_name=index_full_name)
        return True
    except Exception:
        return False


def infer_embedding_dimension(source_table: str, embedding_column: str) -> int:
    df = (
        spark.table(source_table)
        .select(F.size(F.col(embedding_column)).alias("dim"))
        .where(F.col(embedding_column).isNotNull())
        .limit(1)
    )
    rows = df.collect()
    if not rows or rows[0]["dim"] is None:
        raise ValueError(f"No pude inferir embedding_dimension: no hay embeddings no nulos en {source_table}.{embedding_column}")
    return int(rows[0]["dim"])


def ensure_delta_sync_index(
    vsc: VectorSearchClient,
    endpoint_name: str,
    index_full_name: str,
    source_table: str,
    primary_key: str,
    embedding_column: str,
    pipeline_type: str,
    embedding_dimension: Optional[int],
) -> None:
    if index_exists(vsc, endpoint_name, index_full_name):
        print(f"[vector_search] Índice ya existe: {index_full_name}")
        return

    if embedding_dimension is None:
        embedding_dimension = infer_embedding_dimension(source_table, embedding_column)

    print("[vector_search] Creando Delta Sync Index:")
    print(f"  endpoint       = {endpoint_name}")
    print(f"  index_name     = {index_full_name}")
    print(f"  source_table   = {source_table}")
    print(f"  primary_key    = {primary_key}")
    print(f"  embedding_col  = {embedding_column}")
    print(f"  dimension      = {embedding_dimension}")
    print(f"  pipeline_type  = {pipeline_type}")

    vsc.create_delta_sync_index(
        endpoint_name=endpoint_name,
        index_name=index_full_name,
        source_table_name=source_table,
        pipeline_type=pipeline_type,             # <-- requerido por tu SDK
        primary_key=primary_key,
        embedding_dimension=embedding_dimension,
        embedding_vector_column=embedding_column,
    )

    print(f"[vector_search] Índice creado: {index_full_name}")


def get_index_state(vsc: VectorSearchClient, endpoint_name: str, index_full_name: str) -> str:
    idx = vsc.get_index(endpoint_name=endpoint_name, index_name=index_full_name)
    desc = idx.describe()
    # status suele estar acá:
    status = desc.get("status") or desc.get("index_status") or desc
    return _normalize_state(status)


def wait_index_online(
    vsc: VectorSearchClient,
    endpoint_name: str,
    index_full_name: str,
    poll_seconds: int,
    timeout_seconds: int
) -> None:
    print(f"[vector_search] Esperando index ONLINE (timeout={timeout_seconds}s)...")
    t0 = time.time()

    # Estados "buenos" para Delta Sync index
    OK_STATES = {
        "ONLINE",
        "READY",
        "ONLINE_NO_PENDING_UPDATE",
        "ONLINE_TRIGGERED_UPDATE",   # si está actualizando, igual está usable
    }

    # Estados "malos" (si los ves, mejor fallar rápido)
    FAIL_STATES = {
        "FAILED",
        "ERROR",
        "OFFLINE",
        "DELETED",
    }

    while True:
        state = get_index_state(vsc, endpoint_name, index_full_name)
        print(f"[vector_search] index_state={state}")

        if state in OK_STATES:
            print("[vector_search] Index ONLINE/READY (OK).")
            return

        if state in FAIL_STATES:
            raise RuntimeError(f"Index {index_full_name} en estado de error: {state}")

        if time.time() - t0 > timeout_seconds:
            raise TimeoutError(
                f"Timeout esperando index {index_full_name} ONLINE/READY. Último state={state}"
            )

        time.sleep(poll_seconds)

def trigger_index_sync_if_supported(vsc: VectorSearchClient, endpoint_name: str, index_full_name: str) -> None:
    """
    En TRIGGERED, disparar sync para incorporar cambios del CDF.
    El método puede variar por versión; lo manejamos con fallbacks.
    """
    idx = vsc.get_index(endpoint_name=endpoint_name, index_name=index_full_name)

    if hasattr(idx, "sync"):
        print("[vector_search] Disparando sync() (TRIGGERED)...")
        idx.sync()
        return

    # Fallbacks por si cambia el SDK
    if hasattr(idx, "trigger_sync"):
        print("[vector_search] Disparando trigger_sync() (TRIGGERED)...")
        idx.trigger_sync()
        return

    print("[vector_search] WARN: No encontré método sync()/trigger_sync() en este SDK. Omitiendo sync TRIGGERED.")


def main():
    args = parse_args()

    vsc = VectorSearchClient()

    if args.ensure_cdf:
        ensure_cdf_enabled(args.source_table)

    ensure_vs_endpoint(vsc, args.vs_endpoint)
    wait_endpoint_ready(
        vsc=vsc,
        endpoint_name=args.vs_endpoint,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.endpoint_timeout_seconds,
    )

    ensure_delta_sync_index(
        vsc=vsc,
        endpoint_name=args.vs_endpoint,
        index_full_name=args.vs_index_full_name,
        source_table=args.source_table,
        primary_key=args.primary_key,
        embedding_column=args.embedding_column,
        pipeline_type=args.pipeline_type,
        embedding_dimension=args.embedding_dimension,
    )

    wait_index_online(
        vsc=vsc,
        endpoint_name=args.vs_endpoint,
        index_full_name=args.vs_index_full_name,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.index_timeout_seconds,
    )

    if args.pipeline_type == "TRIGGERED" and args.trigger_sync:
        trigger_index_sync_if_supported(vsc, args.vs_endpoint, args.vs_index_full_name)
        # opcional: esperar un toque a que termine el ciclo de sync
        wait_index_online(
            vsc=vsc,
            endpoint_name=args.vs_endpoint,
            index_full_name=args.vs_index_full_name,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.index_timeout_seconds,
        )

    print("[vector_search] OK.")


if __name__ == "__main__":
    main()
