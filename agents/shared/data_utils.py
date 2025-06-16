import logging
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, BadRequest
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# Import monitoring conditionally
try:
    from utils.monitoring import monitoring
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    monitoring = None


def execute_sql_file(sql_path, params, project_id):
    logger.info(f"Executing SQL file: {sql_path}")
    try:
        with open(sql_path, "r") as file:
            sql = file.read()

        logger.debug(f"SQL parameters: {params}")

        sql = sql.format(**params)
        client = bigquery.Client(project=project_id)
        query_job = client.query(sql)

        result = query_job.result()
        logger.info(f"SQL execution successful for {sql_path}")
        return result
        
    except (NotFound, BadRequest) as e:
        logger.warning(f"BigQuery execution error for {sql_path}: {e}")
        if MONITORING_AVAILABLE:
            monitoring.log_error("bigquery_not_found", str(e), {"sql_path": sql_path})
        return None  # Let caller handle the fallback logic

    except Exception as e:
        logger.error(f"Unexpected error during BigQuery query execution for {sql_path}: {e}", exc_info=True)
        if MONITORING_AVAILABLE:
            monitoring.log_error("bigquery_execution_error", str(e), {"sql_path": sql_path})
        return None


def upload_parquet_to_bigquery(project_id, dataset_id, table_name, parquet_path):
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_id}.{table_name}"
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    with open(parquet_path, "rb") as file:
        load_job = client.load_table_from_file(file, table_id, job_config=job_config)
    load_job.result()


def normalize_value(field: str, value: int) -> int:
    """Normalize special codes 98/99 based on context."""
    if field == "emergency_savings" and value in (99,):
        return 2  # Treat as 'No'
    elif field == "retirement_planning" and value in (98, 99):
        return 3  # Treat as 'Unknown'
    elif field in {"age", "education", "income"} and value in (98, 99):
        return -1  # Median
    return value
